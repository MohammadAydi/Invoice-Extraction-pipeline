# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The Python half of a handwritten-Arabic invoice digitization system. It takes an invoice
photo and returns a structured invoice — merchant, invoice number, date, city, total, and
the line-items table — with every value carrying a confidence score and a ranked list of
catalog matches for the user to confirm. The system is completely offline (invoices never
leave the machine) and must run acceptably on low-end laptops with no GPU.

This repo is one of three components:

1. **This repo** — Python AI pipeline: OpenCV preprocessing, OCR, table extraction, fuzzy
   matching. Exposed over local HTTP.
2. **`InvoiceDigitizationApp`** (separate repo) — WinUI 3 desktop app in C#, MVVM. Displays
   the corrected image on the left and extracted label/value + table data on the right,
   with clickable regions on the image tied to bounding boxes from this pipeline's output.
   Owns the SQLite database (`Microsoft.Data.Sqlite`), the job queue for batch processing,
   and a mobile-upload companion feature.
3. A mobile app that lets users photograph invoices and send them to the desktop app.

Two documents are the binding contracts with the C# side — **read them before touching
anything that crosses the boundary**. Both now live in the repo's shared `docs/` folder,
beside the database schema; the files of the same names in this directory are three-line
pointers to them:

- [../docs/api-contract.md](../docs/api-contract.md) — the `/extract` request/response JSON
  shape. The request is **three** multipart parts (`file`, `options`, `config`) and the
  response has one named key per invoice field.
- [../docs/settings-config-contract.md](../docs/settings-config-contract.md) — the
  object-keyed `config` part the desktop settings page sends, and how it's assembled per
  preprocessing step.

Changing either side (Pydantic schemas here, C# DTOs there) without updating the other and
this document breaks the system.

## Commands

```bash
# Setup
python -m venv .venv
.venv/Scripts/activate                    # Windows
pip install -r requirements.txt

# Run the HTTP service the desktop app talks to
python -m api.main                        # http://127.0.0.1:8765

# Run one image from the CLI (same PipelineOrchestrator, same config, no HTTP)
python main.py path/to/invoice.jpg
python main.py path/to/invoice.jpg config/qwen_config.yaml
python run_batch.py bills/ --config config/qwen_config.yaml

# Tests
pytest
pytest tests/test_invoice_parser.py               # one file
pytest tests/test_invoice_parser.py -k some_test   # one test
```

`pytest.ini` puts the project root on `sys.path` (`pythonpath = .`), so packages import as
top-level names (`config`, `core`, `ocr`, ...) — no `src/` layer, no installed package.

`tests/conftest.py` autouse-fixtures persistence off and redirects it to a tmp dir for the
whole session, so running the suite never writes into `results/`.

The `stub` OCR engine (`ocr.engine: stub` — deterministic, fabricates output from an image
hash) is what `test_api.py` uses to exercise the full pipeline end-to-end over HTTP without
any real OCR installed. Reach for it when you need a pipeline run in a test and don't care
about actual recognition.

## Architecture

### The one decision that shapes everything: geometric correction runs once

Only two preprocessing steps change pixel coordinates: **perspective correction** and
**deskew**. Everything else (channel selection, illumination normalization, CLAHE,
bilateral filtering, thresholding, morphology) is purely photometric.

Because of that, geometric correction is factored into a single upstream stage
(`preprocessing.geometric_steps`) that runs **once**, before the pipeline forks into two
photometric branches — `ocr_photometric_steps` and `table_photometric_steps` — both
operating on that same geometrically corrected image. Every OCR fragment and every table
cell produced downstream therefore lives in exactly one coordinate space: the geometrically
corrected image, which is also the exact image the desktop UI displays on both sides (plain
on the left, with clickable overlays on the right). No bounding box is ever remapped
between spaces. See `core/domain/geometry.py` for the full reasoning, and
`orchestration/pipeline_orchestrator.py` for where the fork happens.

### Pipeline shape (`orchestration/pipeline_orchestrator.py`)

`PipelineOrchestrator` is the **only** module that knows the full pipeline end to end —
every other module knows only its own interface. `PipelineOrchestrator.run()`:

1. Runs `geometric_pipeline` once → the canonical display image.
2. Runs `ocr_photometric_pipeline` and `table_photometric_pipeline` on that image. Each
   `PreprocessingPipeline.run()` leaves its input payload untouched, which is what makes
   this a fork rather than a chain — see Traps.
3. `flow.run()` — layout analysis, recognition and mapping, sequenced by the configured
   flow (`orchestration/flows/`). This is the only part of the pipeline the three reading
   strategies disagree on.
4. `string_matcher.match()` — generic keyword-dictionary correction, run against every
   element (table cell or free field alike) using the shared keyword dictionary.
5. `InvoiceParser.parse()` — reads the elements as an invoice: header fields (merchant,
   invoice number, date, city, total) + line items + warnings, matching merchant/city/product
   names against the **request's** catalogs (`known_merchants`/`known_products`/`known_cities`
   — these come from the request, not from config, because they're the C# side's live
   Customers/Products tables and the distinct `Customers.City` values).
6. `invoice.reconciliation.reconcile()` — arithmetic repair of the line items. The
   recognizers misplace the decimal separator far more often than they misread a digit, so
   every legal separator position is tried and the one satisfying `price × qty = total` is
   kept; a row with two readings and one blank has the third derived. A row the equation
   cannot settle is left exactly as it was read. It runs on the parsed draft rather than
   inside the parser because it is a property of the three numbers **together**.
7. Assembles a `PipelineResult` (with a full `config_snapshot` for audit/reproducibility),
   persists it via `ResultStore`, hands it to `output_formatter.format()`.

Any stage that raises `NotImplementedStrategyError` (a registered-but-unbuilt strategy) is
caught by `_run_optional_stage` and substituted with an empty fallback of the right type --
a missing piece degrades that stage, it never crashes a whole run. It catches *only* that
exception, not `NotImplementedError` in general: a plain one escaping from working code is
a bug, and swallowing it turned that bug into a silently blank page.

### The three flows (`orchestration/flows/`)

How a page gets read is a **Template Method**. `ExtractionFlow.run()` fixes the sequence --
`_analyze_layout` → `_read_text` → `_assemble` — and is never overridden; three subclasses
fill the hooks. Selected by `flow.name` in config.

| Flow | Reads by | Notes |
|---|---|---|
| `single_engine` | one `OCREngine` over the whole page | **Default.** `stub`, `tesseract`, `surya`, `easyocr`, `qwen_vlm`. |
| `detector_driven` | detector → refiner → cropper → recognizer | What `surya_qwen` used to be, decomposed. Reads text outside any ruled box; needs column boundaries declared as page-width fractions; misses short isolated digits the detector never boxes. |
| `layout_driven` | the printed grid → cropper → recognizer | Each classified cell is read with the prompt its **role** implies, so the grid supplies real column boundaries and nothing is declared. **No detector is constructed at all.** Text outside a ruled box is not read; an unruled page returns empty plus a `NO_LAYOUT_DETECTED` warning rather than silently falling back to another strategy. |

Each flow declares its expensive collaborators in `REQUIRES`, and `flows/components.py`
builds exactly those. That is both why adding a fourth flow edits no existing file, and the
mechanism — not merely the intent — behind "layout_driven does not use the detector".

### Layout: where the boxes are, then what they mean

Two separately registered families, because they change for different reasons.

* **`table_extraction/extractors/`** — "where are the ruled boxes on this photograph", the
  same computer-vision problem for every form. `contour_based` takes the closed regions
  directly and carries a stack filter that tells several lines of stacked handwriting apart
  from a set of rules; `grid_line` reconstructs the grid from its lines and recovers exact
  topology when the rules are clean.
* **`table_extraction/classifiers/`** — "the box above the table on the left is the invoice
  number", knowledge of one supplier's printed form. `bill_layout` labels the standard
  Arabic sales-invoice form; `passthrough` (the default) labels nothing, which is exactly
  the behaviour that existed before classification.

Together they produce an `InvoiceLayout` of `LayoutRegion`s, each carrying a `CellRole`
(which field), a `Zone` (header/table/footer) and a `ContentKind` (what kind of characters,
which selects the recognition prompt). `ContentKind` is the seam that keeps the recognizer
generic: it knows about digits and Arabic text, never about invoices.

### Strategy + Registry pattern (how every family of algorithms is pluggable)

`core/registry.py` defines a generic `Registry[T]`: concrete implementations self-register
via `@registry.register("name")` at import time, and a `factory.py` builds instances purely
from the string name found in config. This is the backbone of every swappable family:
preprocessing steps, OCR engines, table extractors, string matchers, output formatters. It's
Strategy + Factory, applied uniformly, and it's why adding an implementation never requires
touching `orchestration/` or existing factories (Open/Closed).

**To add a new OCR engine** (same pattern for preprocessing steps, table extractors, string
matchers, output formatters):
1. Create `ocr/engines/my_engine.py` implementing the `OCREngine` protocol
   (`core/interfaces/ocr_engine.py`).
2. Decorate the class with `@engine_registry.register("my_engine")`.
3. Add it to the import list in `ocr/engines/__init__.py`.
4. Set `ocr.engine: my_engine` and `flow.name: single_engine` in the YAML config.

The same pattern governs the finer-grained families the hybrid engine was split into:
`ocr/detectors/`, `ocr/refiners/`, `ocr/cropping/`, `ocr/recognizers/`,
`table_extraction/classifiers/` and `orchestration/flows/`, each with its own registry and
factory.

To make it selectable from the desktop settings page, add a matching entry to
`InvoiceDigitizationApp/Services/Pipeline/PipelineCatalog.cs` (other repo).

Heavy OCR engines (`surya`, `qwen_vlm`), the Surya detector and the Qwen recognizer are
imported defensively in their packages' `__init__.py` — a machine missing their
dependencies still starts the service; those components just don't register. `requirements.txt` installs only the light path
(`tesseract`); installing `surya`/VLM deps is opt-in (see README.md's OCR engine table).

### Configuration: two shapes, one bridge

- **`config/schema.py`** (`AppConfig`) — the *internal* shape the orchestrator and
  `PreprocessingPipelineBuilder` actually run. Preprocessing steps are **ordered lists**
  (`geometric_steps`, `ocr_photometric_steps`, `table_photometric_steps`); list position is
  execution order. This is what YAML files on disk (`config/default_config.yaml`,
  `config/qwen_config.yaml`, `config/surya_config.yaml`) are written in, and what
  `config/loader.py` parses into.
- **`config/settings_contract.py`** (`PipelineSettings`) — the *wire* shape a request's
  `configuration` object arrives in over HTTP: each preprocessing step is a **fixed, named
  key** (not an array — see `settings-config-contract.md` §1), and the UI controls only
  `enabled` / `algorithm` / `params` per step. Execution order is fixed by the backend
  (`GEOMETRIC_STEP_KEYS` + `PHOTOMETRIC_STEP_KEYS` constants), never by JSON key order.
  `PipelineSettings.to_app_config()` / `.from_app_config()` are the bridge, and
  `ALGORITHMS_BY_STEP` / `UNIMPLEMENTED_ALGORITHMS` validate that a step's chosen algorithm
  is legal for that step and actually implemented, rejecting bad requests at the API
  boundary rather than failing mid-pipeline.

When editing preprocessing config shape, both files (plus `settings-config-contract.md`)
need to move together.

### HTTP layer (`api/`)

- `api/main.py` — FastAPI app; lifespan builds a `PipelinePool` and optionally warms the OCR
  engine at startup so `/health`'s `engine_ready` reflects the truth rather than lying while
  weights load in the background.
- `api/pipeline_pool.py` — **two separate caches**, because rebuilding an OCR engine per
  request would reload model weights (~a minute for the VLM path) for a change unrelated to
  OCR: an *engine* cache keyed by the OCR config section alone, and an LRU *orchestrator*
  cache keyed by the whole configuration. Both are lock-guarded since uvicorn runs sync
  handlers on a thread pool.
- `api/routes.py`, `api/schemas.py` — the actual `/api/v1/health` and `/api/v1/extract`
  endpoints and their Pydantic wire models, per `api-contract.md`.
- `api/service_settings.py` — process-level settings (host/port, which config file, upload
  limits, warmup behavior), env-prefixed `INVOICE_AI_`, distinct from pipeline config: this
  is "how the process runs," not "how a page is read." **Binds to loopback (127.0.0.1) only
  by design** — binding `0.0.0.0` would break the offline/no-LAN-exposure guarantee.

### Domain objects (`core/domain/`)

**Every** domain model lives here and nowhere else. DTOs passed between stages:
`ImagePayload`, `OCRResult`/`OCRFragment`, `TextRegion`/`DetectionResult`/`RegionCrop`,
`TableExtractionResult`, `InvoiceLayout`/`LayoutRegion`, `StructuredDocument`
(`TableCellElement` / `FreeFieldElement`, unioned as `DocumentElement` so the UI can treat
"table cell" and "detected field" uniformly), `MatchedElement`/`MatchResult`, `InvoiceDraft`
and the rest of `invoice.py`, `Catalogs`/`CatalogEntry`/`FieldMatch`, `Word`/`TextLine`,
`PipelineResult`, `PipelineRun`. `BoundingBox` and `Transform` (`geometry.py`) are the
shared coordinate primitives described above, and `roles.py` holds
`CellRole`/`Zone`/`ContentKind`.

The packages beside it hold *behaviour* only — `invoice/` parses, `string_matching/`
matches, `table_extraction/` detects and classifies. `core/interfaces/` holds the Protocol
each pluggable family implements.

### String matching (two algorithms, because names fail two different ways)

| Field | Algorithm | Why |
|---|---|---|
| Merchant, city, governorate | Normalized Levenshtein over the whole string | Wrong one letter at a time. |
| Product name | Order-independent word matching | Words are usually right, order isn't — "جاكيت صوف أزرق" vs "جاكيت أزرق صوف" is the same product; each OCR word takes its best score against catalog words, divided by the larger word count. |

Both return the **top five** matches ranked highest-first, never a single answer. A matched
field on the wire carries **no value at all** — only `original_value`, which is always the
raw OCR text, and `results`, the ranked list. Applying the **0.75** threshold to that list
is the desktop app's job, done in exactly one place (`ExtractionMapper`). This side reports
the scores and decides nothing: a wrong *confident* match silently corrupts an invoice, and
a shape that carries both a value and a candidate list cannot say which of the two the
value was.

Normalization (`string_matching/normalization.py`) runs before any comparison: Arabic-Indic
digits → ASCII, diacritics/tatweel stripped, أ/إ/آ→ا, ى→ي, ة→ه, punctuation stripped,
whitespace collapsed, case folded. Digits are kept — they're part of real product names.

**Normalization and matching live here and nowhere else.** The desktop app used to carry
its own copies in `Helpers/{TextNormalizer,FuzzyMatch,CatalogMatcher}.cs` so it could
re-match locally when the service returned no candidates; those are deleted. The app shows
the ranked `results` this side produced and, below them, the rest of the catalog — so
there is no second implementation for a change here to drift from.

### Preprocessing pipeline detail

Three phases (mirrored in `settings-config-contract.md` §2 and `settings_contract.py`'s
`GEOMETRIC_STEP_KEYS`/`PHOTOMETRIC_STEP_KEYS`):

1. **Get the document** (`geometric_steps`, runs once): perspective correction (crop/warp),
   then deskew. Deskew has two strategies — `deskew_hough` (implemented) and
   `deskew_min_area_rect` (registered but raises `NotImplementedStrategyError` — not built
   yet).
2. **Normalize the surface** (photometric): channel selection → illumination normalization
   (`blur_divide` implemented; `blackhat` not implemented) → contrast enhancement (`clahe` or
   `plain_equalization`, both implemented).
3. **Binarize and clean** (photometric): denoising (`median_blur` / `gaussian_blur` /
   `nlm_denoise` / `bilateral_filter` — NLM is noticeably slower, relevant given the low-end
   hardware constraint) → thresholding (`fixed_threshold` / `otsu_threshold` /
   `adaptive_threshold`) → morphological cleanup (single algorithm, `operation` param
   selects open/close/erode/dilate).

Steps self-register in `preprocessing/steps/registry.py`; `preprocessing/pipeline_builder.py`
builds an ordered `PreprocessingPipeline` from a `list[StepConfig]`. Geometric steps live
under `preprocessing/steps/geometric/`, photometric under `preprocessing/steps/photometric/`.

The "enhanced image" returned to the UI as a debug artifact
(`enhanced_image_png`/`ocr_input_image_png` in the API response, gated behind
`options.return_debug_images`) is captured as the pipeline runs — specifically the frame
just before the first binarizing step (`_BINARIZING_STEPS` in `pipeline_orchestrator.py`),
since that's the last point a human can still read the page; the binarized frame after it is
what the OCR engine actually sees.

### Cell mapping

`mapping/cell_mapper.py` assigns fragments to the layout's regions — **all** of them, the
header captions and the totals strip included, not only table cells. That is what lets the
parser read a labelled invoice number instead of inferring one from keywords. A fragment
carrying a `source_id` (the layout-driven flow's crops do) is assigned by identity, since
the crop came from a known region and re-deriving it from geometry would be discarding a
fact to guess at it; everything else falls back to centroid containment, smallest region
first. Fragments landing in no region become `FreeFieldElement`s.

### Traps

**`PreprocessingPipeline.run()` must not mutate its input.** It used to, so the two
photometric "branches" were really a chain: the table branch read the OCR branch's output,
and the display image handed to the UI — the one every bbox is measured against and the
user actually clicks — was the final binarized mask rather than the corrected page.
`tests/test_coordinate_space.py` pins this.

**Table extractors do no preprocessing.** They require a binary image with ink at 255 and
say so precisely when they do not get it (`extractors/binary_input.py`). They used to
re-deskew and re-threshold whatever arrived, which rotated the page a second time and put
every cell bbox in a coordinate space no OCR bbox shared — invisibly, because a
wrongly-placed box is still a perfectly valid box.

**A stray box above the header used to steal the invoice number.** `bill_layout` scored the
invoice-number candidate *within the topmost header row*. A row holding one cell normalizes
to zero on both axes — `max(range, 1.0)` divided into a spread of zero — so whatever sat in
it won by default and its position was never consulted. A 71×222 sliver of the page edge,
closed as its own contour at the top right of `images/test1.jpg`, therefore took
`invoice_number`, was cropped, read with the free-text prompt, and came back an invented
Arabic sentence, while the box reading `رقم الفاتورة : 00010` stayed `unknown`. Scoring
across the whole header block is what makes the corner mean something.

**Preprocessing tuned for a binarizing engine can destroy handwriting for a VLM.** The
recognizer crops from the OCR-photometric image, not from the page the UI shows, so a step
that helps printed text can quietly wreck a field while leaving the printed caption beside
it crisp — the field looks read, just wrong. `illumination_normalization_blur_divide` does
exactly that to light blue ballpoint; see the measurement in `config/default_config.yaml`.
When a handwritten field reads badly, compare the crop against the display image before
touching prompts or models.

**Deskew exists once.** There were three copies (the step, `grid_utils`,
`temp/table_det.py`) and they disagreed. The survivor folds angles modulo 90 so vertical
rules vote too, and expands the canvas so no corner is clipped.
`tools/compare_deskew.py` renders old against new over a folder of real invoices;
`tools/render_layout.py` draws what the extractor found and what the classifier made of it.

### Output formatters and persistence

`output/formatters/invoice_json_formatter.py` produces the JSON the desktop app consumes —
`../docs/api-contract.md`'s response shape, one named key per invoice field, each carrying
a `bounding_box` for the clickable overlay. **The HTTP layer forces this formatter onto
every request** (`api/routes.py`'s `WIRE_FORMATTER`): it *is* the documented response
shape, so a configuration naming another one would return a body the desktop app cannot
read while every other part of the request looked valid.

The other two are for people, not for the app. `ui_overlay_formatter.py` is the lossless
view — one entry per detected box, including text no invoice field claimed — and
`csv_formatter.py` is the flat table. Either can be selected from a YAML file for a CLI
run; neither is a valid `/extract` response. `persistence/file_result_store.py` writes
`results/<invoice_id>/{result.json, display.png, preprocessing/<step-debug-images>}` — the
audit trail, which serializes the `PipelineResult` itself and is therefore unaffected by
which formatter is configured; `write_debug_images`/`persist` are turned off for HTTP requests by default
(`api/service_settings.py`) since writing one PNG per step per request would dominate
request latency, and are off for the whole test session via `tests/conftest.py`.

## Working across the repo boundary

This repo does not contain the WinUI/C# app — treat any C# path mentioned here
(`InvoiceDigitizationApp/...`) as documentation of what has to change *elsewhere* when you
touch the corresponding contract on this side. If a task changes the `/extract` response
shape, the settings `configuration` shape, or the normalization/matching rules, say so
explicitly rather than treating the Python-side change as complete.

## Universal Instructions

1. Follow SOLID prinicples
2. use Design Pattern if appropriate
3. the current project have some issues
