# Invoice Extraction Pipeline

The Python half of the Invoice Digitization system. It reads an invoice image and returns
a structured invoice — merchant, number, date, city, total, and the table's line items —
each value carrying a confidence and a ranked list of catalog matches for the user to
confirm.

Two ways in, one pipeline behind both:

| Entry point | Used by |
|---|---|
| `api/main.py` — HTTP, per [docs/api-contract.md](../docs/api-contract.md) | The WinUI desktop app |
| `main.py` / `run_batch.py` — CLI | Reproducing a bad extraction offline, tuning config |

Both build the same `PipelineOrchestrator` from the same YAML config, so anything the
desktop app sees can be reproduced from a terminal with one command.

## Key architectural decision: geometric correction runs once

Of all the preprocessing steps, only **perspective correction** and **deskew** change
pixel coordinates. Everything else (channel selection, illumination normalization, CLAHE,
bilateral filtering, thresholding, morphology) is purely photometric.

Because of that, geometric correction is factored out as a single upstream stage
(`preprocessing.geometric_steps` in config) that runs **once**, before the pipeline forks
into the OCR-photometric and table-photometric branches. Both branches — and therefore
every OCR fragment and every table cell they produce — end up in exactly one coordinate
space: that of the geometrically corrected image. That same image is what the UI displays
on both sides (plain on the left, with overlays on the right). This is what guarantees no
bounding box ever needs to be remapped between coordinate spaces — see
`core/domain/geometry.py` for the full reasoning.

## Directory layout

```
api/                HTTP layer: FastAPI app, routes, wire schemas, pipeline pool
config/             AppConfig schema (pydantic) + YAML loader + settings_contract.py
                     (the object-keyed wire shape the desktop settings page sends)
core/
  domain/           Immutable DTOs passed between stages (ImagePayload, OCRResult,
                     TableExtractionResult, StructuredDocument, MatchedElement,
                     PipelineResult, ...)
  interfaces/       Protocols every pluggable family implements
  registry.py       Generic name -> class registry (Strategy + Factory backbone)
  exceptions.py
preprocessing/
  steps/geometric/     perspective_correction, deskew  (run once, produce the display image)
  steps/photometric/   channel_selection, illumination_normalization, clahe,
                        bilateral_filter, thresholding, morphological_cleanup
  pipeline.py           PreprocessingPipeline: runs an ordered step list
  pipeline_builder.py   builds a Pipeline from config + the step registry
ocr/                Engine adapters (stub, tesseract, surya, qwen_vlm, surya_qwen) + factory
table_extraction/   Extractor adapters (grid_line, contour_based) + factory
mapping/            CellMapper: OCR fragments + table cells -> StructuredDocument
string_matching/    Normalization, the two similarity algorithms, catalog matching,
                     and the keyword-dictionary matcher adapters + factory
invoice/            The semantic layer: elements -> header fields + line items + warnings
output/             Formatter adapters (invoice_json, ui_overlay_json, csv) + factory
persistence/        ResultStore adapter (file_result_store) for audit/reproducibility
orchestration/      PipelineOrchestrator: the one module that knows the full pipeline shape
tests/              pytest suite, including end-to-end tests of the HTTP contract
```

## How the pieces fit together

`PipelineOrchestrator.run()`:

1. Runs `geometric_pipeline` once → the canonical display image.
2. Runs `ocr_photometric_pipeline` and `table_photometric_pipeline` sequentially on that
   same image.
3. Calls `ocr_engine.recognize()` and `table_extractor.extract()` sequentially.
4. `CellMapper.map()` assigns OCR fragments to table cells (or to free-standing fields).
5. `string_matcher.match()` runs against the shared keyword dictionary for every element,
   table cell or free field alike.
6. `InvoiceParser.parse()` reads those elements as an invoice, matching the merchant,
   city and product names against the catalogs the request carried.
7. Assembles a `PipelineResult` — including a full `config_snapshot` for
   audit/reproducibility — persists it via `ResultStore`, then hands it to
   `output_formatter.format()`.

Everything the orchestrator depends on comes from one `AppConfig` object. The catalogs are
the one thing that does not: they belong to the *request*, not the configuration, so they
are passed to `run()`.

## String matching

Two algorithms, because names fail in two different ways:

| Field | Algorithm | Why |
|---|---|---|
| Merchant, city, governorate | Normalized Levenshtein over the whole string | Read as one run of characters; they go wrong one letter at a time. |
| Product name | Order-independent word matching | The words are usually right but their order is not. "جاكيت صوف أزرق" and "جاكيت أزرق صوف" are the same product; each OCR word takes its best score against the catalog words, and the total is divided by the larger word count so extra words on either side dilute it. |

Both return the **top five** matches ranked highest-first, not a single answer: the
desktop app pre-selects the best one and offers the rest, so confirming a correct read
costs nothing and correcting a wrong one costs a click. Anything scoring below 0.75 is
returned with `requires_manual_review` set and the raw OCR text left as the value — a
wrong confident match silently corrupts an invoice.

Everything is normalized first (`string_matching/normalization.py`): Arabic-Indic digits
folded to ASCII, diacritics and tatweel removed, أ/إ/آ→ا, ى→ي, ة→ه, punctuation stripped,
whitespace collapsed, case folded. Digits survive — they are part of real product names.

**These rules live here alone.** The C# copies in `InvoiceDigitizationApp/Helpers/{TextNormalizer,FuzzyMatch,CatalogMatcher}.cs` are deleted: two implementations of "the same string" made the answer depend on which side computed it.
The C# copies exist because the app re-matches locally when the service returned no
candidates. If one side changes, the other must change with it.

## Adding a new implementation (why this scales across a team)

To add a new OCR engine, for example:

1. Create `ocr/engines/my_engine.py` implementing `OCREngine.recognize()`.
2. Decorate the class with `@engine_registry.register("my_engine")`.
3. Add it to the import list in `ocr/engines/__init__.py`.
4. Set `ocr.engine: my_engine` in the YAML config.

Nothing in `orchestration/`, `ocr/factory.py`, or any other module changes. The same
pattern applies to preprocessing steps, table extractors, string matchers, and output
formatters. To make it selectable from the desktop settings page, add a matching entry to
`InvoiceDigitizationApp/Services/Pipeline/PipelineCatalog.cs`.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

# The HTTP service the desktop app talks to.
python -m api.main              # http://127.0.0.1:8765

# Or one image from the command line.
python main.py path/to/invoice.jpg
python run_batch.py bills/ --config config/qwen_config.yaml
```

### OCR engines

`requirements.txt` installs only the light path. Which engine runs comes from
`ocr.engine` in the config, or from the desktop settings page.

| Engine | Needs | Notes |
|---|---|---|
| `stub` | nothing | Fabricates a deterministic invoice from a hash of the image. For verifying the desktop↔service link before a real engine is installed — never for real invoices. |
| `tesseract` | the Tesseract **binary** plus its `ara`/`eng` traineddata, which pip cannot install | The default. No weights, no GPU. |
| `surya` | `pip install surya-ocr==0.17.1 "transformers<5.0"` | Downloads models on first run. Its recognizer is weak on handwritten Arabic. |
| `surya_qwen` | the above, **plus** a second virtualenv with `transformers>=5.0` and a local Qwen model | The most accurate on handwriting: Surya's detector locates the fields, Qwen reads each crop. Paths are set under `ocr.engine_params` in `config/qwen_config.yaml`. |

`ocr/engines/__init__.py` imports the heavy engines defensively, so a machine with none of
their dependencies still starts the service — the missing engines simply do not register.

### Configuration

`config/default_config.yaml` is what the service runs when a request sends no
`configuration` of its own. `config/qwen_config.yaml` is the tuned VLM setup, with the
reasoning for every enabled and disabled step written inline — read it before changing
preprocessing.

Environment variables prefixed `INVOICE_AI_` override service-level settings (host, port,
which config file to load, whether to persist results). See `api/service_settings.py`.

## Tests

```bash
pytest
```

Covers the normalization rules, both matching algorithms, the invoice parser, the
configuration contract's translation and validation, and the HTTP contract end to end —
the API tests run the real pipeline over a synthetic image using the `stub` engine, so
they need no OCR installation.
