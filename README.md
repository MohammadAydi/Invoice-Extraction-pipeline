# Invoice Digitization — Architecture Backbone

This is the project backbone only: interfaces, DTOs, configuration, registries,
factories, and the orchestrator that wires them together. Every concrete
algorithm (perspective correction, OCR, table extraction, string matching,
output formatting) is a stub that raises `NotImplementedError` with a
docstring pointing at the reference material it should be built from.

## Key architectural decision: geometric correction runs once

Of all the preprocessing steps, only **perspective correction** and
**deskew** change pixel coordinates. Everything else (channel selection,
illumination normalization, CLAHE, bilateral filtering, thresholding,
morphology) is purely photometric.

Because of that, geometric correction is factored out as a single upstream
stage (`preprocessing.geometric_steps` in config) that runs **once**, before
the pipeline forks into the OCR-photometric and table-photometric branches.
Both branches — and therefore every OCR fragment and every table cell they
produce — end up in exactly one coordinate space: that of the geometrically
corrected image. That same image is what the UI displays on both sides
(plain on the left, with overlays on the right). This is what guarantees no
bounding box ever needs to be remapped between coordinate spaces — see
`core/domain/geometry.py` for the full reasoning.

## Directory layout

```
config/            AppConfig schema (pydantic) + YAML loader + default_config.yaml
core/
  domain/          Immutable DTOs passed between stages (ImagePayload, OCRResult,
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
ocr/                Engine adapters (tesseract, easyocr) + factory
table_extraction/   Extractor adapters (contour_based) + factory
mapping/            CellMapper: OCR fragments + table cells -> StructuredDocument
string_matching/    Matcher adapters (levenshtein, embedding) + factory
output/              Formatter adapters (ui_overlay_json, csv) + factory
persistence/        ResultStore adapter (file_result_store) for audit/reproducibility
orchestration/       PipelineOrchestrator: the one module that knows the full pipeline shape
main.py             Usage example / entry point
```

## How the pieces fit together

`PipelineOrchestrator.run()`:

1. Runs `geometric_pipeline` once → the canonical display image.
2. Runs `ocr_photometric_pipeline` and `table_photometric_pipeline`
   sequentially on that same image.
3. Calls `ocr_engine.recognize()` and `table_extractor.extract()`
   sequentially.
4. `CellMapper.map()` assigns OCR fragments to table cells (or to
   free-standing fields).
5. `string_matcher.match()` runs against the single shared keyword
   dictionary for every element, table cell or free field alike.
6. Assembles a `PipelineResult` — including a full `config_snapshot` for
   audit/reproducibility — persists it via `ResultStore`, then hands it to
   `output_formatter.format()` for the UI (or whatever format is configured).

Everything the orchestrator depends on (which OCR engine, which table
extractor, which matcher, which formatter, which preprocessing steps run and
in what order) comes from one `AppConfig` object, loaded from
`config/default_config.yaml`.

## Adding a new implementation (why this scales across a team)

To add a new OCR engine, for example:

1. Create `ocr/engines/my_engine.py` implementing `OCREngine.recognize()`.
2. Decorate the class with `@engine_registry.register("my_engine")`.
3. Add it to the import list in `ocr/engines/__init__.py`.
4. Set `ocr.engine: my_engine` in the YAML config.

Nothing in `orchestration/`, `ocr/factory.py`, or any other module changes.
The same pattern applies to preprocessing steps, table extractors, string
matchers, and output formatters — each is a self-contained file, a registry
entry, and a config value.

## Running it

```bash
pip install -r requirements.txt
python main.py path/to/invoice.jpg
```

This will run correctly through config loading, registry-driven pipeline
construction, and orchestrator wiring, then raise `NotImplementedError` at
the first real algorithm step (`perspective_correction`) — which is expected
until that stub, and the others, are filled in.

## Verified wiring (already tested during scaffolding)

- Config loads and validates against the pydantic schema.
- All 8 registered preprocessing steps build correctly from
  `default_config.yaml` across all three pipelines (2 geometric, 6 OCR
  photometric, 2 table photometric).
- OCR engine, table extractor, string matcher, and output formatter all
  resolve to the correct concrete class via their factories.
- `PipelineOrchestrator.run()` executes stages in the correct order and
  fails exactly at the first unimplemented stub, not before.

## Notes for whoever implements each stage

- Every stub's docstring names the lesson script(s) in your existing
  preprocessing work that its logic should be adapted from.
- `Transform.then()` / `Transform.invert()` / `Transform.apply_to_point()`
  are fully implemented (pure linear algebra) — geometric steps just need
  to compose their own transform into `ctx.payload.transform`.
- `keywords/ar_invoice_terms.json` is a placeholder; replace with the real
  keyword dictionary.
