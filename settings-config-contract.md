# Settings / Pipeline Configuration Contract — ai-service ⇄ desktop-app

This document describes every configurable knob of the extraction pipeline, for the
person building the C# Settings page. It is the sibling of [api-contract.md](api-contract.md):
that file documents the `/extract` request/response shape; this file documents the
`configuration` object that gets embedded inside that request so the user can control
*how* extraction runs.

The UI should let the user pick, per step, **which algorithm** runs and **what
parameters** it uses, then serialize the whole thing to the JSON shape below and send it
as part of the extract request.

> **Note on backend status:** this document describes the **target** wire shape —
> object-keyed by step name, not the array-of-steps shape currently implemented in
> `config/schema.py` / `preprocessing/pipeline_builder.py`. Today's Python code takes
> `geometric_steps: list[StepConfig]` / `ocr_photometric_steps: list[StepConfig]`, where
> array position determines execution order. This MD documents where the contract is
> headed so the C# settings UI can be built against the final shape; the Python schema,
> loader, and pipeline builder still need a follow-up change to accept this object shape
> (translating each named key to the right internal step + execution order) before the
> service actually accepts it end-to-end.

---

## 1. Top-level shape

Each preprocessing step is a **fixed, named key** — not an array entry. The step's
identity and its position in the pipeline are both fixed by the backend; the only things
the UI controls per step are: is it **enabled**, which **algorithm** implements it, and
that algorithm's **params**.

```json
{
  "preprocessing": {
    "perspective_correction": {
      "enabled": true,
      "algorithm": "perspective_correction",
      "params": { "canny_low": 30, "canny_high": 150 }
    },
    "deskew": {
      "enabled": true,
      "algorithm": "deskew_hough",
      "params": { "hough_threshold": 100, "min_line_length": 150, "max_line_gap": 10, "max_angle_deg": 20 }
    },
    "channel_selection": {
      "enabled": true,
      "algorithm": "channel_selection",
      "params": { "channel": "gray" }
    },
    "illumination_normalization": {
      "enabled": true,
      "algorithm": "illumination_normalization_blur_divide",
      "params": { "blur_kernel": 95 }
    },
    "contrast_enhancement": {
      "enabled": true,
      "algorithm": "clahe",
      "params": { "clip_limit": 2.5, "tile_grid_size": [8, 8] }
    },
    "denoising": {
      "enabled": true,
      "algorithm": "bilateral_filter",
      "params": { "d": 20, "sigma_color": 25, "sigma_space": 50 }
    },
    "thresholding": {
      "enabled": true,
      "algorithm": "adaptive_threshold",
      "params": { "block_size": 51, "c": 35 }
    },
    "morphological_cleanup": {
      "enabled": true,
      "algorithm": "morphological_cleanup",
      "params": { "operation": "open", "kernel_size": 2 }
    }
  },
  "ocr": { "engine": "tesseract", "engine_params": { } },
  "table_extraction": { "extractor": "grid_line", "extractor_params": { } },
  "string_matching": {
    "algorithm": "levenshtein",
    "algorithm_params": { },
    "dictionary_path": "keywords/ar_invoice_terms.json"
  },
  "output": { "formatter": "ui_overlay_json", "formatter_params": { } },
  "persistence": { "store": "file_result_store", "store_params": { "output_dir": "results/" } }
}
```

Notes on the shape, since they affect how the UI must serialize things:

- **Every one of the 8 keys under `preprocessing` is fixed and required** — always send
  all 8, even the ones with only one possible algorithm (`perspective_correction`,
  `channel_selection`, `morphological_cleanup`). Execution order is fixed by the backend
  (Phase 1 → Phase 2 → Phase 3, in the order listed in Section 2) and is **not**
  controlled by key order in the JSON object — don't build a reorderable-list UI for v1.
- **`enabled: false` keeps the key present but skips that step at runtime.** This is how
  the settings UI should represent a toggle switch/checkbox next to each step — never
  omit a step's key to "disable" it; omitting a required key should be treated as invalid.
- **`algorithm` selects which strategy implements the step.** For steps with only one
  known strategy today, still send it explicitly (e.g. `"algorithm": "channel_selection"`)
  rather than assuming a default — keeps the contract self-describing and future-proof
  for when a second strategy is added.
- **`params` is a free-form object** whose valid keys depend on `algorithm`, not on the
  step key. Section 3 below is the exhaustive list of valid `algorithm` values per step
  and the params each accepts.
- **`table_photometric_steps` (table-branch preprocessing) is intentionally omitted**
  from this contract — not implemented in the settings UI yet.
- **`ocr`, `table_extraction`, `string_matching`, `output`, `persistence`** are single-choice
  configs (pick one engine/algorithm/formatter/store). Their `params`
  documented in Section 4 are today's best-known values from `default_config.yaml` —
  treat them as placeholders/starting defaults for the UI, not a finalized contract,
  since there's no dedicated design for these yet.

---

## 2. UI grouping: 3 phases

Group the 8 step keys into the following phases in the UI. This matches how the
pipeline is conceptually organized and documented in code comments. All 8 keys are
siblings under `preprocessing` (see Section 1) — phase grouping is purely a UI
convenience for readability; it has no effect on the JSON shape.

| Phase | Steps in this phase, in order | Step keys |
|---|---|---|
| **Phase 1 — Get the document** | 1. Perspective Correction (crop/warp)<br>2. Deskew | `perspective_correction`, `deskew` |
| **Phase 2 — Normalize the surface** | 3. Channel Selection<br>4. Illumination Normalization<br>5. Contrast Enhancement | `channel_selection`, `illumination_normalization`, `contrast_enhancement` |
| **Phase 3 — Binarization and cleaning** | 6. Denoising<br>7. Thresholding<br>8. Morphological Cleanup | `denoising`, `thresholding`, `morphological_cleanup` |

Phase 1 always runs first, producing the single corrected image that Phases 2 and 3
(together forming the OCR-photometric branch) operate on in sequence.

Within each step, several are "pick one algorithm, then fill in that algorithm's
parameters" — i.e. a radio button / dropdown selecting `name`, followed by a parameter
form that changes based on the selection. A few steps have only one algorithm (no
choice to present, just params). This is marked per-step below.

---

## 3. Preprocessing steps — full reference

Legend for parameter tables: **Type**, **Default** (from `default_config.yaml`),
**Constraints** (validated by the Python step — violating these throws a `ValueError`
at pipeline build/run time), **UI control** (suggested widget).

### Phase 1 — Get the document

#### 1.1 Perspective Correction — step key `perspective_correction`
Single algorithm, no alternative — `algorithm` is always `"perspective_correction"`. No
enable/disable choice needed conceptually, but still send `enabled: true`/`false` like
every other step.

```json
"perspective_correction": { "enabled": true, "algorithm": "perspective_correction", "params": { "canny_low": 30, "canny_high": 150 } }
```

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `canny_low` | int | 30 | > 0, should be < `canny_high` | Slider or number input |
| `canny_high` | int | 150 | > 0, should be > `canny_low` | Slider or number input |

#### 1.2 Deskew — step key `deskew`
Algorithm choice (`algorithm`):

| `algorithm` value | Label suggestion | Status |
|---|---|---|
| `deskew_hough` | "Hough Line Transform" | **Implemented** — usable |
| `deskew_min_area_rect` | "Minimum Area Rectangle (text/ink pixels)" | **Not implemented.** Selecting it raises `NotImplementedStrategyError` at run time. UI should either hide this option for now, or show it disabled/greyed with a "coming soon" note. |

```json
"deskew": { "enabled": true, "algorithm": "deskew_hough", "params": { "hough_threshold": 100, "min_line_length": 150, "max_line_gap": 10, "max_angle_deg": 20 } }
```

Params for `deskew_hough`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `hough_threshold` | int | 100 | > 0 | Number input / slider |
| `min_line_length` | int | 150 | > 0 | Number input / slider |
| `max_line_gap` | int | 10 | ≥ 0 | Number input / slider |
| `max_angle_deg` | float | 20.0 | > 0 | Slider (degrees) |

`deskew_min_area_rect` currently accepts arbitrary params (they're captured but unused
since it always raises) — no param UI needed while it's disabled.

---

### Phase 2 — Normalize the surface

#### 2.1 Channel Selection — step key `channel_selection`
Single algorithm, no alternative — `algorithm` is always `"channel_selection"`. This
step is a required prerequisite (every step after it expects a single-channel/grayscale
image) — consider defaulting it to `enabled: true` and discouraging the user from
disabling it, though the UI does not need to hard-block disabling it.

```json
"channel_selection": { "enabled": true, "algorithm": "channel_selection", "params": { "channel": "gray" } }
```

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `channel` | enum string | `"gray"` | one of `gray`, `b`, `g`, `r` | Dropdown |

#### 2.2 Illumination Normalization — step key `illumination_normalization`
Algorithm choice (`algorithm`):

| `algorithm` value | Label suggestion | Status |
|---|---|---|
| `illumination_normalization_blur_divide` | "Blur & Divide" | **Implemented** — usable |
| `illumination_normalization_blackhat` | "Morphological Black-hat / Top-hat" | **Not implemented.** Raises `NotImplementedStrategyError`. Hide or disable in UI for now. |

```json
"illumination_normalization": { "enabled": true, "algorithm": "illumination_normalization_blur_divide", "params": { "blur_kernel": 95 } }
```

Params for `illumination_normalization_blur_divide`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `blur_kernel` | int | 95 | **positive odd integer** | Number input restricted to odd values (e.g. a stepper that increments by 2), or slider with odd-only steps |

`illumination_normalization_blackhat` — no param UI needed while disabled.

#### 2.3 Contrast Enhancement — step key `contrast_enhancement`
Algorithm choice (`algorithm`):

| `algorithm` value | Label suggestion | Status |
|---|---|---|
| `clahe` | "CLAHE (adaptive)" | **Implemented** — usable |
| `plain_equalization` | "Plain Histogram Equalization" | **Implemented** — usable, no params |

```json
"contrast_enhancement": { "enabled": true, "algorithm": "clahe", "params": { "clip_limit": 2.5, "tile_grid_size": [8, 8] } }
```

Params for `clahe`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `clip_limit` | float | 2.5 | > 0 | Slider / number input |
| `tile_grid_size` | [int, int] | [8, 8] | both > 0 | Two number inputs (width, height) |

Params for `plain_equalization`: none — selecting it should show an empty/no-params panel.

---

### Phase 3 — Binarization and cleaning

#### 3.1 Denoising — step key `denoising`
Algorithm choice (`algorithm`):

| `algorithm` value | Label suggestion | Status |
|---|---|---|
| `median_blur` | "Median Blur" | Implemented |
| `gaussian_blur` | "Gaussian Blur" | Implemented |
| `nlm_denoise` | "Non-Local Means (NLM)" | Implemented |
| `bilateral_filter` | "Bilateral Filter" | Implemented |

```json
"denoising": { "enabled": true, "algorithm": "bilateral_filter", "params": { "d": 20, "sigma_color": 25, "sigma_space": 50 } }
```

Params for `median_blur`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `k` | int | 5 | **positive odd integer** | Odd-stepping number input |

Params for `gaussian_blur`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `ksize` | [int, int] | [5, 5] | typically odd positive ints (OpenCV convention; not enforced in code) | Two number inputs |
| `sigma` | float | 0 | ≥ 0 (0 = auto-computed from ksize) | Number input |

Params for `nlm_denoise`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `h` | float | 10 | > 0 (filter strength) | Slider |
| `template_window_size` | int | 7 | > 0, odd (OpenCV convention; not enforced in code) | Number input |
| `search_window_size` | int | 21 | > 0, odd (OpenCV convention; not enforced in code) | Number input |

Params for `bilateral_filter`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `d` | int | 20 | > 0 (pixel neighborhood diameter) | Slider / number input |
| `sigma_color` | float | 25 | > 0 | Slider |
| `sigma_space` | float | 50 | > 0 | Slider |

This is a **low-end-laptop-sensitive step** — `nlm_denoise` in particular is
significantly slower than the others. Consider a UI hint like "slower, higher quality"
next to NLM, since the app must run acceptably on low-spec hardware with no GPU.

#### 3.2 Thresholding — step key `thresholding`
Algorithm choice (`algorithm`):

| `algorithm` value | Label suggestion | Status |
|---|---|---|
| `fixed_threshold` | "Fixed Threshold" | Implemented |
| `otsu_threshold` | "Otsu (automatic)" | Implemented, no user params |
| `adaptive_threshold` | "Adaptive Threshold" | Implemented |

```json
"thresholding": { "enabled": true, "algorithm": "adaptive_threshold", "params": { "block_size": 51, "c": 35 } }
```

**Every** thresholding algorithm additionally takes `invert`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `invert` | bool | `false` | — | Checkbox, "Ink is white (inverted)" |

`invert` selects `THRESH_BINARY_INV` — white ink on a black page instead of the usual
black ink on white. It is not a preference: morphological line reconstruction, which is
how both table extractors find the printed grid, erodes and dilates the **foreground**, so
the table branch needs ink at 255 while an OCR engine wants ink at 0. The table branch's
own preprocessing is not exposed in this contract (see §1), so in practice `invert` only
appears on the settings page for completeness — the server-side table branch sets it.

Params for `fixed_threshold`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `t` | int | 127 | 0–255 | Slider (0–255) |

Params for `otsu_threshold`: none — Otsu picks its own threshold automatically. Empty
params panel. (The chosen threshold is reported back in the pipeline's internal metadata
but is not currently part of the `/extract` API response.)

Params for `adaptive_threshold`:

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `block_size` | int | 51 | **odd integer ≥ 3** | Odd-stepping number input, min 3 |
| `c` | float | 35 | any (constant subtracted from the mean) | Slider / number input |

#### 3.3 Morphological Cleanup — step key `morphological_cleanup`
Single algorithm, no alternative — `algorithm` is always `"morphological_cleanup"`. The
choice here is not *which algorithm* but *which morphological operation*, via a param:

```json
"morphological_cleanup": { "enabled": true, "algorithm": "morphological_cleanup", "params": { "operation": "open", "kernel_size": 2 } }
```

| Param | Type | Default | Constraints | UI control |
|---|---|---|---|---|
| `operation` | enum string | `"open"` | one of `open`, `close`, `erode`, `dilate` | Dropdown (present this as the primary control for this step) |
| `kernel_size` | int | 2 | > 0 | Number input |

---

## 4. Other pipeline stages (single-choice, not step lists)

These are not part of `preprocessing` and are not phase-grouped — they're independent
sections the user picks once each. **There is no finalized design for these yet.** The
tables below simply transcribe today's `default_config.yaml` so the UI has *something*
concrete to scaffold against; treat every field here as provisional and expect it to
change as OCR/table-extraction/string-matching work matures. Build the settings page
so these sections are easy to extend (e.g. driven by the same "pick a `name`, then show
that choice's params" pattern used above) rather than hardcoding today's one option as
the only possible one.

### 4.0 Flow — which reading strategy runs

```json
{ "flow": { "name": "single_engine", "params": {} } }
```

| `name` | What it does | Which `ocr` fields it uses |
|---|---|---|
| `single_engine` | One OCR engine reads the whole page. **Default.** | `engine`, `engine_params` |
| `detector_driven` | A text detector finds boxes anywhere on the page, a refiner squares them up against a declared column layout, each is cropped and read. Reads text outside any printed box; needs the column boundaries declared, and misses short isolated digits the detector never boxes. | `detector`, `refiner`, `cropper`, `recognizer` |
| `layout_driven` | The printed grid is detected and classified first, then each labelled cell is cropped and read with the prompt its role implies. The grid supplies the real column boundaries, so nothing has to be declared. **No text detector runs at all.** Text outside a ruled box is not read. | `cropper`, `recognizer` |

Omitting `flow` entirely means `single_engine`, so a client written against the previous
version of this contract keeps working and keeps getting the behaviour it was written for.

Only the fields the selected flow names are built. That is the mechanism, not just the
intent, behind "layout_driven does not use the detector": no detector object is
constructed, so it cannot be reached.

`layout_driven` with no detected grid returns an empty extraction plus a
`NO_LAYOUT_DETECTED` warning. It deliberately does **not** fall back to the detector — a
silent switch to a different reading strategy is what makes a bad extraction impossible to
diagnose.

### 4.1 OCR

```json
{
  "ocr": {
    "engine": "tesseract",
    "engine_params": { "lang": "ara" },

    "detector":   { "name": "surya_detector", "params": { "min_side": 12 } },
    "refiner":    { "name": "column_fraction", "params": { "column_kinds": [] } },
    "cropper":    { "name": "padded_crop", "params": { "pad": 10, "upscale": 2 } },
    "recognizer": { "name": "qwen", "params": { "backend": "subprocess" } }
  }
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `engine` | string or null | `"tesseract"` | `single_engine` only. Registered: `stub`, `tesseract`, `surya`, `easyocr`, `qwen_vlm`. |
| `engine_params` | object | `{}` | Engine-specific. Tesseract takes `lang`. |
| `detector` | object or null | — | `detector_driven` only. Registered: `surya_detector`. |
| `refiner` | object or null | `{"name": "noop"}` | `detector_driven` only. Registered: `column_fraction`, `noop`. |
| `cropper` | object or null | `{"name": "padded_crop"}` | Both region-based flows. Params: `pad`, `upscale`, `min_side`. |
| `recognizer` | object or null | — | Both region-based flows. Registered: `qwen`, `echo`. |

`surya_qwen` no longer exists as an engine. It did detection, box refinement, cropping,
recognition and debug-artifact writing in one class, so none of the five could be swapped
independently. It is now `flow.name: "detector_driven"` composed from the four components
above, and its behaviour is unchanged.

### 4.2 Table Extraction

```json
{
  "table_extraction": {
    "extractor": "grid_line",
    "extractor_params": {
      "dot_bridge_scale": 150,
      "main_kernel_scale": 30,
      "min_line_length_ratio": 0.05,
      "intersection_tolerance": 15,
      "merge_proximity": 15,
      "min_extend_length_ratio": 0.5,
      "coverage_ratio": 0.5,
      "save_debug_images": true
    }
  }
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `extractor` | string | `"grid_line"` | `grid_line` reconstructs the grid from its lines; `contour_based` takes the closed regions directly and copes better with broken rules and handwriting crossing them. |
| `classifier` | string | `"passthrough"` | **New.** What the detected boxes *mean*. `passthrough` labels nothing (the behaviour before classification existed); `bill_layout` labels the standard Arabic sales-invoice form. |
| `classifier_params` | object | `{}` | For `bill_layout`: `table_row_count`, `table_col_count`, `column_roles`, `header_roles`, `footer_roles`. |
| `dot_bridge_scale` | int | 150 | |
| `main_kernel_scale` | int | 30 | |
| `min_line_length_ratio` | float | 0.05 | |
| `intersection_tolerance` | int | 15 | |
| `merge_proximity` | int | 15 | |
| `min_extend_length_ratio` | float | 0.5 | |
| `coverage_ratio` | float | 0.5 | |

`save_debug_images` is gone. Rendering is `tools/render_layout.py` now — a library that
writes PNGs as a side effect of being called is not something to expose as a setting.

**Both extractors now require a binary image with ink at 255**, produced by the table
branch's `adaptive_threshold` with `invert: true`. They used to accept anything and
silently re-threshold — and re-deskew — whatever arrived, which put every cell bbox in a
different coordinate space from every OCR bbox. An image of the wrong shape or polarity is
now refused with a message naming the config that fixes it.

The classifier is what makes header fields and the totals row available to the desktop app
as labelled boxes: `role` and `zone` on each overlay element, and a `bbox` on each header
field pointing at the box the value was actually read from. See `api-contract.md`.

### 4.3 String Matching

```json
{
  "string_matching": {
    "algorithm": "levenshtein",
    "algorithm_params": { "max_distance": 2 },
    "dictionary_path": "keywords/ar_invoice_terms.json"
  }
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `algorithm` | string | `"levenshtein"` | Only known value today. |
| `algorithm_params.max_distance` | int | 2 | Max edit distance for a fuzzy match. |
| `dictionary_path` | string | `"keywords/ar_invoice_terms.json"` | Path is service-local; likely not user-editable from the UI (no file picker across the HTTP boundary) — probably should be fixed/hidden rather than exposed. |

### 4.4 Output

```json
{ "output": { "formatter": "ui_overlay_json", "formatter_params": {} } }
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `formatter` | string | `"ui_overlay_json"` | Only known value today; this is what shapes the `/extract` response documented in api-contract.md. Likely not meant to be user-facing at all. |

### 4.5 Persistence

```json
{ "persistence": { "store": "file_result_store", "store_params": { "output_dir": "results/" } } }
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `store` | string | `"file_result_store"` | Only known value today. |
| `store_params.output_dir` | string | `"results/"` | Server-local output directory; likely not user-editable from the UI. |

---

## 5. Recommendation for the settings page structure

- **Tab/section 1: "Image Preparation"** — the three phases from Section 2, each a
  collapsible group, each step a card with an enable toggle, an algorithm dropdown (when
  there's a choice), and a params form that swaps based on the selected algorithm.
- **Tab/section 2: "Recognition & Matching"** — OCR, Table Extraction, String Matching
  from Section 4. Since there's only one option per field today, render these as
  read-only/fixed info or a single pre-selected dropdown entry, but keep the same
  "dropdown + dynamic param form" pattern so adding a second engine later is a drop-in,
  not a redesign.
- Do not expose `table_photometric_steps`, `save_debug_images`, `dictionary_path`, or
  `output`/`persistence` internals to the end user — these are either not implemented,
  server-local, or debug-only.
- Send the not-implemented strategies (`deskew_min_area_rect`,
  `illumination_normalization_blackhat`) only if you choose to surface them as
  disabled/"coming soon" options; otherwise omit them from the dropdown entirely. Either
  way, never let the UI actually select them and submit, since the service will throw at
  run time.
- Whatever the user configures, serialize it into a single JSON object matching Section 1
  and attach it under the extract request's `options`/config field per api-contract.md's
  request shape (confirm the exact key name/location with the API contract owner if it's
  not yet wired into `/extract`'s `options` object).
