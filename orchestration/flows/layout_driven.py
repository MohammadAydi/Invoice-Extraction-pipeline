"""The printed grid decides what gets read.

Layout analysis runs first. Every labelled cell -- header captions, item rows,
totals strip alike -- becomes a region to read, and each is read with the prompt
its *role* implies: the quantity column gets the digits-only prompt, the product
column the Arabic-text prompt, the date box the date prompt.

Three things follow from that, and they are the reason this flow exists:

1. **No text detector runs.** Not "runs and is ignored" -- it is never
   constructed. A cell exists because it is printed on the form, so there is
   nothing to detect. This is what fixes the detector's blind spot for short
   isolated digits: a quantity of "٢" has a box whether or not any detector
   would have found the ink in it.
2. **No column fractions are declared.** The grid supplies the real boundaries,
   measured on this photograph, rather than the 0.163/0.285/0.775 constants
   somebody measured once on one form.
3. **Mapping needs no geometry.** Each fragment carries the id of the region it
   came from, so assignment is by identity, not by guessing containment.

The trade is real and deliberate: text outside a ruled box is not read at all.
On an unruled receipt this flow returns nothing and says so. It does **not**
quietly fall back to the detector -- a silent switch to a different reading
strategy is exactly the kind of invisible behaviour change that makes a bad
extraction impossible to diagnose.
"""

from __future__ import annotations

from core.domain.image_payload import ImagePayload
from core.domain.layout import InvoiceLayout
from core.domain.ocr import OCRResult, TextRegion
from core.domain.roles import CellRole
from orchestration.flows.base import ExtractionFlow
from orchestration.flows.registry import flow_registry

# Cells whose content is printed on the form rather than written on it. Reading
# them costs a model call per cell and tells nobody anything new.
_SKIPPED_ROLES = frozenset({CellRole.COLUMN_HEADER, CellRole.LABEL})


@flow_registry.register("layout_driven")
class LayoutDrivenFlow(ExtractionFlow):
    name = "layout_driven"

    # No detector, and that is the point -- see the module docstring. Because
    # this list is what `build_ocr_components` builds from, one is never
    # constructed, so it cannot be reached by accident from anywhere.
    REQUIRES = ("cropper", "recognizer")

    @classmethod
    def from_config(cls, config, components, table_extractor, layout_classifier, cell_mapper):
        return cls(
            table_extractor,
            layout_classifier,
            cell_mapper,
            cropper=components.cropper,
            recognizer=components.recognizer,
            **config.flow.params,
        )

    def __init__(
        self,
        table_extractor,
        layout_classifier,
        cell_mapper,
        cropper,
        recognizer,
        read_printed_labels: bool = False,
    ):
        super().__init__(table_extractor, layout_classifier, cell_mapper)
        self.cropper = cropper
        self.recognizer = recognizer

        # Off by default. Turn on to render the printed captions in the overlay
        # -- useful while tuning a new form's classifier, expensive otherwise.
        self.read_printed_labels = read_printed_labels

    def _regions(self, layout: InvoiceLayout) -> list[TextRegion]:
        return [
            TextRegion(
                bbox=region.bbox,
                # A printed cell carries no detection score, and there is no
                # recognition score either. None means unknown, which is honest;
                # downstream reads it as "needs a human", not as zero.
                confidence=None,
                content_kind=region.content_kind,
                source_id=region.id,
            )
            for region in layout.regions
            if self.read_printed_labels or region.role not in _SKIPPED_ROLES
        ]

    def _read_text(self, payload: ImagePayload, layout: InvoiceLayout) -> OCRResult:
        regions = self._regions(layout)

        return self._crop_and_read(
            payload,
            regions,
            engine_name=f"layout+{self.recognizer.name}",
            raw_output={"layout_source": layout.source, "regions": len(regions)},
        )

    def _warnings(self, layout: InvoiceLayout, ocr_result: OCRResult) -> list[str]:
        if not layout.regions:
            return [
                "No ruled layout was detected, and the layout_driven flow reads only "
                "cells of a detected grid. Nothing on this page was read. Use the "
                "detector_driven or single_engine flow for unruled documents."
            ]
        if not layout.has_roles:
            return [
                f"The layout was detected ({len(layout.regions)} region(s)) but no "
                f"cell was classified -- '{layout.source}' labelled nothing. Every "
                "cell was still read, but header fields fall back to text heuristics."
            ]
        return []

    def components(self) -> list[object]:
        return super().components() + [self.cropper, self.recognizer]

    @property
    def engine_name(self) -> str:
        return f"layout+{self.recognizer.name}"
