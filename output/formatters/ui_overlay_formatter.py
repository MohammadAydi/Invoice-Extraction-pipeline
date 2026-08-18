"""Serializes a PipelineResult into the JSON body the desktop app consumes.

This is the Python half of the `/api/v1/extract` 200 response documented in
`docs/api-contract.md`; the C# DTOs in
`InvoiceDigitizationApp/Services/AiServiceClient/Contracts.cs` are the other
half. Change the document first, then both sides.

The formatter produces the *body* of the extraction -- header, line items,
warnings, overlay elements, raw text. The envelope around it (request id,
timings, source dimensions, debug images) belongs to the HTTP layer, which is
the only place that knows about the request, and is added in `api/routes.py`.
"""

from __future__ import annotations

from core.domain.document import TableCellElement
from core.domain.result import PipelineResult
from output.registry import formatter_registry


@formatter_registry.register("ui_overlay_json")
class UIOverlayFormatter:
    """The default formatter. `include_elements` can be turned off for clients
    that only want the invoice fields: the overlay array is one entry per
    detected box and dominates the payload on a dense invoice.
    """

    def __init__(self, include_elements: bool = True, **params):
        self.include_elements = include_elements
        self.params = params

    def format(self, result: PipelineResult) -> dict:
        invoice = result.invoice

        payload: dict = {
            "invoice_id": result.invoice_id,
            "pipeline_version": result.pipeline_version,
            "ocr_engine": result.ocr_engine,
            "raw_text": result.raw_text,
            "header": invoice.header.to_wire() if invoice else {},
            "line_items": [item.to_wire() for item in invoice.line_items] if invoice else [],
            "warnings": [w.to_wire() for w in invoice.warnings] if invoice else [],
        }

        if self.include_elements:
            payload["elements"] = [
                self._element_to_wire(matched) for matched in result.elements
            ]

        return payload

    @staticmethod
    def _element_to_wire(matched) -> dict:
        """One overlay box: where it is, what was read, and what it may be.

        `editable` is always true today -- every box on the verification screen
        can be corrected by hand -- but it is sent explicitly so a future
        read-only element type does not need a contract change.
        """
        element = matched.element
        bbox = element.bbox

        wire = {
            "id": element.id,
            "kind": element.kind,
            # What the layout stage decided this box holds, and roughly where it
            # sits. "unknown" when no classifier ran, which is the honest answer
            # and what every configuration written before classification says.
            "role": element.role.value,
            "zone": element.zone.value,
            "bbox": [bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h],
            "raw_text": element.merged_text,
            "corrected_text": matched.match.corrected_text,
            "confidence": round(float(matched.match.confidence or 0.0), 4),
            "candidates": matched.match.candidates,
            "editable": True,
        }

        if isinstance(element, TableCellElement):
            wire["table_id"] = element.table_id
            wire["row"] = element.row
            wire["col"] = element.col
            wire["row_span"] = element.row_span
            wire["col_span"] = element.col_span

        return wire
