"""The classifier that classifies nothing.

Default, and deliberately so. Every detected cell becomes a LayoutRegion with
role UNKNOWN, which is exactly the information a bare TableExtractionResult
carried before layout classification existed. Downstream, `InvoiceLayout.has_roles`
is then False and the parser uses its keyword heuristics -- so a configuration
that does not opt into a form-specific classifier behaves precisely as it did
before.

It also has a real job of its own: an unruled receipt, or a supplier's form
nobody has written a classifier for, still needs its cells mapped and its text
read. Zones are assigned from geometry alone, which needs no knowledge of the
form.
"""

from __future__ import annotations

from core.domain.layout import InvoiceLayout, LayoutRegion
from core.domain.roles import CellRole, ContentKind, Zone
from core.domain.table import TableExtractionResult
from table_extraction.classifiers.registry import classifier_registry


@classifier_registry.register("passthrough")
class PassthroughLayoutClassifier:
    name = "passthrough"

    def __init__(self, **params):
        self.params = params

    def classify(
        self, table: TableExtractionResult, image_size: tuple[int, int]
    ) -> InvoiceLayout:
        regions: list[LayoutRegion] = []

        for structure in table.tables:
            top = structure.bbox.y
            bottom = structure.bbox.y + structure.bbox.h

            for cell in structure.cells:
                center_y = cell.bbox.y + cell.bbox.h / 2.0
                if center_y < top:
                    zone = Zone.HEADER
                elif center_y > bottom:
                    zone = Zone.FOOTER
                else:
                    zone = Zone.TABLE

                regions.append(
                    LayoutRegion(
                        id=f"{cell.table_id}:r{cell.row}c{cell.col}",
                        bbox=cell.bbox,
                        role=CellRole.UNKNOWN,
                        zone=zone,
                        content_kind=ContentKind.ANY,
                        row=cell.row,
                        col=cell.col,
                        row_span=cell.row_span,
                        col_span=cell.col_span,
                        table_id=cell.table_id,
                    )
                )

        first = table.tables[0] if table.tables else None
        return InvoiceLayout(
            regions=regions,
            table_bbox=first.bbox if first else None,
            table_id=first.table_id if first else None,
            source=f"{table.extractor_name}+{self.name}",
            raw_output=table.raw_extractor_output,
        )
