"""The page, as a set of labelled regions.

This is the output of the layout stage -- a table extractor said where the
boxes are, a layout classifier said what each one means -- and it is what the
whole downstream pipeline reads instead of a bare
:class:`~core.domain.table.TableExtractionResult`.

It exists because a `TableExtractionResult` describes only the item table, and
an invoice is not only its item table: the invoice number, the city, the date,
the merchant and the grand total are all boxes on the same printed form. Mapping
them was previously left to keyword heuristics running over reconstructed text
lines, which cannot see that a value sits inside a ruled box directly under a
printed caption.

Every bbox here is in the one canonical coordinate space -- the geometrically
corrected display image -- exactly like every OCR bbox. See
`core/domain/geometry.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.domain.geometry import BoundingBox
from core.domain.roles import CellRole, ContentKind, Zone, content_kind_for


@dataclass(frozen=True)
class LayoutRegion:
    """One box on the page, with what it means and how it should be read.

    `row`/`col` are the table grid position and are None outside the table.
    `content_kind` is carried explicitly rather than derived from `role` on
    demand, so a classifier can override it for an unusual form without the
    recognizer needing a special case.
    """

    id: str
    bbox: BoundingBox
    role: CellRole = CellRole.UNKNOWN
    zone: Zone = Zone.UNKNOWN
    content_kind: ContentKind = ContentKind.ANY

    row: int | None = None
    col: int | None = None
    row_span: int = 1
    col_span: int = 1
    table_id: str | None = None

    @classmethod
    def from_role(
        cls,
        id: str,
        bbox: BoundingBox,
        role: CellRole,
        zone: Zone,
        **kwargs,
    ) -> "LayoutRegion":
        """Build a region with the content kind implied by its role."""
        return cls(
            id=id,
            bbox=bbox,
            role=role,
            zone=zone,
            content_kind=content_kind_for(role),
            **kwargs,
        )

    @property
    def area(self) -> int:
        return self.bbox.w * self.bbox.h


@dataclass(frozen=True)
class InvoiceLayout:
    """Every labelled region on one page.

    `source` records which extractor and classifier produced this, because the
    two are configured independently and a surprising layout is almost always a
    question about which pair ran.
    """

    regions: list[LayoutRegion] = field(default_factory=list)
    table_bbox: BoundingBox | None = None
    table_id: str | None = None
    source: str = ""
    raw_output: object = None

    def __bool__(self) -> bool:
        return bool(self.regions)

    def in_zone(self, zone: Zone) -> list[LayoutRegion]:
        return [r for r in self.regions if r.zone is zone]

    @property
    def header_regions(self) -> list[LayoutRegion]:
        return self.in_zone(Zone.HEADER)

    @property
    def table_regions(self) -> list[LayoutRegion]:
        return self.in_zone(Zone.TABLE)

    @property
    def footer_regions(self) -> list[LayoutRegion]:
        return self.in_zone(Zone.FOOTER)

    def with_role(self, role: CellRole) -> list[LayoutRegion]:
        return [r for r in self.regions if r.role is role]

    def first_with_role(self, role: CellRole) -> LayoutRegion | None:
        return next((r for r in self.regions if r.role is role), None)

    def by_id(self) -> dict[str, LayoutRegion]:
        return {r.id: r for r in self.regions}

    @property
    def has_roles(self) -> bool:
        """Whether any region was actually classified.

        The passthrough classifier produces regions with UNKNOWN roles, which is
        indistinguishable from "no layout stage ran" as far as the parser is
        concerned -- and the parser needs to know, because it decides between
        reading labelled fields and falling back to keyword heuristics.
        """
        return any(r.role is not CellRole.UNKNOWN for r in self.regions)
