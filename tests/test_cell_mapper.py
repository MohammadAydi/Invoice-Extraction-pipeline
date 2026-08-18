"""Mapping fragments onto the page's labelled regions.

CellMapper used to take a TableExtractionResult and therefore only knew about
table cells. The invoice number, city, date, merchant and totals boxes were left
to the parser's keyword heuristics running over reconstructed text lines -- which
cannot see that a value sits inside a ruled box under a printed caption.
"""

from __future__ import annotations

from core.domain.document import FreeFieldElement, TableCellElement
from core.domain.geometry import BoundingBox
from core.domain.layout import InvoiceLayout, LayoutRegion
from core.domain.ocr import OCRFragment, OCRResult
from core.domain.roles import CellRole, Zone
from mapping.cell_mapper import CellMapper

TABLE_ID = "t1"


def fragment(text: str, x: int, y: int, w: int = 60, h: int = 20, source_id: str = None):
    return OCRFragment(
        text=text,
        bbox=BoundingBox(x=x, y=y, w=w, h=h),
        confidence=0.9,
        source_id=source_id,
    )


def ocr(*fragments) -> OCRResult:
    return OCRResult(fragments=list(fragments), engine_name="test")


def table_region(id: str, x: int, y: int, row: int, col: int, role=CellRole.UNKNOWN, w=100, h=50):
    return LayoutRegion.from_role(
        id=id,
        bbox=BoundingBox(x=x, y=y, w=w, h=h),
        role=role,
        zone=Zone.TABLE,
        row=row,
        col=col,
        table_id=TABLE_ID,
    )


def field_region(id: str, x: int, y: int, role: CellRole, zone: Zone, w=200, h=50):
    return LayoutRegion.from_role(
        id=id, bbox=BoundingBox(x=x, y=y, w=w, h=h), role=role, zone=zone
    )


class TestIdentityAssignment:
    """The layout-driven path: the crop came from a known region, so which
    region it belongs to is a fact, not something to re-derive from geometry."""

    def test_a_fragment_goes_to_the_region_it_was_cropped_from(self):
        layout = InvoiceLayout(
            regions=[
                table_region("a", 0, 0, 0, 0),
                table_region("b", 100, 0, 0, 1),
            ]
        )
        # Deliberately positioned over region "a" while claiming to come from
        # "b": identity has to win, or the fact is being thrown away.
        document = CellMapper().map(ocr(fragment("2", 10, 10, source_id="b")), layout)

        by_id = {el.id: el for el in document.elements}
        assert by_id["b"].merged_text == "2"
        assert by_id["a"].merged_text == ""

    def test_an_unknown_source_id_is_not_papered_over_with_a_guess(self):
        """A fragment naming a region the layout does not have means the two
        came from different runs. That is a bug, and falling back to geometry
        would hide it."""
        layout = InvoiceLayout(regions=[table_region("a", 0, 0, 0, 0)])
        document = CellMapper().map(ocr(fragment("2", 10, 10, source_id="ghost")), layout)

        free = [el for el in document.elements if isinstance(el, FreeFieldElement)]
        assert [f.merged_text for f in free] == ["2"]


class TestCentroidAssignment:
    """The detector-driven and single-engine paths, where nothing knows which
    region a fragment came from."""

    def test_fragments_land_in_the_region_containing_their_centroid(self):
        layout = InvoiceLayout(
            regions=[table_region("a", 0, 0, 0, 0), table_region("b", 100, 0, 0, 1)]
        )
        document = CellMapper().map(
            ocr(fragment("سكر", 10, 10, w=30), fragment("2", 110, 10, w=20)), layout
        )

        by_id = {el.id: el for el in document.elements}
        assert by_id["a"].merged_text == "سكر"
        assert by_id["b"].merged_text == "2"

    def test_the_smallest_containing_region_wins(self):
        """A layout can carry a whole-table box alongside its cells, and a
        fragment inside both belongs to the cell."""
        layout = InvoiceLayout(
            regions=[
                field_region("whole", 0, 0, CellRole.UNKNOWN, Zone.TABLE, w=400, h=200),
                table_region("cell", 10, 10, 0, 0, w=80, h=40),
            ]
        )
        document = CellMapper().map(ocr(fragment("سكر", 20, 20, w=30)), layout)

        by_id = {el.id: el for el in document.elements}
        assert by_id["cell"].merged_text == "سكر"
        assert by_id["whole"].merged_text == ""

    def test_a_fragment_outside_every_region_becomes_a_free_field(self):
        layout = InvoiceLayout(regions=[table_region("a", 0, 0, 0, 0)])
        document = CellMapper().map(ocr(fragment("خارج", 10, 400)), layout)

        free = [el for el in document.elements if isinstance(el, FreeFieldElement)]
        assert [f.merged_text for f in free] == ["خارج"]


class TestWholePageMapping:
    def test_header_and_footer_regions_are_mapped_too(self):
        """The change this class exists for: everything on the page, not only
        the item table."""
        layout = InvoiceLayout(
            regions=[
                field_region("no", 0, 0, CellRole.INVOICE_NUMBER, Zone.HEADER),
                field_region("city", 0, 60, CellRole.CITY, Zone.HEADER),
                table_region("cell", 0, 200, 0, 0),
                field_region("total", 0, 400, CellRole.TOTAL_AMOUNT, Zone.FOOTER),
            ]
        )
        document = CellMapper().map(
            ocr(
                fragment("00008", 10, 10),
                fragment("دمشق", 10, 70),
                fragment("سكر", 10, 210),
                fragment("795.25", 10, 410),
            ),
            layout,
        )

        by_id = {el.id: el for el in document.elements}
        assert by_id["no"].merged_text == "00008"
        assert by_id["city"].merged_text == "دمشق"
        assert by_id["total"].merged_text == "795.25"

    def test_roles_and_zones_reach_the_elements(self):
        layout = InvoiceLayout(
            regions=[field_region("city", 0, 0, CellRole.CITY, Zone.HEADER)]
        )
        document = CellMapper().map(ocr(fragment("دمشق", 10, 10)), layout)

        element = document.elements[0]
        assert element.role is CellRole.CITY
        assert element.zone is Zone.HEADER
        assert document.has_roles

    def test_a_table_region_becomes_a_table_cell_and_a_field_a_free_field(self):
        layout = InvoiceLayout(
            regions=[
                table_region("cell", 0, 200, 3, 2),
                field_region("city", 0, 0, CellRole.CITY, Zone.HEADER),
            ]
        )
        document = CellMapper().map(ocr(), layout)

        by_id = {el.id: el for el in document.elements}
        assert isinstance(by_id["cell"], TableCellElement)
        assert by_id["cell"].row == 3 and by_id["cell"].col == 2
        assert isinstance(by_id["city"], FreeFieldElement)

    def test_empty_regions_are_kept(self):
        """A cell the recognizer could not read still needs a box to click on,
        and a missing element would silently shift the row."""
        layout = InvoiceLayout(regions=[table_region("a", 0, 0, 0, 0)])
        document = CellMapper().map(ocr(), layout)

        assert len(document.elements) == 1
        assert document.elements[0].merged_text == ""

    def test_an_empty_layout_makes_everything_a_free_field(self):
        document = CellMapper().map(ocr(fragment("سكر", 10, 10)), InvoiceLayout())

        assert len(document.elements) == 1
        assert isinstance(document.elements[0], FreeFieldElement)


class TestReadingOrder:
    def test_fragments_in_one_region_merge_right_to_left(self):
        """Arabic reading order matters even inside one cell: a two-line
        description merged the wrong way scores against the catalog as a
        different product."""
        layout = InvoiceLayout(regions=[table_region("a", 0, 0, 0, 0, w=300, h=60)])
        document = CellMapper().map(
            ocr(fragment("أزرق", 20, 10, w=60), fragment("جاكيت", 150, 10, w=60)), layout
        )

        assert document.elements[0].merged_text == "جاكيت أزرق"
