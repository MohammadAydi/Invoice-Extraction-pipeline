"""End-to-end tests of the HTTP contract in docs/api-contract.md.

These run the real pipeline -- real preprocessing, the deterministic stub OCR
engine, real parsing and matching -- over a synthetic image, so a break
anywhere between the upload and the JSON body fails here.
"""

from __future__ import annotations

import base64
import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    # The lifespan handler builds the pipeline pool; TestClient as a context
    # manager is what runs it.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def invoice_png() -> bytes:
    """A blank page. The stub engine derives its output from the pixels, so the
    content does not matter -- only that it decodes and has a realistic size.
    """
    page = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    # A few dark strokes so preprocessing has something to work on.
    cv2.rectangle(page, (60, 300), (940, 900), (0, 0, 0), 2)
    cv2.line(page, (60, 360), (940, 360), (0, 0, 0), 2)

    ok, buffer = cv2.imencode(".png", page)
    assert ok
    return buffer.tobytes()


def post_extract(client, image: bytes, options: dict | None = None, filename="invoice.png"):
    files = {"file": (filename, io.BytesIO(image), "image/png")}
    data = {"options": json.dumps(options)} if options is not None else None
    return client.post("/api/v1/extract", files=files, data=data)


STUB_OPTIONS = {"configuration": None}


def with_stub_engine(options: dict | None = None) -> dict:
    """Force the stub engine, so the tests never need Tesseract installed."""
    from api.pipeline_pool import PipelinePool
    from api.service_settings import get_settings
    from config.settings_contract import PipelineSettings

    pool = PipelinePool(get_settings().config_path)
    settings = PipelineSettings.from_app_config(pool.default_config)
    settings.ocr.engine = "stub"
    settings.ocr.engine_params = {}

    merged = dict(options or {})
    merged["configuration"] = settings.model_dump(mode="json")
    return merged


class TestHealth:
    def test_reports_the_configured_engine(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]
        assert "ocr_engine" in body
        assert isinstance(body["engine_ready"], bool)
        assert body["languages"]

    def test_root_points_at_health(self, client):
        assert client.get("/").json()["health"] == "/api/v1/health"


class TestDefaultConfiguration:
    def test_serves_all_eight_steps_in_the_wire_shape(self, client):
        response = client.get("/api/v1/configuration")
        assert response.status_code == 200

        preprocessing = response.json()["preprocessing"]
        assert set(preprocessing) == {
            "perspective_correction", "deskew", "channel_selection",
            "illumination_normalization", "contrast_enhancement", "denoising",
            "thresholding", "morphological_cleanup",
        }
        for step in preprocessing.values():
            assert set(step) >= {"enabled", "algorithm", "params"}

    def test_round_trips_back_into_extract(self, client, invoice_png):
        configuration = client.get("/api/v1/configuration").json()
        response = post_extract(client, invoice_png, {"configuration": configuration})
        assert response.status_code in (200, 503)


class TestFileValidation:
    def test_rejects_an_unsupported_extension(self, client, invoice_png):
        response = post_extract(client, invoice_png, filename="invoice.docx")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "UNSUPPORTED_FORMAT"

    def test_rejects_an_empty_upload(self, client):
        response = post_extract(client, b"")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "EMPTY_FILE"

    def test_rejects_bytes_that_are_not_an_image(self, client):
        response = post_extract(client, b"this is not a png")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "CORRUPT_FILE"

    def test_rejects_malformed_options(self, client, invoice_png):
        files = {"file": ("invoice.png", io.BytesIO(invoice_png), "image/png")}
        response = client.post(
            "/api/v1/extract", files=files, data={"options": "{not json"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_OPTIONS"

    def test_rejects_an_invalid_configuration(self, client, invoice_png):
        options = with_stub_engine()
        options["configuration"]["preprocessing"]["thresholding"]["algorithm"] = "clahe"

        response = post_extract(client, invoice_png, options)
        assert response.status_code == 422
        assert response.json()["error"]["code"] in (
            "INVALID_OPTIONS", "INVALID_CONFIGURATION"
        )


class TestExtract:
    def test_returns_the_documented_envelope(self, client, invoice_png):
        response = post_extract(client, invoice_png, with_stub_engine())
        assert response.status_code == 200

        body = response.json()
        assert body["request_id"]
        assert body["processing_ms"] >= 0
        assert body["source"]["filename"] == "invoice.png"
        assert body["source"]["width"] > 0
        assert body["source"]["height"] > 0
        assert set(body["header"]) == {
            "merchant_name", "invoice_number", "invoice_date", "city", "total_amount"
        }

    def test_every_field_uses_the_same_envelope(self, client, invoice_png):
        body = post_extract(client, invoice_png, with_stub_engine()).json()

        for name, field in body["header"].items():
            assert set(field) >= {
                "value", "confidence", "raw", "matched_to", "matched_id",
                "matched_name", "match_score", "candidates",
                "requires_manual_review", "bbox",
            }, name
            assert 0.0 <= field["confidence"] <= 1.0

    def test_line_items_carry_an_arithmetic_flag(self, client, invoice_png):
        body = post_extract(client, invoice_png, with_stub_engine()).json()

        assert body["line_items"], "the stub engine always prints an item table"
        for index, item in enumerate(body["line_items"]):
            assert item["row_index"] == index
            assert isinstance(item["arithmetic_ok"], bool)
            assert set(item) >= {
                "row_index", "product_name", "quantity", "unit_price",
                "total_price", "arithmetic_ok",
            }

    def test_quantities_are_integers_and_amounts_are_numbers(self, client, invoice_png):
        body = post_extract(client, invoice_png, with_stub_engine()).json()

        for item in body["line_items"]:
            quantity = item["quantity"]["value"]
            if quantity is not None:
                assert isinstance(quantity, int)
            for key in ("unit_price", "total_price"):
                value = item[key]["value"]
                assert value is None or isinstance(value, (int, float))

    def test_date_is_iso_or_null(self, client, invoice_png):
        value = post_extract(client, invoice_png, with_stub_engine()).json()[
            "header"]["invoice_date"]["value"]

        if value is not None:
            assert len(value) == 10 and value[4] == "-" and value[7] == "-"

    def test_a_known_merchant_is_matched_to_its_record(self, client, invoice_png):
        # Every name the stub can print, so whichever it picks resolves.
        merchants = [
            {"customer_id": 1, "name": "متجر النور"},
            {"customer_id": 2, "name": "مؤسسة الرياض التجارية"},
            {"customer_id": 3, "name": "Ahmad Trading Est."},
            {"customer_id": 4, "name": "سوبرماركت السلام"},
            {"customer_id": 5, "name": "Gulf Office Supplies"},
        ]

        body = post_extract(
            client, invoice_png, with_stub_engine({"known_merchants": merchants})
        ).json()

        merchant = body["header"]["merchant_name"]
        assert merchant["matched_id"] in {1, 2, 3, 4, 5}
        assert merchant["value"] == merchant["matched_to"]
        assert not merchant["requires_manual_review"]

    def test_candidates_are_ranked_and_capped(self, client, invoice_png):
        merchants = [{"customer_id": i, "name": f"متجر رقم {i}"} for i in range(20)]

        body = post_extract(
            client, invoice_png, with_stub_engine({"known_merchants": merchants})
        ).json()

        candidates = body["header"]["merchant_name"]["candidates"]
        assert 0 < len(candidates) <= 5

        scores = [c["similarity_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 100.0 for s in scores)

    def test_max_candidates_is_honoured(self, client, invoice_png):
        merchants = [{"customer_id": i, "name": f"متجر رقم {i}"} for i in range(20)]
        options = with_stub_engine(
            {"known_merchants": merchants, "max_candidates": 2}
        )

        body = post_extract(client, invoice_png, options).json()
        assert len(body["header"]["merchant_name"]["candidates"]) <= 2

    def test_bare_name_strings_are_still_accepted(self, client, invoice_png):
        options = with_stub_engine({"known_merchants": ["متجر النور"]})
        assert post_extract(client, invoice_png, options).status_code == 200

    def test_an_unmatched_field_keeps_the_ocr_text(self, client, invoice_png):
        options = with_stub_engine(
            {"known_merchants": [{"customer_id": 99, "name": "لا شيء مشابه إطلاقا"}]}
        )
        merchant = post_extract(client, invoice_png, options).json()["header"]["merchant_name"]

        assert merchant["matched_id"] is None
        assert merchant["requires_manual_review"]
        assert merchant["value"], "OCR text must survive a failed match"

    def test_debug_images_are_absent_unless_requested(self, client, invoice_png):
        body = post_extract(client, invoice_png, with_stub_engine()).json()
        assert body["enhanced_image_png"] is None
        assert body["ocr_input_image_png"] is None

    def test_debug_images_are_downscaled_png(self, client, invoice_png):
        options = with_stub_engine({"return_debug_images": True})
        body = post_extract(client, invoice_png, options).json()

        for key in ("enhanced_image_png", "ocr_input_image_png"):
            encoded = body[key]
            assert encoded, key

            decoded = cv2.imdecode(
                np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_UNCHANGED
            )
            assert decoded is not None
            assert decoded.shape[1] <= 1200

    def test_overlay_elements_carry_boxes(self, client, invoice_png):
        body = post_extract(client, invoice_png, with_stub_engine()).json()

        assert body["elements"]
        for element in body["elements"]:
            assert len(element["bbox"]) == 4
            assert element["kind"] in ("table_cell", "free_field")

    def test_raw_text_is_returned(self, client, invoice_png):
        body = post_extract(client, invoice_png, with_stub_engine()).json()
        assert body["raw_text"].strip()

    def test_a_borderless_page_still_extracts(self, client):
        """A receipt with no ruled lines or page edge is a normal invoice.

        Both geometric steps have nothing to measure on one -- no long straight
        segments to deskew by, no page contour to crop to. Neither may fail the
        extraction over having no work to do.
        """
        page = np.full((1400, 1000, 3), 255, dtype=np.uint8)
        cv2.putText(page, "receipt", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        ok, buffer = cv2.imencode(".png", page)
        assert ok

        response = post_extract(client, buffer.tobytes(), with_stub_engine())

        assert response.status_code == 200, response.json()
        assert response.json()["line_items"]

    def test_a_blank_page_does_not_crash(self, client):
        page = np.full((900, 700, 3), 255, dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", page)
        assert ok

        response = post_extract(client, buffer.tobytes(), with_stub_engine())
        assert response.status_code == 200, response.json()

    def test_the_same_file_extracts_identically(self, client, invoice_png):
        first = post_extract(client, invoice_png, with_stub_engine()).json()
        second = post_extract(client, invoice_png, with_stub_engine()).json()

        assert first["header"] == second["header"]
        assert first["line_items"] == second["line_items"]
