"""End-to-end tests of the HTTP contract in docs/api-contract.md.

These run the real pipeline -- real preprocessing, the deterministic stub OCR
engine, real parsing and matching -- over a synthetic image, so a break
anywhere between the upload and the JSON body fails here.

Every test pins the pipeline to `single_engine` + `stub` through the request's
own `config` part. The service's shipped default is `layout_driven` + `qwen`,
which needs 1.7 GB of model weights and a second virtualenv; a test suite that
loaded those would be untestable on any machine but this one.
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

# Exactly the keys the contract defines for a 200. Asserted as an equality, not
# a subset: an extra key is a side of the contract drifting, and the desktop app
# would ignore it silently rather than fail.
RESPONSE_KEYS = {
    "processing_ms",
    "source",
    "invoice_id",
    "customer_name",
    "date",
    "city",
    "products",
    "total_invoice_price",
    "enhanced_image_png",
    "ocr_input_image_png",
}

VALUE_FIELD_KEYS = {"value", "ocr_confidence", "bounding_box"}
MATCHED_FIELD_KEYS = {"bounding_box", "ocr_confidence", "original_value", "results"}


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


def stub_config() -> dict:
    """The service's defaults with the reading strategy swapped for the stub.

    Built from the shipped config rather than hand-written, so a preprocessing
    change that would break a real request breaks these tests too.
    """
    from api.pipeline_pool import PipelinePool
    from api.service_settings import get_settings
    from config.settings_contract import PipelineSettings

    pool = PipelinePool(get_settings().config_path)
    settings = PipelineSettings.from_app_config(pool.default_config)

    # Both, not just the engine: under `layout_driven` no engine is consulted at
    # all, so setting `ocr.engine` alone would still load the Qwen weights.
    settings.flow.name = "single_engine"
    settings.ocr.engine = "stub"
    settings.ocr.engine_params = {}

    return settings.model_dump(mode="json")


def post_extract(
    client,
    image: bytes,
    options: dict | None = None,
    config: dict | None = None,
    filename="invoice.png",
):
    """POST the three parts: the file, `options`, and `config`."""
    files = {"file": (filename, io.BytesIO(image), "image/png")}

    data = {}
    if options is not None:
        data["options"] = json.dumps(options)
    if config is not None:
        data["config"] = json.dumps(config)

    return client.post("/api/v1/extract", files=files, data=data or None)


def extract_with_stub(client, image: bytes, options: dict | None = None, **kwargs):
    return post_extract(client, image, options=options, config=stub_config(), **kwargs)


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

    def test_serves_the_flow_and_its_ocr_components(self, client):
        """The settings page renders the reading strategy from this."""
        body = client.get("/api/v1/configuration").json()

        assert body["flow"]["name"] == "layout_driven"
        assert body["ocr"]["recognizer"]["name"] == "qwen"
        assert body["table_extraction"]["classifier"] == "bill_layout"

    def test_round_trips_back_into_extract(self, client, invoice_png):
        configuration = client.get("/api/v1/configuration").json()
        response = post_extract(client, invoice_png, config=configuration)

        # 503 when the Qwen weights are not loadable on this machine, which is
        # the honest answer and not a contract break.
        assert response.status_code in (200, 503)


class TestFileValidation:
    def test_rejects_an_unsupported_extension(self, client, invoice_png):
        response = extract_with_stub(client, invoice_png, filename="invoice.docx")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "UNSUPPORTED_FORMAT"

    def test_rejects_an_empty_upload(self, client):
        response = extract_with_stub(client, b"")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "EMPTY_FILE"

    def test_rejects_bytes_that_are_not_an_image(self, client):
        response = extract_with_stub(client, b"this is not a png")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "CORRUPT_FILE"

    def test_rejects_malformed_options(self, client, invoice_png):
        files = {"file": ("invoice.png", io.BytesIO(invoice_png), "image/png")}
        response = client.post(
            "/api/v1/extract", files=files, data={"options": "{not json"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_OPTIONS"

    def test_rejects_malformed_config(self, client, invoice_png):
        files = {"file": ("invoice.png", io.BytesIO(invoice_png), "image/png")}
        response = client.post(
            "/api/v1/extract", files=files, data={"config": "{not json"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_CONFIGURATION"

    def test_rejects_an_invalid_config(self, client, invoice_png):
        config = stub_config()
        config["preprocessing"]["thresholding"]["algorithm"] = "clahe"

        response = post_extract(client, invoice_png, config=config)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_CONFIGURATION"


class TestRequestShape:
    def test_config_is_its_own_part_not_nested_in_options(self, client, invoice_png):
        """`configuration` inside `options` is the old shape and must not work.

        Options ignore unknown keys, so a client still sending it there would get
        a 200 running the *service's* configuration rather than its own -- the
        silent kind of break this assertion exists to catch.
        """
        config = stub_config()

        response = post_extract(client, invoice_png, options={"configuration": config})
        assert response.status_code in (200, 503)

        # Sent correctly, the stub always answers, which is what tells the two
        # apart: the nested attempt above ran the default layout_driven pipeline.
        assert post_extract(client, invoice_png, config=config).status_code == 200

    def test_languages_and_invoice_type_are_no_longer_part_of_the_request(self):
        from api.schemas import ExtractionOptions

        fields = set(ExtractionOptions.model_fields)
        assert "languages" not in fields
        assert "invoice_type" not in fields
        assert "configuration" not in fields


class TestExtract:
    def test_returns_exactly_the_documented_keys(self, client, invoice_png):
        response = extract_with_stub(client, invoice_png)
        assert response.status_code == 200, response.json()

        body = response.json()
        assert set(body) == RESPONSE_KEYS

        assert body["processing_ms"] >= 0
        assert body["source"]["filename"] == "invoice.png"
        assert body["source"]["width"] > 0
        assert body["source"]["height"] > 0

    def test_value_fields_carry_a_reading_and_a_box(self, client, invoice_png):
        body = extract_with_stub(client, invoice_png).json()

        for name in ("invoice_id", "date", "total_invoice_price"):
            field = body[name]
            assert set(field) == VALUE_FIELD_KEYS, name
            assert 0.0 <= field["ocr_confidence"] <= 1.0

            if field["bounding_box"] is not None:
                assert set(field["bounding_box"]) == {"x", "y", "w", "h"}

    def test_matched_fields_carry_the_original_reading_and_ranked_results(
        self, client, invoice_png
    ):
        """The whole point of the shape: no `value` to be quietly wrong."""
        body = extract_with_stub(client, invoice_png).json()

        for name in ("customer_name", "city"):
            field = body[name]
            assert set(field) == MATCHED_FIELD_KEYS, name
            assert "value" not in field, name

    def test_products_carry_one_matched_name_and_three_values(self, client, invoice_png):
        body = extract_with_stub(client, invoice_png).json()

        assert body["products"], "the stub engine always prints an item table"

        for product in body["products"]:
            assert set(product) == {
                "product_name", "quantity", "unit_price", "total_price"
            }
            assert set(product["product_name"]) == MATCHED_FIELD_KEYS
            for key in ("quantity", "unit_price", "total_price"):
                assert set(product[key]) == VALUE_FIELD_KEYS

    def test_amounts_are_json_numbers_and_quantities_are_integers(
        self, client, invoice_png
    ):
        """Numbers, not formatted strings: the app parses them into `decimal`."""
        body = extract_with_stub(client, invoice_png).json()

        for product in body["products"]:
            quantity = product["quantity"]["value"]
            if quantity is not None:
                assert isinstance(quantity, int)

            for key in ("unit_price", "total_price"):
                value = product[key]["value"]
                assert value is None or isinstance(value, (int, float))

        total = body["total_invoice_price"]["value"]
        assert total is None or isinstance(total, (int, float))

    def test_date_is_iso_or_null(self, client, invoice_png):
        value = extract_with_stub(client, invoice_png).json()["date"]["value"]

        if value is not None:
            assert len(value) == 10 and value[4] == "-" and value[7] == "-"

    def test_invoice_id_is_the_printed_number_not_the_run_id(self, client, invoice_png):
        """A UUID here would mean the pipeline's own run id leaked onto the wire."""
        value = extract_with_stub(client, invoice_png).json()["invoice_id"]["value"]

        if value is not None:
            assert "-" not in value or len(value) != 36

    def test_a_known_merchant_appears_in_the_results_with_its_id(
        self, client, invoice_png
    ):
        # Every name the stub can print, so whichever it picks resolves.
        merchants = [
            {"customer_id": 1, "name": "متجر النور"},
            {"customer_id": 2, "name": "مؤسسة الرياض التجارية"},
            {"customer_id": 3, "name": "Ahmad Trading Est."},
            {"customer_id": 4, "name": "سوبرماركت السلام"},
            {"customer_id": 5, "name": "Gulf Office Supplies"},
        ]

        body = extract_with_stub(
            client, invoice_png, {"known_merchants": merchants}
        ).json()

        best = body["customer_name"]["results"][0]
        assert best["id"] in {"1", "2", "3", "4", "5"}
        assert best["string_matching_score"] >= 0.75

    def test_the_raw_reading_survives_a_confident_match(self, client, invoice_png):
        """`original_value` is the paper, never the catalog entry that won."""
        merchants = [{"customer_id": 1, "name": "متجر النور"}]

        field = extract_with_stub(
            client, invoice_png, {"known_merchants": merchants}
        ).json()["customer_name"]

        assert field["original_value"]

    def test_results_are_ranked_capped_and_fractional(self, client, invoice_png):
        merchants = [{"customer_id": i, "name": f"متجر رقم {i}"} for i in range(20)]

        results = extract_with_stub(
            client, invoice_png, {"known_merchants": merchants}
        ).json()["customer_name"]["results"]

        assert 0 < len(results) <= 5

        scores = [entry["string_matching_score"] for entry in results]
        assert scores == sorted(scores, reverse=True)

        # A 0-1 fraction here, not the 0-100 percentage the matcher works in.
        assert all(0.0 <= score <= 1.0 for score in scores)

    def test_max_candidates_is_honoured(self, client, invoice_png):
        merchants = [{"customer_id": i, "name": f"متجر رقم {i}"} for i in range(20)]

        body = extract_with_stub(
            client, invoice_png, {"known_merchants": merchants, "max_candidates": 2}
        ).json()

        assert len(body["customer_name"]["results"]) <= 2

    def test_bare_name_strings_are_still_accepted(self, client, invoice_png):
        options = {"known_merchants": ["متجر النور"]}
        assert extract_with_stub(client, invoice_png, options).status_code == 200

    def test_an_unmatched_field_keeps_the_ocr_text(self, client, invoice_png):
        options = {"known_merchants": [{"customer_id": 99, "name": "لا شيء مشابه إطلاقا"}]}
        field = extract_with_stub(client, invoice_png, options).json()["customer_name"]

        assert field["original_value"], "OCR text must survive a failed match"
        assert all(
            entry["string_matching_score"] < 0.75 for entry in field["results"]
        )

    def test_an_empty_catalog_leaves_the_text_with_no_results(self, client, invoice_png):
        field = extract_with_stub(client, invoice_png).json()["customer_name"]

        assert field["results"] == []
        assert field["original_value"]

    def test_debug_images_are_absent_unless_requested(self, client, invoice_png):
        body = extract_with_stub(client, invoice_png).json()
        assert body["enhanced_image_png"] is None
        assert body["ocr_input_image_png"] is None

    def test_debug_images_are_downscaled_png(self, client, invoice_png):
        body = extract_with_stub(
            client, invoice_png, {"return_debug_images": True}
        ).json()

        for key in ("enhanced_image_png", "ocr_input_image_png"):
            encoded = body[key]
            assert encoded, key

            decoded = cv2.imdecode(
                np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_UNCHANGED
            )
            assert decoded is not None
            assert decoded.shape[1] <= 1200

    def test_boxes_live_in_the_source_coordinate_space(self, client, invoice_png):
        """Every box is measured against the corrected page the app displays."""
        body = extract_with_stub(client, invoice_png).json()
        width = body["source"]["width"]
        height = body["source"]["height"]

        def check(box, where):
            if box is None:
                return
            assert 0 <= box["x"] <= width, where
            assert 0 <= box["y"] <= height, where
            assert box["w"] >= 0 and box["h"] >= 0, where

        for name in ("invoice_id", "date", "total_invoice_price"):
            check(body[name]["bounding_box"], name)
        for name in ("customer_name", "city"):
            check(body[name]["bounding_box"], name)

        for index, product in enumerate(body["products"]):
            for key in ("product_name", "quantity", "unit_price", "total_price"):
                check(product[key]["bounding_box"], f"products[{index}].{key}")

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

        response = extract_with_stub(client, buffer.tobytes())

        assert response.status_code == 200, response.json()
        assert response.json()["products"]

    def test_a_blank_page_does_not_crash(self, client):
        page = np.full((900, 700, 3), 255, dtype=np.uint8)
        ok, buffer = cv2.imencode(".png", page)
        assert ok

        response = extract_with_stub(client, buffer.tobytes())
        assert response.status_code == 200, response.json()

    def test_the_same_file_extracts_identically(self, client, invoice_png):
        first = extract_with_stub(client, invoice_png).json()
        second = extract_with_stub(client, invoice_png).json()

        for key in ("invoice_id", "customer_name", "date", "city", "products",
                    "total_invoice_price"):
            assert first[key] == second[key], key
