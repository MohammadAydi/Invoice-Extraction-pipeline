"""The wire configuration shape, per docs/settings-config-contract.md."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from config.loader import load_config
from config.settings_contract import (
    GEOMETRIC_STEP_KEYS,
    PHOTOMETRIC_STEP_KEYS,
    STEP_KEYS,
    PipelineSettings,
)

VALID = {
    "preprocessing": {
        "perspective_correction": {
            "enabled": True,
            "algorithm": "perspective_correction",
            "params": {"canny_low": 30, "canny_high": 150},
        },
        "deskew": {
            "enabled": True,
            "algorithm": "deskew_hough",
            "params": {"hough_threshold": 100, "min_line_length": 150,
                       "max_line_gap": 10, "max_angle_deg": 20},
        },
        "channel_selection": {
            "enabled": True,
            "algorithm": "channel_selection",
            "params": {"channel": "gray"},
        },
        "illumination_normalization": {
            "enabled": True,
            "algorithm": "illumination_normalization_blur_divide",
            "params": {"blur_kernel": 95},
        },
        "contrast_enhancement": {
            "enabled": True,
            "algorithm": "clahe",
            "params": {"clip_limit": 2.5, "tile_grid_size": [8, 8]},
        },
        "denoising": {
            "enabled": True,
            "algorithm": "bilateral_filter",
            "params": {"d": 20, "sigma_color": 25, "sigma_space": 50},
        },
        "thresholding": {
            "enabled": True,
            "algorithm": "adaptive_threshold",
            "params": {"block_size": 51, "c": 35},
        },
        "morphological_cleanup": {
            "enabled": True,
            "algorithm": "morphological_cleanup",
            "params": {"operation": "open", "kernel_size": 2},
        },
    },
    "ocr": {"engine": "tesseract", "engine_params": {"lang": "ara"}},
    "table_extraction": {"extractor": "grid_line", "extractor_params": {}},
    "string_matching": {
        "algorithm": "levenshtein",
        "algorithm_params": {"max_distance": 2},
        "dictionary_path": "keywords/ar_invoice_terms.json",
    },
    "output": {"formatter": "ui_overlay_json", "formatter_params": {}},
    "persistence": {"store": "file_result_store", "store_params": {"output_dir": "results/"}},
}


def payload(**overrides):
    data = copy.deepcopy(VALID)
    for key, value in overrides.items():
        data["preprocessing"][key] = value
    return data


class TestValidation:
    def test_the_documented_shape_validates(self):
        settings = PipelineSettings.model_validate(VALID)
        assert settings.ocr.engine == "tesseract"

    def test_every_step_key_is_required(self):
        data = copy.deepcopy(VALID)
        del data["preprocessing"]["denoising"]

        with pytest.raises(ValidationError):
            PipelineSettings.model_validate(data)

    def test_an_algorithm_from_the_wrong_step_is_rejected(self):
        data = payload(
            thresholding={"enabled": True, "algorithm": "clahe", "params": {}}
        )
        with pytest.raises(ValidationError, match="not a valid algorithm for step"):
            PipelineSettings.model_validate(data)

    def test_an_unimplemented_algorithm_is_refused_at_the_boundary(self):
        data = payload(
            deskew={"enabled": True, "algorithm": "deskew_min_area_rect", "params": {}}
        )
        with pytest.raises(ValidationError, match="not implemented"):
            PipelineSettings.model_validate(data)

    def test_an_unimplemented_algorithm_is_fine_while_disabled(self):
        data = payload(
            deskew={"enabled": False, "algorithm": "deskew_min_area_rect", "params": {}}
        )
        settings = PipelineSettings.model_validate(data)
        assert settings.preprocessing.deskew.enabled is False


class TestTranslation:
    def test_execution_order_is_fixed_by_the_backend(self):
        # Reversing the JSON key order must not change the pipeline order.
        data = copy.deepcopy(VALID)
        data["preprocessing"] = dict(reversed(list(data["preprocessing"].items())))

        config = PipelineSettings.model_validate(data).to_app_config()

        assert [s.name for s in config.preprocessing.geometric_steps] == [
            "perspective_correction",
            "deskew_hough",
        ]
        assert [s.name for s in config.preprocessing.ocr_photometric_steps] == [
            "channel_selection",
            "illumination_normalization_blur_divide",
            "clahe",
            "bilateral_filter",
            "adaptive_threshold",
            "morphological_cleanup",
        ]

    def test_disabled_steps_stay_present_but_off(self):
        data = payload(
            contrast_enhancement={"enabled": False, "algorithm": "clahe", "params": {}}
        )
        config = PipelineSettings.model_validate(data).to_app_config()

        clahe = next(
            s for s in config.preprocessing.ocr_photometric_steps if s.name == "clahe"
        )
        assert clahe.enabled is False

    def test_params_pass_through_untouched(self):
        config = PipelineSettings.model_validate(VALID).to_app_config()
        threshold = next(
            s for s in config.preprocessing.ocr_photometric_steps
            if s.name == "adaptive_threshold"
        )
        assert threshold.params == {"block_size": 51, "c": 35}

    def test_table_branch_comes_from_the_server_not_the_contract(self):
        from config.schema import StepConfig

        server_side = [StepConfig(name="channel_selection", params={"channel": "gray"})]
        config = PipelineSettings.model_validate(VALID).to_app_config(
            table_photometric_steps=server_side
        )

        assert [s.name for s in config.preprocessing.table_photometric_steps] == [
            "channel_selection"
        ]


class TestRoundTrip:
    def test_the_shipped_default_config_survives_a_round_trip(self):
        original = load_config("config/default_config.yaml")

        wire = PipelineSettings.from_app_config(original)
        restored = wire.to_app_config(
            table_photometric_steps=original.preprocessing.table_photometric_steps
        )

        assert restored.preprocessing == original.preprocessing
        assert restored.ocr == original.ocr
        assert restored.string_matching == original.string_matching

    def test_from_app_config_exposes_all_eight_keys(self):
        wire = PipelineSettings.from_app_config(load_config("config/default_config.yaml"))
        dumped = wire.model_dump()["preprocessing"]

        assert set(dumped) == set(STEP_KEYS)
        assert len(GEOMETRIC_STEP_KEYS) + len(PHOTOMETRIC_STEP_KEYS) == 8
