"""Service-level configuration -- how the process runs, not how a page is read.

Everything that describes *extraction* lives in the pipeline config (a YAML
file, or the `configuration` object a request carries). What is here is the
surrounding service: what address to bind, how large an upload may be, which
config file supplies the defaults. Read from environment variables prefixed
`INVOICE_AI_`, or from a `.env` file beside the service.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_VERSION = "1.0.0"

# 25 MB, matching the API contract. Large enough for a 600 DPI colour scan,
# small enough that a mis-picked file does not stall the service.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})

# The repository root, so default paths work no matter where the process was
# started from. Config files and the keyword dictionary are resolved against it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INVOICE_AI_",
        env_file=".env",
        extra="ignore",
    )

    # --- network ---------------------------------------------------------
    # Loopback only. Binding 0.0.0.0 would expose the service to the LAN and
    # break the offline guarantee, so this default must not be changed casually.
    host: str = "127.0.0.1"
    port: int = 8765

    # --- pipeline defaults ----------------------------------------------
    # The configuration a request inherits when it sends none of its own. The
    # desktop app normally sends a full `configuration` object, but the CLI,
    # the tests and a bare curl all rely on this.
    config_path: str = str(PROJECT_ROOT / "config" / "default_config.yaml")

    default_languages: list[str] = ["ar", "en"]

    # --- behaviour -------------------------------------------------------
    # Load weights / verify binaries at startup rather than on the first
    # request, so the desktop app's readiness poll reflects the truth.
    warmup_on_start: bool = True

    # Widest a returned diagnostic image may be. These travel base64-encoded
    # inside the JSON body, so a full-resolution scan would dominate it.
    debug_image_max_width: int = 1200

    # How many distinct configurations to keep built at once. Each entry holds
    # preprocessing pipelines and a table extractor; the OCR engine is shared
    # across all of them and is never rebuilt by a config change.
    orchestrator_cache_size: int = 4

    # Writing one PNG per preprocessing step per request would make disk I/O
    # the slowest part of the service. On only while diagnosing a bad read.
    write_step_debug_images: bool = False

    # Persist each run's result.json + display image under `results/`. On by
    # default: it is the audit trail the project decided to keep.
    persist_results: bool = True


_settings: ServiceSettings | None = None


def get_settings() -> ServiceSettings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = ServiceSettings()
    return _settings
