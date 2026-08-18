"""Keeps built pipelines alive between requests.

Constructing a `PipelineOrchestrator` builds preprocessing pipelines, a table
extractor, a layout classifier, a string matcher -- all cheap -- and whichever
OCR components the configured flow needs, which are not: the Surya detector and
the Qwen recognizer load model weights, roughly a minute each. Since every
request may carry its own `configuration`, building an orchestrator per request
would pay that cost per invoice.

Two caches solve it:

* **OCR components**, keyed by the OCR section and the flow name together --
  the two things that decide which models get built. A request that changes a
  threshold block size gets a new orchestrator but the *same* detector and
  recognizer objects, so the weights stay loaded.
* **Orchestrators**, keyed by the whole configuration, bounded and
  least-recently-used. In practice the desktop app sends one configuration for
  a whole batch, so the cache holds a single entry and every request after the
  first is a lookup.

Both caches are guarded by a lock: uvicorn runs the sync endpoint handlers on a
thread pool, so two requests can arrive at the pool genuinely concurrently.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

from api.service_settings import PROJECT_ROOT
from config.loader import load_config
from config.schema import AppConfig
from config.settings_contract import PipelineSettings
from orchestration.flows.components import OCRComponents, build_ocr_components
from orchestration.pipeline_orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


def _key(payload) -> str:
    """A stable cache key for a pydantic model or plain dict."""
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload
    return json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)


def resolve_paths(config: AppConfig) -> AppConfig:
    """Anchor the config's relative paths to the project, not to the CWD.

    The YAML files say `keywords/ar_invoice_terms.json` and `results/`, which
    are correct when the pipeline is run from its own directory as the CLI
    does. A service is started from wherever the desktop app happens to launch
    it, so those paths have to be pinned to the project root or the dictionary
    silently disappears and results land somewhere nobody looks.
    """
    dictionary = Path(config.string_matching.dictionary_path)
    if not dictionary.is_absolute():
        config.string_matching.dictionary_path = str(PROJECT_ROOT / dictionary)

    output_dir = config.persistence.store_params.get("output_dir")
    if output_dir and not Path(output_dir).is_absolute():
        config.persistence.store_params["output_dir"] = str(PROJECT_ROOT / output_dir)

    return config


class PipelinePool:
    def __init__(self, config_path: str, cache_size: int = 4):
        self.config_path = config_path
        self.cache_size = max(1, cache_size)

        # The service's own defaults: what a request with no `configuration`
        # runs, and what the settings page is served as its starting point.
        self.default_config: AppConfig = resolve_paths(load_config(config_path))

        self._lock = threading.Lock()
        self._components: dict[str, OCRComponents] = {}
        self._orchestrators: "OrderedDict[str, PipelineOrchestrator]" = OrderedDict()

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @property
    def default_settings(self) -> PipelineSettings:
        """The default config in the wire shape, for the settings page."""
        return PipelineSettings.from_app_config(self.default_config)

    @property
    def default_table_photometric_steps(self):
        """The table branch's preprocessing, which the settings contract omits."""
        return self.default_config.preprocessing.table_photometric_steps

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        """What /health reports as the engine.

        The flow answers this now, because under the two region-based flows
        there is no single engine -- "surya_detector+qwen" is the honest name
        and `ocr.engine` is not even set.
        """
        try:
            return self.get(self.default_config).engine_name
        except Exception:  # noqa: BLE001 - report the intent when it cannot be built
            return self.default_config.ocr.engine or self.default_config.flow.name

    def is_ready(self, config: AppConfig | None = None) -> bool:
        """Whether what `config` selects can serve a request right now.

        Defaults to the service's own configuration, which is what /health
        reports. An extract request must pass the configuration it is actually
        going to run: a request that selects the stub engine has no business
        being refused because Tesseract is not installed.
        """
        target = config or self.default_config

        try:
            return self.get(target).is_ready()
        except Exception:  # noqa: BLE001 - a missing dependency is "not ready"
            logger.warning(
                "Flow '%s' could not be built for readiness check.", target.flow.name
            )
            return False

    def warmup(self) -> None:
        """Build and warm the default flow, so readiness is not a guess."""
        self.get(self.default_config).warmup()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, config: AppConfig) -> PipelineOrchestrator:
        """The orchestrator for `config`, built once and reused afterwards."""
        cache_key = _key(config)

        with self._lock:
            existing = self._orchestrators.get(cache_key)
            if existing is not None:
                self._orchestrators.move_to_end(cache_key)
                return existing

        # Built outside the lock: constructing an engine can take a minute, and
        # holding the lock through it would serialize every other request
        # behind a cache miss. The worst case is two threads building the same
        # orchestrator concurrently, which wastes work but stays correct.
        components = self._components_for(config)
        orchestrator = PipelineOrchestrator(config, ocr_components=components)

        with self._lock:
            self._orchestrators[cache_key] = orchestrator
            self._orchestrators.move_to_end(cache_key)
            while len(self._orchestrators) > self.cache_size:
                evicted, _ = self._orchestrators.popitem(last=False)
                logger.debug("Evicted a cached pipeline configuration: %s", evicted[:120])

        return orchestrator

    def _components_for(self, config: AppConfig) -> OCRComponents:
        """The built models for this configuration, shared across orchestrators.

        Keyed on the OCR section AND the flow name: the same `ocr` block builds
        different objects under different flows -- layout_driven deliberately
        builds no detector at all -- so the flow has to be part of the identity.
        """
        components_key = _key({"ocr": config.ocr.model_dump(), "flow": config.flow.name})

        with self._lock:
            components = self._components.get(components_key)
        if components is not None:
            return components

        logger.info("Building OCR components for flow '%s'.", config.flow.name)
        components = build_ocr_components(config)

        with self._lock:
            # Another thread may have won the race; keep whichever landed first
            # so there is only ever one instance per OCR configuration.
            return self._components.setdefault(components_key, components)
