"""FastAPI application entry point.

Run:
    python -m api.main
    # or
    uvicorn api.main:app --host 127.0.0.1 --port 8765

This is the process the WinUI desktop app talks to. It wraps
`orchestration.PipelineOrchestrator` -- the same pipeline `main.py` and
`run_batch.py` run from the command line -- in the HTTP contract documented in
`docs/api-contract.md`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.pipeline_pool import PipelinePool
from api.routes import router
from api.service_settings import SERVICE_VERSION, get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("invoice-pipeline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    logger.info("Starting invoice extraction service v%s", SERVICE_VERSION)
    logger.info("Pipeline config: %s", settings.config_path)

    pool = PipelinePool(
        config_path=settings.config_path,
        cache_size=settings.orchestrator_cache_size,
    )
    app.state.pool = pool

    logger.info("OCR engine: %s", pool.engine_name)

    if settings.warmup_on_start:
        try:
            pool.warmup()
            logger.info("OCR engine ready.")
        except Exception:
            # Log and keep serving: /health reports engine_ready=false, so the
            # desktop app shows a clear "service not ready" state instead of the
            # process simply vanishing at startup.
            logger.exception("OCR engine warmup failed; the service will report not-ready.")

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Invoice Extraction Pipeline Service",
    version=SERVICE_VERSION,
    description=(
        "Offline OCR and invoice-structuring service. Binds to loopback only and "
        "never makes outbound network calls."
    ),
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "invoice-extraction-pipeline",
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


def main() -> None:
    import uvicorn

    settings = get_settings()

    # Loopback by default; see ServiceSettings for why that matters.
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
