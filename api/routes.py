"""API routes. Mirrors docs/api-contract.md exactly."""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.images import CorruptFileError, encode_png_base64, load_document
from api.pipeline_pool import PipelinePool, resolve_paths
from api.schemas import (
    ErrorBody,
    ErrorCodes,
    ErrorEnvelope,
    ExtractionOptions,
    ExtractionResult,
    ExtractionSource,
    ExtractionWarning,
    HealthStatus,
    WarningCodes,
)
from api.service_settings import (
    MAX_UPLOAD_BYTES,
    SERVICE_VERSION,
    SUPPORTED_EXTENSIONS,
    get_settings,
)
from config.schema import AppConfig
from core.domain.image_payload import ImagePayload
from core.exceptions import ConfigurationError, StepExecutionError
from core.domain.catalog import Catalogs
from core.domain.catalog import NamedEntry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def error_response(
    status: int, code: str, message: str, detail: str | None = None
) -> JSONResponse:
    envelope = ErrorEnvelope(error=ErrorBody(code=code, message=message, detail=detail))
    return JSONResponse(status_code=status, content=envelope.model_dump())


def _pool(request: Request) -> PipelinePool:
    return request.app.state.pool


@router.get("/health", response_model=HealthStatus)
def health(request: Request) -> HealthStatus:
    pool: PipelinePool | None = getattr(request.app.state, "pool", None)
    settings = get_settings()

    return HealthStatus(
        status="ok",
        version=SERVICE_VERSION,
        ocr_engine=pool.engine_name if pool else "none",
        engine_ready=bool(pool and pool.is_ready()),
        languages=settings.default_languages,
    )


@router.get("/configuration")
def default_configuration(request: Request) -> JSONResponse:
    """The service's own default pipeline configuration, in the wire shape.

    The settings page loads this the first time it opens, so the defaults the
    user sees are the ones the service actually runs rather than a second copy
    hardcoded in the desktop app that quietly drifts from it.
    """
    pool = _pool(request)
    return JSONResponse(
        status_code=200, content=pool.default_settings.model_dump(mode="json")
    )


@router.post("/extract")
async def extract(
    request: Request,
    file: UploadFile = File(...),
    options: str | None = Form(default=None),
) -> JSONResponse:
    started = time.perf_counter()
    settings = get_settings()

    # --- options ---------------------------------------------------------
    if options:
        try:
            parsed_options = ExtractionOptions.model_validate(json.loads(options))
        except (json.JSONDecodeError, ValidationError) as exc:
            return error_response(
                422,
                ErrorCodes.INVALID_OPTIONS,
                "The 'options' part was not valid JSON matching the expected schema.",
                str(exc),
            )
    else:
        parsed_options = ExtractionOptions()

    # --- file validation -------------------------------------------------
    filename = file.filename or "upload"
    extension = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""

    if extension not in SUPPORTED_EXTENSIONS:
        return error_response(
            400,
            ErrorCodes.UNSUPPORTED_FORMAT,
            f"File extension '{extension or filename}' is not a supported invoice format.",
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    data = await file.read()

    if not data:
        return error_response(400, ErrorCodes.EMPTY_FILE, "The uploaded file was empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        return error_response(
            413,
            ErrorCodes.FILE_TOO_LARGE,
            f"The file is {len(data) / 1_048_576:.1f} MB, over the "
            f"{MAX_UPLOAD_BYTES / 1_048_576:.0f} MB limit.",
        )

    pool = _pool(request)

    # --- configuration ---------------------------------------------------
    try:
        config = _resolve_config(pool, parsed_options)
    except (ValidationError, ValueError) as exc:
        return error_response(
            422,
            ErrorCodes.INVALID_CONFIGURATION,
            "The 'configuration' object was rejected by the pipeline.",
            str(exc),
        )

    # --- decode ----------------------------------------------------------
    # Before the readiness check, deliberately. A file that cannot be decoded is
    # a client error whatever the engine is doing, and answering "the engine is
    # still loading" would send the user off to wait for something that was
    # never going to help.
    try:
        document = load_document(data, filename)
    except CorruptFileError as exc:
        return error_response(400, ErrorCodes.CORRUPT_FILE, str(exc))

    # --- engine readiness ------------------------------------------------
    # Checked against the configuration this request will actually run, not the
    # service default: a request selecting a different engine must not be
    # refused because the default one is unavailable.
    if not pool.is_ready(config):
        # Named by flow, not by `ocr.engine`: under the two region-based flows
        # there is no single engine and `ocr.engine` is not even set, so
        # reporting it would say "engine 'None' is not ready".
        return error_response(
            503,
            ErrorCodes.ENGINE_NOT_READY,
            f"The '{config.flow.name}' pipeline is not ready "
            f"({config.ocr.engine or 'detector/recognizer components'}). It may "
            "still be loading, or its dependencies may not be installed.",
        )

    # --- run the pipeline ------------------------------------------------
    try:
        orchestrator = pool.get(config)
        run = orchestrator.run(
            ImagePayload(image=document.image),
            catalogs=_catalogs(parsed_options),
            write_debug_images=settings.write_step_debug_images,
            persist=settings.persist_results,
        )
    except (ConfigurationError, KeyError) as exc:
        # An unknown component name in the configuration -- a step, engine,
        # extractor or matcher the registry does not have.
        return error_response(
            422,
            ErrorCodes.INVALID_CONFIGURATION,
            "The configuration names a component this service does not have.",
            str(exc),
        )
    except StepExecutionError as exc:
        return error_response(
            500,
            ErrorCodes.OCR_FAILED,
            f"Preprocessing step '{exc.step_name}' failed on this document.",
            str(exc.original),
        )
    except Exception as exc:  # noqa: BLE001 - any engine failure maps to one code
        logger.exception("Extraction failed for '%s'.", filename)
        return error_response(
            500,
            ErrorCodes.OCR_FAILED,
            "The OCR engine failed while processing this document.",
            f"{type(exc).__name__}: {exc}",
        )

    body = run.output if isinstance(run.output, dict) else {}
    height, width = run.display_image.shape[:2]

    warnings = [ExtractionWarning(**w) for w in body.get("warnings", [])]

    # --- diagnostic images -----------------------------------------------
    # Deliberately outside the try above: a PNG that fails to encode is not an
    # OCR failure, and must not turn a good extraction into a 500.
    enhanced_png: str | None = None
    ocr_input_png: str | None = None

    if parsed_options.return_debug_images:
        max_width = settings.debug_image_max_width
        enhanced_png = encode_png_base64(run.enhanced_image, max_width=max_width)
        ocr_input_png = encode_png_base64(run.ocr_input_image, max_width=max_width)

        if enhanced_png is None or ocr_input_png is None:
            warnings.append(
                ExtractionWarning(
                    code=WarningCodes.DEBUG_IMAGE_UNAVAILABLE,
                    message=(
                        "A diagnostic image could not be encoded; the extraction "
                        "itself is unaffected."
                    ),
                )
            )

    result = ExtractionResult(
        request_id=str(uuid.uuid4()),
        processing_ms=int((time.perf_counter() - started) * 1000),
        source=ExtractionSource(
            filename=filename,
            page_count=document.page_count,
            page_used=document.page_used,
            width=width,
            height=height,
        ),
        header=body.get("header") or {},
        line_items=body.get("line_items", []),
        warnings=warnings,
        raw_text=body.get("raw_text", ""),
        invoice_id=body.get("invoice_id"),
        ocr_engine=body.get("ocr_engine"),
        elements=body.get("elements", []),
        enhanced_image_png=enhanced_png,
        ocr_input_image_png=ocr_input_png,
    )

    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


def _resolve_config(pool: PipelinePool, options: ExtractionOptions) -> AppConfig:
    """The configuration this request runs under.

    A request that sends none inherits the service's default file, which is
    what the CLI and the tests rely on. One that sends a `configuration` object
    gets exactly that, with the table branch's preprocessing filled in from the
    default -- the settings contract deliberately does not expose it.
    """
    if options.configuration is None:
        return pool.default_config

    # Resolved against the project root for the same reason the default file is:
    # the desktop app sends `keywords/ar_invoice_terms.json` back unchanged, and
    # the service is not started from this directory.
    return resolve_paths(
        options.configuration.to_app_config(
            table_photometric_steps=pool.default_table_photometric_steps
        )
    )


def _catalogs(options: ExtractionOptions) -> Catalogs:
    """Convert the request's catalogs into the parser's match targets."""

    def entries(items) -> list[NamedEntry]:
        return [
            NamedEntry(
                name=item.name,
                entry_id=item.entry_id,
                aliases=list(getattr(item, "aliases", [])),
            )
            for item in items
            if item.names
        ]

    return Catalogs(
        merchants=entries(options.known_merchants),
        products=entries(options.known_products),
        cities=entries(options.known_cities),
        top_k=options.max_candidates,
    )
