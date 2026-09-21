"""FastAPI application exposing Laya behind Jev's HTTP contract.

The single route mirrors TypeSafe's evaluation endpoint, so a client can be
pointed here by setting TYPESAFE_BASE_URL and needs no other change.
"""

import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import Settings
from .engine import Engine
from .errors import JevError, Unauthorized, ValidationFailed
from .models import list_models
from .schemas import SystemOneRequest

logger = logging.getLogger("laya_server")

SYSTEMONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
HEALTH_PATH = "/health"
READY_PATH = "/ready"


def _describe(exc: ValidationError) -> str:
    """Render pydantic's errors as one readable sentence."""
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'body'}: {e['msg']}"
        for e in exc.errors()
    )


async def _enforce_body_limit(request: Request, limit: int) -> None:
    """Reject an oversized body before it is parsed or tokenized.

    Validation tokenizes the state on the event loop before any lock is taken,
    so a large body stalls every other request -- including the health probes,
    which exist precisely so they never wait. Content-Length is checked first
    to avoid reading the body at all; the read size is then re-checked because
    the header is client-supplied and may be absent or wrong.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise ValidationFailed(
            f"request body too large: {declared} bytes, limit is {limit}"
        )

    body = await request.body()
    if len(body) > limit:
        raise ValidationFailed(
            f"request body too large: {len(body)} bytes, limit is {limit}"
        )


def _check_auth(settings: Settings, authorization: str | None) -> None:
    """Jev answers a missing or wrong key with 401; mirror that when a key is set."""
    if not settings.auth_enabled:
        return
    expected = f"Bearer {settings.api_key}"
    # compare_digest, not ==: a short-circuiting comparison leaks the key
    # through response timing.
    if not secrets.compare_digest(authorization or "", expected):
        raise Unauthorized("missing or invalid API key")


def _apply_log_level(level: str) -> None:
    """Apply LAYA_SERVER_LOG_LEVEL, which previously had no effect anywhere.

    An unrecognised name falls back to INFO rather than crashing startup: a
    typo in a log level should not take the server down.
    """
    resolved = logging.getLevelNamesMapping().get(level.strip().upper(), logging.INFO)
    logger.setLevel(resolved)


def create_app(
    settings: Settings | None = None, engine: Engine | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    _apply_log_level(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Preloading here means `Router.load` is a dict hit during requests and
        # its unsynchronized build path never runs concurrently.
        if app.state.engine is None:
            app.state.engine = Engine.create(settings)
        yield

    app = FastAPI(title="laya-server", lifespan=lifespan)
    app.state.engine = engine
    app.state.settings = settings

    @app.exception_handler(JevError)
    async def handle_jev_error(request: Request, exc: JevError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body())

    # Probes deliberately take no lock and touch no checkpoint, so a server
    # busy on a forward pass still answers them. They are also exempt from auth:
    # orchestrators probe without credentials. Neither is part of Jev's API, so
    # a drop-in client never sees them.
    @app.get(HEALTH_PATH)
    async def health() -> JSONResponse:
        """Liveness: the process is up and serving. Nothing more is claimed."""
        return JSONResponse(content={"status": "ok"})

    @app.get(READY_PATH)
    async def ready() -> JSONResponse:
        """Readiness: a checkpoint is resident, so a request can be served."""
        engine = app.state.engine
        try:
            status = engine.status() if engine is not None else None
        except Exception:
            # A probe reports unhealthy; it never crashes. Without this a
            # status() failure surfaces as a 500 traceback instead of 503.
            logger.exception("readiness probe failed")
            status = None

        ready = status is not None and bool(status["checkpoints"]["loaded"])

        # The route is auth-exempt so orchestrators can probe without
        # credentials. When a key is configured, withhold the detail rather
        # than publishing the device and checkpoint inventory to anyone --
        # /v1/models sits behind the same key.
        detail: dict[str, Any] = {}
        if not settings.auth_enabled and status is not None:
            detail = status

        if not ready:
            return JSONResponse(
                status_code=503, content={"status": "not ready", **detail}
            )
        return JSONResponse(content={"status": "ready", **detail})

    @app.get(MODELS_PATH)
    async def models(
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        """Every model name this server accepts. Jev sends Authorization on
        this route, so it is behind the same key as inference."""
        _check_auth(settings, authorization)
        return JSONResponse(content=list_models())

    @app.post(SYSTEMONE_PATH)
    async def system_one(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        _check_auth(settings, authorization)

        await _enforce_body_limit(request, settings.max_body_bytes)

        # Validated here rather than as a body parameter so that an
        # unauthenticated caller gets 401 without learning whether its body
        # was well-formed.
        try:
            payload = SystemOneRequest.model_validate(await request.json())
        except ValidationError as exc:
            raise ValidationFailed(_describe(exc)) from exc

        started = time.perf_counter()
        result = await request.app.state.engine.predict(
            payload.state, payload.questions, requested_model=payload.model
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "systemone checkpoint=%s reason=%s questions=%d latency_ms=%.1f",
            result.checkpoint,
            result.reason,
            len(payload.questions),
            elapsed_ms,
        )

        return JSONResponse(
            content=result.body,
            headers={"X-Laya-Checkpoint": result.checkpoint},
        )

    return app


# Module-level app for `uvicorn laya_server.app:app`. Constructing it does not
# load any checkpoint; that happens in the lifespan hook at startup.
app = create_app()
