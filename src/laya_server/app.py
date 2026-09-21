"""FastAPI application exposing Laya behind Jev's HTTP contract.

The single route mirrors TypeSafe's evaluation endpoint, so a client can be
pointed here by setting TYPESAFE_BASE_URL and needs no other change.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import Settings
from .engine import Engine
from .errors import JevError, Unauthorized, ValidationFailed
from .schemas import SystemOneRequest

logger = logging.getLogger("laya_server")

SYSTEMONE_PATH = "/v1/systemone"


def _describe(exc: ValidationError) -> str:
    """Render pydantic's errors as one readable sentence."""
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'body'}: {e['msg']}"
        for e in exc.errors()
    )


def _check_auth(settings: Settings, authorization: str | None) -> None:
    """Jev answers a missing or wrong key with 401; mirror that when a key is set."""
    if not settings.auth_enabled:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise Unauthorized("missing or invalid API key")


def create_app(settings: Settings | None = None, engine: Engine | None = None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Preloading here means `Router.load` is a dict hit during requests and
        # its unsynchronized build path never runs concurrently.
        if app.state.engine is None:
            app.state.engine = Engine.create(settings)
        yield

    app = FastAPI(title="laya-server", lifespan=lifespan)
    app.state.engine = engine
    app.state.settings = settings

    @app.exception_handler(JevError)
    async def handle_jev_error(request: Request, exc: JevError):
        return JSONResponse(status_code=exc.status_code, content=exc.body())

    @app.exception_handler(RequestValidationError)
    async def handle_schema_error(request: Request, exc: RequestValidationError):
        # FastAPI's default 422 body differs from Jev's; normalize it.
        return JSONResponse(
            status_code=422,
            content=ValidationFailed(str(exc.errors())).body(),
        )

    @app.post(SYSTEMONE_PATH)
    async def system_one(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        _check_auth(settings, authorization)

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
