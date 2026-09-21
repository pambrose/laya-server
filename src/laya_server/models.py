"""The catalog served by `GET /v1/models`.

Jev documents `name` as "the model ID or alias, as accepted by the `model`
field", so this lists every value `resolve_model` accepts: the Jev aliases,
which let Laya route, and Laya's own checkpoint names, which force one.

`release_date` is the PyPI release of the `laya` distribution that provides
these checkpoints, not a Jev date -- inventing one would misdescribe what
actually answers the request. Update it when the laya pin moves.
"""

from importlib.metadata import version
from typing import Any

from .engine import JEV_MODELS, LAYA_MODELS

# PyPI upload dates for the laya releases whose checkpoints this server has
# served. `pyproject.toml` pins `laya>=0.3.4`, so an upgrade can install a
# version that is not listed here; a test asserts the installed version is
# present, which turns a silently stale `release_date` into a failing build.
LAYA_RELEASE_DATES: dict[str, str] = {
    "0.3.4": "2026-09-20",
}


def _release_date() -> str:
    """The release date of the installed laya, or the newest one recorded."""
    installed = version("laya")
    if installed in LAYA_RELEASE_DATES:
        return LAYA_RELEASE_DATES[installed]
    return LAYA_RELEASE_DATES[max(LAYA_RELEASE_DATES)]


_ALIAS_DESCRIPTION = (
    "Alias: routes by script and language detection to a Laya checkpoint."
)

_LAYA_DESCRIPTIONS: dict[str, str] = {
    "english": "Laya English checkpoint (ModernBERT-large, 512-token context).",
    "multilingual": (
        "Laya multilingual checkpoint (mmBERT-base, 1024-token context, "
        "100+ languages)."
    ),
    "typed-decisions": (
        "Laya checkpoint fine-tuned for typed-decision workflows "
        "(ModernBERT-large, 1024-token context)."
    ),
}


# Keyed off the engine's constants so a model added there cannot go missing
# here, which would make /v1/models under-report a name the API accepts.
_DESCRIPTIONS: dict[str, str] = {
    **{name: _ALIAS_DESCRIPTION for name in JEV_MODELS},
    **{name: _LAYA_DESCRIPTIONS[name] for name in LAYA_MODELS},
}


def list_models() -> dict[str, list[dict[str, Any]]]:
    """The `ListModelsResponse` body: one entry per accepted model name."""
    return {
        "models": [
            {
                "name": name,
                "description": description,
                "release_date": _release_date(),
            }
            for name, description in _DESCRIPTIONS.items()
        ]
    }
