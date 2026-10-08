"""Environment-driven settings.

Every deployment knob lives here so the local-first run and a future container
differ only by environment, never by code.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

ALL_CHECKPOINTS = ("english", "multilingual", "typed-decisions")

# Where language-agnostic routing sends a state whose language it cannot
# identify. laya 0.4.0 moved its own default to "multilingual"; this keeps the
# English fallback the server has always had, so upgrading laya changes no
# routing on its own. See the README for when to flip it.
DEFAULT_CHECKPOINT = "english"

DEFAULT_MAX_QUEUE_DEPTH = 32

# Generous next to the 512-1024 token context, but finite: an unbounded body
# is read into memory and then tokenized before any lock is taken.
DEFAULT_MAX_BODY_BYTES = 1_000_000


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    """Read an integer setting, failing with a message that names the variable.

    A bare int() raises `invalid literal for int()` mentioning neither the
    variable nor which value was wrong, which is a poor first experience of a
    misconfigured container.
    """
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    device: str | None = None
    preload: tuple[str, ...] = ALL_CHECKPOINTS
    default_checkpoint: str = DEFAULT_CHECKPOINT
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    log_level: str = "INFO"

    @property
    def auth_enabled(self) -> bool:
        """Auth is opt-in: no key configured means an open local server."""
        return self.api_key is not None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env

        preload = env.get("LAYA_SERVER_PRELOAD") or ""
        # A list with no names (unset, empty, or just commas) means all of
        # them. laya 0.4 reads `preload([])` as "load nothing" where 0.3.4
        # loaded everything, which would leave /ready failing until a request
        # arrived that the orchestrator would never route to it.
        checkpoints = (
            tuple(name.strip() for name in preload.split(",") if name.strip())
            or ALL_CHECKPOINTS
        )

        return cls(
            api_key=env.get("LAYA_SERVER_API_KEY"),
            device=env.get("LAYA_SERVER_DEVICE"),
            preload=checkpoints,
            default_checkpoint=env.get(
                "LAYA_SERVER_DEFAULT_CHECKPOINT", DEFAULT_CHECKPOINT
            ).strip(),
            max_queue_depth=_int_env(
                env, "LAYA_SERVER_MAX_QUEUE", DEFAULT_MAX_QUEUE_DEPTH
            ),
            max_body_bytes=_int_env(
                env, "LAYA_SERVER_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES
            ),
            log_level=env.get("LAYA_SERVER_LOG_LEVEL", "INFO"),
        )
