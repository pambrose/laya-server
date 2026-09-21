"""Environment-driven settings.

Every deployment knob lives here so the local-first run and a future container
differ only by environment, never by code.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

ALL_CHECKPOINTS = ("english", "multilingual", "typed-decisions")

DEFAULT_MAX_QUEUE_DEPTH = 32


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    device: str | None = None
    preload: tuple[str, ...] = ALL_CHECKPOINTS
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH
    log_level: str = "INFO"

    @property
    def auth_enabled(self) -> bool:
        """Auth is opt-in: no key configured means an open local server."""
        return self.api_key is not None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env

        preload = env.get("LAYA_SERVER_PRELOAD")
        checkpoints = (
            tuple(name.strip() for name in preload.split(",") if name.strip())
            if preload
            else ALL_CHECKPOINTS
        )

        return cls(
            api_key=env.get("LAYA_SERVER_API_KEY"),
            device=env.get("LAYA_SERVER_DEVICE"),
            preload=checkpoints,
            max_queue_depth=int(
                env.get("LAYA_SERVER_MAX_QUEUE", DEFAULT_MAX_QUEUE_DEPTH)
            ),
            log_level=env.get("LAYA_SERVER_LOG_LEVEL", "INFO"),
        )
