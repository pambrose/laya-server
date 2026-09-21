"""Pydantic models for Jev's documented request contract.

Only the request is modelled. Responses are built by `translate` from Laya's
output so that the body can never gain a field Jev would not send.
"""

from typing import Any

from pydantic import BaseModel, Field

State = str | dict[str, Any] | list[Any]


class SystemOneRequest(BaseModel):
    state: State = Field(..., description="Content to evaluate")
    model: str = Field(..., description="e.g. jev-latest")
    questions: dict[str, Any] = Field(..., description="question_id -> definition")
