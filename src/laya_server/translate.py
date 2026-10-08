"""Normalize Laya's inference output into Jev's documented wire shape.

Laya returns everything Jev does plus an `action` block and an
`answer_confidence` on every answer, a `confidence` on `noul` that Jev
explicitly does not have, and truncation and option diagnostics in `usage`.
"""

from collections.abc import Mapping
from typing import Any

# Keys Jev returns for each primitive, in the order its docs list them.
_JEV_ANSWER_KEYS = {
    "choice": ("type", "choice", "probabilities", "confidence"),
    "score": ("type", "score", "legend", "probabilities", "confidence"),
    "noul": ("type", "noul"),
}

# Jev's `Usage` has exactly these. Laya 0.4 adds `state_tokens`, `truncated`,
# `options` and more (agent.py:1541-1566).
_JEV_USAGE_KEYS = ("input_tokens", "output_tokens")


def _jev_legend(laya_legend: dict[str, Any], criteria: Any) -> dict[str, Any]:
    """Each score level as the caller wrote it, where Jev's type allows it.

    Laya renders every level to text (agent.py:1384-1390), so a level written
    as an object comes back as a JSON string. Jev's legend admits objects and
    arrays (typesafe_sdk response_types.py:55), so those are echoed verbatim.
    Anything else keeps Laya's text, which is also what lets a numeric level
    parse: the SDK is strict, and 0.3.4's raw int echo was not a valid value.
    """
    if not isinstance(criteria, (list, tuple)):
        return laya_legend
    return {
        key: criteria[int(key)]
        if isinstance(criteria[int(key)], (dict, list))
        else text
        for key, text in laya_legend.items()
    }


def to_jev_answer(
    answer: dict[str, Any], question: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Keep only the keys Jev documents for this answer's primitive."""
    keys = _JEV_ANSWER_KEYS[answer["type"]]
    jev = {k: answer[k] for k in keys if k in answer}
    if "legend" in jev and question is not None:
        jev["legend"] = _jev_legend(jev["legend"], question.get("criteria"))
    return jev


def to_jev_response(
    result: dict[str, Any],
    requested_model: str,
    questions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Shape a `Router.predict` result into a Jev response body.

    `model` echoes what the client asked for; the checkpoint that actually served
    the request is reported in the X-Laya-Checkpoint header instead, and Laya's
    `routing` key is dropped so the body carries no fields Jev would not send.
    """
    questions = questions or {}
    return {
        "model": requested_model,
        "answers": {
            qid: to_jev_answer(answer, questions.get(qid))
            for qid, answer in result["answers"].items()
        },
        "usage": {
            k: result["usage"][k] for k in _JEV_USAGE_KEYS if k in result["usage"]
        },
    }
