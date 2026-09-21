"""Normalize Laya's inference output into Jev's documented wire shape.

Laya returns everything Jev does plus an `action` block on every answer, and a
`confidence` on `noul` that Jev explicitly does not have.
"""

from typing import Any

# Keys Jev returns for each primitive, in the order its docs list them.
_JEV_ANSWER_KEYS = {
    "choice": ("type", "choice", "probabilities", "confidence"),
    "score": ("type", "score", "legend", "probabilities", "confidence"),
    "noul": ("type", "noul"),
}


def to_jev_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """Keep only the keys Jev documents for this answer's primitive."""
    keys = _JEV_ANSWER_KEYS[answer["type"]]
    return {k: answer[k] for k in keys if k in answer}


def to_jev_response(result: dict[str, Any], requested_model: str) -> dict[str, Any]:
    """Shape a `Router.predict` result into a Jev response body.

    `model` echoes what the client asked for; the checkpoint that actually served
    the request is reported in the X-Laya-Checkpoint header instead, and Laya's
    `routing` key is dropped so the body carries no fields Jev would not send.
    """
    return {
        "model": requested_model,
        "answers": {
            qid: to_jev_answer(answer) for qid, answer in result["answers"].items()
        },
        "usage": result["usage"],
    }
