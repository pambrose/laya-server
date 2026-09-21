"""Request validation, because Laya does none of its own.

Laya 0.3.4 has no schema checks and no exception types: a missing key is a raw
KeyError, a null `criteria` an AttributeError, and empty criteria reaches numpy
as a zero-size reduction. Every one of those is a client-triggerable 500, so
each is pre-empted here and reported as a 422 instead.

Limits (255 choice options, 2-10 score levels) are Jev's documented ones, kept
here so this server rejects what the real API would reject.
"""

import json
from collections.abc import Mapping
from typing import Any

from laya.common import build_sequence, render_options, serialize_state

from .errors import ValidationFailed

QUESTION_TYPES = ("choice", "score", "noul")

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


def _reject(qid: str, problem: str) -> None:
    raise ValidationFailed(f"question {qid!r}: {problem}")


def _validate_choice(qid: str, criteria: Any) -> None:
    if not isinstance(criteria, (Mapping, list, tuple)):
        _reject(qid, "'criteria' must be an object or array of options")
    if len(criteria) == 0:
        _reject(qid, "'criteria' must define at least one option")
    if len(criteria) > MAX_CHOICE_OPTIONS:
        _reject(
            qid,
            f"'criteria' defines {len(criteria)} options, "
            f"more than the maximum of {MAX_CHOICE_OPTIONS}",
        )


def _validate_score(qid: str, criteria: Any) -> None:
    if not isinstance(criteria, (list, tuple)):
        _reject(qid, "'criteria' must be an array of ordered levels")
    if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
        _reject(
            qid,
            f"'criteria' defines {len(criteria)} levels; "
            f"a score needs between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS}",
        )


def _validate_noul(qid: str, criteria: Any) -> None:
    # Optional, but when present Laya reads criteria.get("true"/"false").
    if criteria is not None and not isinstance(criteria, Mapping):
        _reject(qid, "'criteria' must be an object with 'true' and/or 'false' keys")


def validate_question(qid: str, question: Any) -> None:
    if not isinstance(question, Mapping):
        _reject(qid, "must be an object")

    qtype = question.get("type")
    if qtype is None:
        _reject(qid, "missing required field 'type'")
    if qtype not in QUESTION_TYPES:
        _reject(
            qid,
            f"unknown type {qtype!r}; expected one of {', '.join(QUESTION_TYPES)}",
        )

    instructions = question.get("instructions")
    if instructions is None:
        _reject(qid, "missing required field 'instructions'")

    criteria = question.get("criteria")
    if qtype == "choice":
        _validate_choice(qid, criteria)
    elif qtype == "score":
        _validate_score(qid, criteria)
    else:
        _validate_noul(qid, criteria)


def validate_questions(questions: Any) -> None:
    """Raise ValidationFailed if Laya would crash or Jev would reject."""
    if not isinstance(questions, Mapping):
        raise ValidationFailed("'questions' must be an object of question definitions")
    if not questions:
        raise ValidationFailed("'questions' must contain at least one question")
    for qid, question in questions.items():
        validate_question(qid, question)


def _to_internal(qdef: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror of `laya.Agent._to_internal` (agent.py:229-238).

    Reimplemented rather than called so budget checks never touch a private
    method on a loaded model; kept deliberately identical.
    """
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, (list, tuple)):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def _room_for_state(agent: Any, qdef: Mapping[str, Any]) -> int:
    """Tokens of state this question leaves room for on this checkpoint.

    Derived by building the real sequence with an empty state rather than
    recomputing Laya's header arithmetic, so it cannot drift from
    `build_sequence` (common.py:49-85).
    """
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    empty, _ = build_sequence(agent.tok, "", _to_internal(qdef), max_len, head_max_len)
    return max(0, int(max_len) - len(empty))


def _validate_options_fit(agent: Any, qid: str, qdef: Mapping[str, Any]) -> None:
    """Reject options Laya cannot place inside head_max_len.

    Laya raises a bare ValueError when the option markers do not all survive
    the head budget (agent.py:262-263). Jev's documented 255-option ceiling is
    far looser, so without this a legal-looking request reaches Laya and comes
    back as a 500. The check mirrors Laya's own by building the sequence and
    comparing marker count, so it cannot drift from build_sequence.
    """
    internal = _to_internal(qdef)
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    _, markers = build_sequence(agent.tok, "", internal, max_len, head_max_len)
    wanted = len(render_options(internal))
    if len(markers) != wanted:
        _reject(
            qid,
            f"{wanted} options do not fit the model's {head_max_len}-token "
            f"question budget; only {len(markers)} could be encoded. "
            "Use fewer or shorter options",
        )


def validate_budget(
    agent: Any,
    state: str | dict[str, Any] | list[Any],
    questions: Mapping[str, Any],
    checkpoint: str,
) -> None:
    """Reject state Laya would silently truncate.

    Laya trims the tail of an oversized state without any signal
    (common.py:82-83), which would hand a caller migrating from Jev an answer
    computed on partial input. Jev allows 32k tokens of state; these
    checkpoints allow 512-1024, so the difference is reported explicitly.
    """
    text = serialize_state(state)
    state_tokens = len(agent.tok(text, add_special_tokens=False)["input_ids"])

    for qid, qdef in questions.items():
        _validate_options_fit(agent, qid, qdef)

        room = _room_for_state(agent, qdef)
        if state_tokens > room:
            raise ValidationFailed(
                f"state exceeds model budget: {state_tokens} tokens, but question "
                f"{qid!r} leaves room for {room} on checkpoint {checkpoint!r}"
            )
