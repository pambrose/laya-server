"""Request validation: Jev's rules first, then Laya's own.

Laya 0.3.4 validated nothing, so this module pre-empted each raw KeyError,
AttributeError and numpy error it would raise. Laya 0.4 checks a
question itself (`Agent._check_question`, agent.py:967-1106) and raises a
ValueError naming it. The server still checks first, for two reasons: Jev's
documented limits (255 choice options, 2-10 score levels) reject what Laya
accepts, and a 422 should name the problem in Jev's terms before any
checkpoint is touched. Laya's check then runs here too, rather than inside
inference, so every malformed question is rejected before tokenizing.

Laya's private static helpers are called directly instead of mirrored. A
mirror of `_to_internal` drifted silently on the first upgrade; a call fails
the fast suite loudly if Laya ever renames one.
"""

from collections.abc import Mapping
from typing import Any

from laya.agent import Agent
from laya.common import (
    build_sequence,
    encode_text,
    render_options,
    serialize_state,
    state_room,
)

from .errors import ValidationFailed

QUESTION_TYPES = ("choice", "score", "noul")

# The question fields Jev defines. Laya has grown its own (`labels` for noul,
# `option_order`), which change what the model is shown; a Jev client cannot
# send them, and passing them through would expose behavior Jev lacks. Like
# any other unknown field, they are ignored.
JEV_QUESTION_FIELDS = ("type", "instructions", "criteria")

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
    # Optional, but when present it must be keyed "true"/"false" (agent.py:1070-1087).
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

    # Laya's own rules (blank instructions, null or duplicate labels, noul keys
    # other than true/false). Run on exactly what Laya will be handed. Before
    # this, an unhashable list label reached the budget check's dict
    # comprehension and escaped as a TypeError 500. TypeError is caught too, as
    # in the engine's backstop: this runs before it, so nothing else would.
    try:
        Agent._check_question(qid, jev_question(question))
    except (ValueError, TypeError) as exc:
        raise ValidationFailed(str(exc)) from exc


def jev_question(question: Mapping[str, Any]) -> dict[str, Any]:
    """The question as Jev defines it, without fields only Laya understands."""
    return {k: question[k] for k in JEV_QUESTION_FIELDS if k in question}


def jev_questions(questions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """`jev_question` over a validated request: what routing and Laya receive."""
    return {qid: jev_question(q) for qid, q in questions.items()}


def validate_questions(questions: Any) -> None:
    """Raise ValidationFailed if Jev would reject or Laya would refuse."""
    if not isinstance(questions, Mapping):
        raise ValidationFailed("'questions' must be an object of question definitions")
    if not questions:
        # Laya now answers this with an empty result (agent.py:1486-1491), but
        # Jev requires at least one question.
        raise ValidationFailed("'questions' must contain at least one question")
    for qid, question in questions.items():
        validate_question(qid, question)


def _to_internal(qdef: Mapping[str, Any]) -> dict[str, Any]:
    """Laya's own conversion (agent.py:1108-1130): a pure staticmethod, so no
    loaded model is touched. Mirroring it instead drifted on the first upgrade
    (noul key case, non-ASCII instructions)."""
    internal: dict[str, Any] = Agent._to_internal(dict(qdef))
    return internal


def _budgets(agent: Any) -> tuple[int, int]:
    return int(agent.cfg.get("max_len", 512)), int(agent.cfg.get("head_max_len", 192))


def _room_for_state(agent: Any, qdef: Mapping[str, Any]) -> int:
    """Tokens of state this question leaves room for on this checkpoint.

    Laya's public `state_room` (common.py:350-359) is documented as exactly
    what `build_sequence` keeps, so this cannot drift from the real cut.
    """
    max_len, head_max_len = _budgets(agent)
    return int(state_room(agent.tok, _to_internal(qdef), max_len, head_max_len))


def _validate_options_fit(agent: Any, qid: str, qdef: Mapping[str, Any]) -> None:
    """Reject options Laya cannot place inside the sequence.

    Laya raises a ValueError when the option markers do not all survive
    `max_len` (agent.py:1161-1177). Jev's documented 255-option ceiling is far
    looser, so a legal-looking request would otherwise fail only after it had
    queued for inference. The check mirrors Laya's own by building the
    sequence and comparing marker count; `state_room` cannot stand in here
    because only `build_sequence` drops the markers past `max_len`.
    """
    internal = _to_internal(qdef)
    max_len, head_max_len = _budgets(agent)
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
    """Reject state Laya would truncate.

    Laya cuts an oversized state to fit (common.py:199-204) and answers anyway.
    It now records the cut in `usage["truncated"]`, but Jev's usage has no
    such field, so a caller migrating from Jev would get an answer computed on
    partial input with no sign of it. Jev allows 32k tokens of state; these
    checkpoints allow 512-1024, so the difference is reported explicitly.
    """
    # Counted exactly as Laya encodes the state (agent.py:1149-1153), and
    # through its `encode_text`: Laya now enables truncation on the shared
    # tokenizer while encoding options, so an unlocked call here could race an
    # in-flight request's encode and fail with "Already borrowed".
    text = serialize_state(state).replace(agent.tok.mask_token, " ")
    state_tokens = len(
        encode_text(agent.tok, text, add_special_tokens=False)["input_ids"]
    )

    for qid, qdef in questions.items():
        _validate_options_fit(agent, qid, qdef)

        room = _room_for_state(agent, qdef)
        if state_tokens > room:
            raise ValidationFailed(
                f"state exceeds model budget: {state_tokens} tokens, but question "
                f"{qid!r} leaves room for {room} on checkpoint {checkpoint!r}"
            )
