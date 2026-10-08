"""Test doubles light enough to avoid loading a 421M-parameter checkpoint."""

import pytest


class FakeTokenizer:
    """Whitespace tokenizer with the HF surface `laya.common.build_sequence` uses."""

    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    pad_token_id = 0
    mask_token = "[MASK]"

    def __call__(
        self, text, add_special_tokens=False, truncation=False, max_length=None
    ):
        # laya 0.4 encodes each option with truncation=True, max_length=48
        # (common.py:244-250); honoring it keeps option-heavy budgets faithful.
        ids = [4] * len(str(text).split())
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class FakeAgent:
    """Stands in for `laya.Agent` for budget checks; never runs inference."""

    def __init__(self, max_len=512, head_max_len=192, device="cpu"):
        self.tok = FakeTokenizer()
        self.cfg = {"max_len": max_len, "head_max_len": head_max_len}
        self.device = device


@pytest.fixture
def agent():
    return FakeAgent()


# Shaped like laya 0.4.1's output (agent.py:1396-1405, 1541-1572), including
# the keys Jev does not have, so a test sees them leak if translation regresses.
LAYA_NOUL_ANSWER = {
    "type": "noul",
    "noul": 0.91,
    "confidence": 0.91,
    "answer_confidence": 0.91,
    "action": {"act_probability": 0.2},
}

LAYA_USAGE = {
    "input_tokens": 42,
    "output_tokens": 0,
    "state_tokens": 9,
    "state_tokens_dropped": 0,
    "truncated": False,
    "truncated_questions": [],
}


class FakeInferenceAgent(FakeAgent):
    """FakeAgent that also answers, recording concurrency as it goes."""

    def __init__(self, on_call=None, **kw):
        super().__init__(**kw)
        self.calls = []
        self.on_call = on_call

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if self.on_call is not None:
            self.on_call()
        return {
            "model": "laya-rl-agent",
            "answers": {qid: dict(LAYA_NOUL_ANSWER) for qid in questions},
            "usage": dict(LAYA_USAGE),
        }


class FakeRouter:
    """Stands in for `laya.Router`, recording how it was asked to route."""

    def __init__(self, checkpoint="english", agents=None):
        self.checkpoint = checkpoint
        # Named as laya names it, so code reading the store works against both.
        self._agents = agents or {}
        self.route_calls = []
        self.preloaded = []

    def route(self, state, questions, model=None, task=None, lang=None):
        self.route_calls.append({"model": model, "state": state})
        name = model or self.checkpoint
        return {
            "model": name,
            "repo": "convaiinnovations/laya",
            "reason": "explicit" if model else "detection",
            "detection": None,
            "workflow": None,
        }

    @property
    def agents(self):
        return self._agents

    def load(self, name):
        return self._agents.setdefault(name, FakeInferenceAgent())

    def preload(self, names=None):
        self.preloaded = list(names or [])


class LoadedFakeRouter(FakeRouter):
    """FakeRouter that reports resident checkpoints, as `laya.Router` does.

    `loaded` mirrors laya's `_order` list and `load()` reorders it through the
    same `_touch` semantics (router.py:169-185). The earlier version returned a
    fixed list, which made it more forgiving than the real Router and hid a bug
    where a read-only probe reordered what the next probe reported.
    """

    def __init__(self, loaded=("english",), **kw):
        super().__init__(**kw)
        self._order = list(loaded)

    @property
    def loaded(self):
        return list(self._order)

    def load(self, name):
        if name in self._order:
            self._order.remove(name)
            self._order.append(name)
        return super().load(name)
