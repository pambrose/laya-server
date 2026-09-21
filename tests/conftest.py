"""Test doubles light enough to avoid loading a 421M-parameter checkpoint."""

import pytest


class FakeTokenizer:
    """Whitespace tokenizer with the HF surface `laya.common.build_sequence` uses."""

    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    pad_token_id = 0
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [4] * len(str(text).split())}


class FakeAgent:
    """Stands in for `laya.Agent` for budget checks; never runs inference."""

    def __init__(self, max_len=512, head_max_len=192, device="cpu"):
        self.tok = FakeTokenizer()
        self.cfg = {"max_len": max_len, "head_max_len": head_max_len}
        self.device = device


@pytest.fixture
def agent():
    return FakeAgent()


LAYA_NOUL_ANSWER = {
    "type": "noul",
    "noul": 0.91,
    "confidence": 0.91,
    "action": {"act_probability": 0.2},
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
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }


class FakeRouter:
    """Stands in for `laya.Router`, recording how it was asked to route."""

    def __init__(self, checkpoint="english", agents=None):
        self.checkpoint = checkpoint
        self.agents = agents or {}
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

    def load(self, name):
        return self.agents.setdefault(name, FakeInferenceAgent())

    def preload(self, names=None):
        self.preloaded = list(names or [])
