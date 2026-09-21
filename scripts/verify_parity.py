"""End-to-end parity: ONNX + a torch-free postprocess vs laya's own Agent.system_one.

The ONNX check in export_onnx.py only proves the logits match. What a non-Python
port actually has to reproduce is the *answer payload* -- the rounded
choice/score/noul values, probability maps and confidences the server returns.
This script reimplements collate + postprocess in plain numpy (no torch), runs
the exported graph, and diffs the result against laya's own output.

The numpy code below is deliberately written as a port reference: it is what a
Kotlin/JVM,
Rust or Go implementation has to mirror line for line.

    uv run python scripts/verify_parity.py --only english
"""

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
from laya.agent import Agent

# build_sequence is the tokenizer-coupled half of the port; reuse it here so this script
# isolates the ONNX + postprocess question. Porting it is tracked separately.
from laya.common import build_sequence

REPO = "convaiinnovations/laya"
QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


# ------------------------------------------------- port reference (no torch)


def to_internal(qdef: dict) -> dict:
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def collate_numpy(items: list[dict], pad_id: int) -> dict[str, np.ndarray]:
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = np.full((n, L), pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = it["ids"]
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = it["markers"]
        mmask[i, :k] = True
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
    }


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPE_NAMES[int(qtype)]}:{size}"


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def postprocess(
    cfg: dict,
    qids: list[str],
    questions: dict,
    items: list[dict],
    logits: np.ndarray,
    act_logits: np.ndarray,
    n_tokens: int,
) -> dict:
    temperature = cfg.get("temperature", [1.0, 1.0, 1.0])
    by_opts = cfg.get("temperature_by_options", {})
    e = np.exp(act_logits - act_logits.max(-1, keepdims=True))
    act = e / e.sum(-1, keepdims=True)

    answers = {}
    for r, qid in enumerate(qids):
        q = to_internal(questions[qid])
        k = len(items[r]["markers"])
        qt = QTYPES[q["t"]]
        t_scale = by_opts.get(temp_bucket(qt, k), temperature[qt])
        z = logits[r, :k] / max(1e-3, float(t_scale))
        p = np.exp(z - z.max())
        p = p / p.sum()

        conf = round(confidence_from_probs(p, k), 4)
        ext = {"act_probability": round(float(act[r, 0]), 4)}

        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            answers[qid] = {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {
                    kk: round(float(v), 4) for kk, v in zip(keys, p, strict=True)
                },
                "confidence": conf,
                "action": ext,
            }
        elif q["t"] == "score":
            answers[qid] = {
                "type": "score",
                "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": conf,
                "action": ext,
            }
        else:
            answers[qid] = {
                "type": "noul",
                "noul": round(float(p[1]), 4),
                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                "action": ext,
            }
    return {
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": {"input_tokens": n_tokens, "output_tokens": 0},
    }


def onnx_system_one(sess, agent: Agent, state, questions: dict) -> dict:
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    qids = list(questions)
    items = []
    for qid in qids:
        q = to_internal(questions[qid])
        seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    b = collate_numpy(items, agent.tok.pad_token_id)
    logits, act = sess.run(["logits", "act_logits"], {k: v for k, v in b.items()})
    return postprocess(
        agent.cfg, qids, questions, items, logits, act, int(b["attention_mask"].sum())
    )


# ------------------------------------------------------------- test corpus

TRIAGE = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want in `message`?",
        "criteria": {
            "refund": "money returned",
            "technical_help": "a bug or outage",
            "billing_question": "an invoice question",
            "other": "none fits",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["calm", "concerned", "annoyed", "very angry"],
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "Does `message` communicate time pressure?",
    },
}
WIDE = {
    "department": {
        "type": "choice",
        "instructions": "Which department owns this?",
        "criteria": {f"dept_{i}": f"team number {i}" for i in range(11)},
    },
    "spam": {"type": "noul", "instructions": "Is this message spam?"},
}
CASES: list[tuple[str, Any, dict]] = [
    ("short_en", {"message": "I was charged twice, fix it today please."}, TRIAGE),
    (
        "plain_str",
        "The build has been failing since this morning and nobody replied.",
        TRIAGE,
    ),
    ("wide_choice", {"message": "My invoice is wrong and support ignored me."}, WIDE),
    (
        "long_en",
        {
            "message": "The checkout page returns a 500 error. " * 60,
            "plan": "enterprise",
            "tenure_months": 41,
        },
        TRIAGE,
    ),
    (
        "unicode",
        {"message": "La facturación está mal y necesito ayuda urgente. 请尽快处理。"},
        TRIAGE,
    ),
    ("empty_ish", {"message": ""}, TRIAGE),
]


def compare(ref: dict, got: dict) -> list[str]:
    """Exact comparison of the rounded payload, with `action` reported separately."""
    diffs = []
    if ref["usage"] != got["usage"]:
        diffs.append(f"usage: {ref['usage']} != {got['usage']}")
    for qid in ref["answers"]:
        a, b = dict(ref["answers"][qid]), dict(got["answers"][qid])
        ax, bx = a.pop("action", None), b.pop("action", None)
        if a != b:
            for key in a:
                if a[key] != b.get(key):
                    diffs.append(f"{qid}.{key}: {a[key]!r} != {b.get(key)!r}")
        if ax != bx:
            diffs.append(f"[action-only] {qid}.action: {ax} != {bx}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="check just this checkpoint")
    ap.add_argument("--out", default="build/onnx")
    a = ap.parse_args()

    targets = [
        ("english", None),
        ("typed-decisions", "typed-decisions"),
        ("multilingual", "multilingual"),
    ]
    if a.only:
        targets = [t for t in targets if t[0] == a.only]

    overall, report = True, []
    for name, sub in targets:
        onnx_path = os.path.join(a.out, name, "model.onnx")
        if not os.path.exists(onnx_path):
            print(f"{name}: no export at {onnx_path} -- run export_onnx.py first")
            overall = False
            continue
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        agent = Agent(REPO, device="cpu", subfolder=sub)
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        for label, state, questions in CASES:
            ref = agent.system_one(state, questions)
            got = onnx_system_one(sess, agent, state, questions)
            diffs = compare(ref, got)
            hard = [d for d in diffs if not d.startswith("[action-only]")]
            soft = [d for d in diffs if d.startswith("[action-only]")]
            status = "PASS" if not hard else "FAIL"
            note = f"  ({len(soft)} action-only diff)" if soft else ""
            print(f"  [{status}] {label:14s}{note}", flush=True)
            for d in hard[:6]:
                print(f"           {d}", flush=True)
            for d in soft[:2]:
                print(f"           {d}", flush=True)
            overall &= not hard
            report.append(
                {
                    "checkpoint": name,
                    "case": label,
                    "pass": not hard,
                    "hard_diffs": hard,
                    "action_only_diffs": soft,
                }
            )

    parity_path = os.path.join(a.out, "parity.json")
    with open(parity_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'PASS' if overall else 'FAIL'} overall -> {parity_path}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
