"""Export each Laya checkpoint's DecisionModel to ONNX and verify it against PyTorch.

Feasibility spike: answers whether the ModernBERT encoder plus Laya's custom
typed-decision head survive an ONNX export with dynamic batch / sequence /
marker axes, which is the prerequisite for running Laya from any non-Python
ONNX Runtime binding.

    uv run python scripts/export_onnx.py                 # all three checkpoints
    uv run python scripts/export_onnx.py --only english
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from laya.agent import Agent
from laya.common import QTYPES, build_sequence, collate_items

REPO = "convaiinnovations/laya"

# (name, subfolder) -- the English checkpoint lives at the repo root.
CHECKPOINTS: list[tuple[str, str | None]] = [
    ("english", None),
    ("typed-decisions", "typed-decisions"),
    ("multilingual", "multilingual"),
]

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]

DYNAMIC_AXES = {
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "marker_pos": {0: "batch", 1: "markers"},
    "marker_mask": {0: "batch", 1: "markers"},
    "qtype": {0: "batch"},
    "logits": {0: "batch", 1: "markers"},
    "act_logits": {0: "batch"},
}


class ExportWrapper(torch.nn.Module):
    """Pins away DecisionModel.forward's optional `detach_encoder` flag.

    The traced graph must have exactly the five tensor inputs; a bool keyword
    would otherwise
    be baked in as a constant input and confuse the ONNX signature.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)


def small_questions() -> dict:
    """3 questions, one of each type, few options -> small markers axis."""
    return {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want in `message`?",
            "criteria": {
                "refund": "money returned",
                "technical_help": "a bug or outage",
                "billing_question": "an invoice or plan question",
                "other": "none of the other options fits",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer sound?",
            "criteria": ["calm", "concerned", "annoyed", "very angry"],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does `message` communicate time pressure?",
        },
    }


def large_questions() -> dict:
    """5 questions with an 11-option choice: different markers axis and temp bucket."""
    q = small_questions()
    q["department"] = {
        "type": "choice",
        "instructions": "Which department owns this?",
        "criteria": {
            f"dept_{i}": f"description for department number {i}" for i in range(11)
        },
    }
    q["churn_risk"] = {
        "type": "noul",
        "instructions": "Does the message suggest the customer may cancel?",
    }
    return q


SHORT_STATE = {"message": "My card was charged twice and I need this fixed today."}
LONG_STATE = {
    "message": "My card was charged twice and I need this fixed today. " * 40,
    "account": {"plan": "enterprise", "tenure_months": 37, "mrr": 4200},
    "history": ["opened ticket", "escalated", "still unresolved"] * 10,
}


def make_batch(agent: Agent, questions: dict, state) -> dict[str, torch.Tensor]:
    """Reproduce exactly what Agent.system_one feeds the model."""
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items = []
    for qid in questions:
        q = agent._to_internal(questions[qid])
        seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    return collate_items([items], agent.tok.pad_token_id)


def torch_forward(model: torch.nn.Module, b: dict[str, torch.Tensor]):
    with torch.no_grad():
        logits, act = model(
            b["input_ids"],
            b["attention_mask"],
            b["marker_pos"],
            b["marker_mask"],
            b["qtype"],
        )
    return logits.float().numpy(), act.float().numpy()


def ort_forward(sess, b: dict[str, torch.Tensor]):
    feed = {
        "input_ids": b["input_ids"].numpy(),
        "attention_mask": b["attention_mask"].numpy(),
        "marker_pos": b["marker_pos"].numpy(),
        "marker_mask": b["marker_mask"].numpy(),
        "qtype": b["qtype"].numpy(),
    }
    out = sess.run(OUTPUT_NAMES, feed)
    return out[0], out[1]


def export_one(
    name: str, subfolder: str | None, out_root: str, tol: float, opset: int = 18
) -> dict:
    import onnxruntime as ort

    result = {"checkpoint": name, "ok": False}
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)

    t0 = time.time()
    agent = Agent(REPO, device="cpu", subfolder=subfolder)
    pos_emb = getattr(agent.model.encoder.config, "position_embedding_type", "?")
    print(
        f"  loaded in {time.time() - t0:.1f}s  "
        f"encoder={agent.cfg.get('encoder')}  max_len={agent.cfg.get('max_len')}  "
        f"pos_emb={pos_emb}",
        flush=True,
    )

    wrapper = ExportWrapper(agent.model).eval()

    # Export on the SMALL batch, verify on BOTH -- that is what proves the axes
    # are dynamic
    # rather than frozen at the traced shapes.
    b_small = make_batch(agent, small_questions(), SHORT_STATE)
    b_large = make_batch(agent, large_questions(), LONG_STATE)
    print(
        f"  export shape: ids={tuple(b_small['input_ids'].shape)} "
        f"markers={tuple(b_small['marker_pos'].shape)}",
        flush=True,
    )
    print(
        f"  verify shape: ids={tuple(b_large['input_ids'].shape)} "
        f"markers={tuple(b_large['marker_pos'].shape)}",
        flush=True,
    )

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")

    args = (
        b_small["input_ids"],
        b_small["attention_mask"],
        b_small["marker_pos"],
        b_small["marker_mask"],
        b_small["qtype"],
    )

    t0 = time.time()
    used = None
    errors = {}
    # dynamo first: torch.export traces with symbolic shapes, which is what stops
    # ModernBERT's
    # internal attention reshapes being frozen at the traced sequence length. The legacy
    # TorchScript exporter bakes them in as constants (it emits TracerWarnings
    # saying so).
    for exporter in ("dynamo", "legacy"):
        try:
            if exporter == "dynamo":
                auto = torch.export.Dim.AUTO
                torch.onnx.export(
                    wrapper,
                    args,
                    onnx_path,
                    input_names=INPUT_NAMES,
                    output_names=OUTPUT_NAMES,
                    dynamic_shapes={
                        "input_ids": {0: auto, 1: auto},
                        "attention_mask": {0: auto, 1: auto},
                        "marker_pos": {0: auto, 1: auto},
                        "marker_mask": {0: auto, 1: auto},
                        "qtype": {0: auto},
                    },
                    opset_version=opset,
                    dynamo=True,
                )
            else:
                torch.onnx.export(
                    wrapper,
                    args,
                    onnx_path,
                    input_names=INPUT_NAMES,
                    output_names=OUTPUT_NAMES,
                    dynamic_axes=DYNAMIC_AXES,
                    opset_version=opset,
                    do_constant_folding=True,
                    dynamo=False,
                )
            used = exporter
            break
        except Exception as e:  # noqa: BLE001 - we want the reason for every failed exporter
            errors[exporter] = f"{type(e).__name__}: {e}"
            print(
                f"  {exporter} exporter failed: {type(e).__name__}: {str(e)[:400]}",
                flush=True,
            )

    if used is None:
        result["errors"] = errors
        return result

    size_mb = (
        sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
        / 1e6
    )
    print(
        f"  exported via {used} in {time.time() - t0:.1f}s -> {size_mb:.0f} MB",
        flush=True,
    )
    result.update(
        exporter=used, size_mb=round(size_mb, 1), export_s=round(time.time() - t0, 1)
    )

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.time()
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    print(f"  ORT session in {time.time() - t0:.1f}s", flush=True)

    checks = {}
    for label, b in (("traced_shape", b_small), ("different_shape", b_large)):
        tl, ta = torch_forward(wrapper, b)
        # Compare only real option slots: padded slots are -1e4 on both sides and would
        # flatter the diff.
        m = b["marker_mask"].numpy()
        try:
            ol, oa = ort_forward(sess, b)
        except Exception as e:  # noqa: BLE001 - a shape failure here is the result, not a crash
            checks[label] = {
                "shape": list(b["input_ids"].shape),
                "markers": list(b["marker_pos"].shape),
                "pass": False,
                "error": f"{type(e).__name__}: {str(e)[:300]}",
            }
            print(f"  [FAIL] {label:16s} ORT run failed: {str(e)[:220]}", flush=True)
            continue

        d_logits = float(np.abs(tl[m] - ol[m]).max())
        d_act = float(np.abs(ta - oa).max())

        # What actually matters is the probability vector the server returns,
        # not raw logits. `m` is default-bound so the closure cannot capture a
        # later iteration's mask.
        def softmax_rows(x, m=m):
            x = np.where(m, x, -np.inf)
            x = x - np.nanmax(
                np.where(np.isfinite(x), x, -np.inf), axis=1, keepdims=True
            )
            e = np.where(m, np.exp(x), 0.0)
            return e / e.sum(axis=1, keepdims=True)

        d_prob = float(np.abs(softmax_rows(tl) - softmax_rows(ol)).max())
        ok = d_logits < tol and d_prob < tol
        checks[label] = {
            "shape": list(b["input_ids"].shape),
            "markers": list(b["marker_pos"].shape),
            "max_abs_logit_diff": d_logits,
            "max_abs_act_diff": d_act,
            "max_abs_prob_diff": d_prob,
            "pass": ok,
        }
        flag = "PASS" if ok else "FAIL"
        print(
            f"  [{flag}] {label:16s} logits={d_logits:.3e}  act={d_act:.3e}  "
            f"prob={d_prob:.3e}",
            flush=True,
        )

    result["checks"] = checks
    result["ok"] = all(c["pass"] for c in checks.values())
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="export just this checkpoint")
    ap.add_argument("--out", default="build/onnx", help="output root (gitignored)")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument(
        "--opset",
        type=int,
        default=18,
        help="18+ is required: ModernBERT's GeGLU lowers to Split(num_outputs)",
    )
    a = ap.parse_args()

    targets = [c for c in CHECKPOINTS if not a.only or c[0] == a.only]
    if not targets:
        print(f"no checkpoint named {a.only!r}; known: {[c[0] for c in CHECKPOINTS]}")
        return 2

    results = []
    for name, sub in targets:
        try:
            results.append(export_one(name, sub, a.out, a.tol, a.opset))
        except Exception as e:  # noqa: BLE001 - one bad checkpoint should not hide the others
            import traceback

            traceback.print_exc()
            results.append(
                {"checkpoint": name, "ok": False, "error": f"{type(e).__name__}: {e}"}
            )

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for r in results:
        print(
            f"  {r['checkpoint']:18s} {'PASS' if r['ok'] else 'FAIL'}"
            f"  exporter={r.get('exporter', '-')}  size={r.get('size_mb', '-')} MB"
        )
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  report -> {os.path.join(a.out, 'report.json')}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
