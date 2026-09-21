# Running Laya outside Python

The server runs Laya in-process through the Python `laya` package. This document records what
the checkpoints actually are, which other languages can run them, and the results of a spike
that exported all three to ONNX and checked them against Laya's own output.

**Verdict: confirmed viable.** All three checkpoints export to ONNX and reproduce the API
payload exactly, so any language with an ONNX Runtime binding can serve Laya in-process. ONNX
Runtime on CPU also turned out to be about twice as fast as PyTorch.

## What the model actually is

This is the constraint that decides everything else. Laya is **not** a stock HuggingFace model,
so "load the safetensors in language X" does not work — `AutoModel` alone gives you the encoder
and none of the decision logic.

`DecisionModel` (`common.py:89`) is a standard **ModernBERT** encoder plus a custom head:

| Component | Definition |
| --- | --- |
| `encoder` | HF `AutoModel`, `model_type: modernbert`, sdpa attention |
| `head` | 2 × `nn.TransformerEncoderLayer(d, d/64 heads, 4d ffn, norm_first=True)` |
| `type_emb` | `nn.Embedding(3, d)` — one vector per question type, added at every position |
| `scorer` | `LayerNorm(d) → Linear(d,d) → GELU → Linear(d,1)` |
| `act_head` | `Linear(d+4, 256) → GELU → Linear(256, 2)` |

The forward signature is five tensors in, two out:

```
(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits, act_logits)
```

Every operation in the head is ordinary — gather, softmax, LayerNorm, GELU, Linear,
masked_fill. The portability problem is packaging, not mathematics.

### How one question becomes one forward pass

`build_sequence` (`common.py:49`) lays out each question as:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

The position of each `[MASK]` is recorded as a **marker**. After the encoder and head run,
those positions are gathered and pushed through `scorer` to produce exactly one logit per
option. That is why the model answers a whole schema in a single non-autoregressive pass: each
question is one row of the batch, and each option is one marker in that row.

### The checkpoints

| Checkpoint | Encoder | hidden / layers / heads | vocab | `max_len` | `head_max_len` | weights |
| --- | --- | --- | --- | --- | --- | --- |
| `english` (repo root) | `answerdotai/ModernBERT-large` | 1024 / 28 / 16 | 50,368 | 512 | 192 | 843 MB |
| `typed-decisions` | `answerdotai/ModernBERT-large` | 1024 / 28 / 16 | 50,368 | 1024 | 256 | 843 MB |
| `multilingual` | `jhu-clsp/mmBERT-base` | 768 / 22 / 12 | 256,000 | 1024 | 256 | 644 MB |

Both ModernBERT-large checkpoints are 421.3M parameters, stored as fp16 on disk (205 F16
tensors plus one F32 `temperature` buffer) and loaded into an fp32 module. `multilingual` uses
`position_embedding_type: "sans_pos"`, which is non-default for ModernBERT — it was the
likeliest place for a silent divergence and it turned out fine.

## The three pieces any port must reproduce

1. **Tokenize and build the sequence** (`common.py:49`). The risky one. A probability-identical
   port has to match the 48-token cap per option, the `opt_budget < 16` re-truncation, the
   `head_max_len` clamp via `max(8, opt_budget)`, the `.replace(mask_tok, " ")` stripping on
   every text field, left vs right state truncation, and the final `ids[:max_len]` plus
   `markers < max_len` filter.
2. **The forward pass.** The easy one — this is what the ONNX graph carries.
3. **Postprocess** (`agent.py:294-343`). Per-bucket temperature, stable softmax, entropy
   confidence `1 - H(p)/log(k)`, expected value for `score`, `p[1]` for `noul`. About 45 lines
   of arithmetic.

For `Router` parity, add `lang.py` (182 lines of dependency-free script and language detection)
and `router.py:241 route()`. A server can skip both by passing `model=` explicitly.

### Temperature is not optional

Each checkpoint carries `temperature` (a 3-element list indexed by question type) and
`temperature_by_options`, a lookup keyed `"<type>:<bucket>"` where the bucket is `2`, `3-5`,
`6-10` or `11+`. The lookup wins when it hits; the list is the fallback.

This matters more than it looks. `typed-decisions` has near-neutral base temperatures
(~1.01–1.06) but inherits `english`'s full `temperature_by_options` table, so a 4-option choice
is actually divided by 1.7602, not ~1.01. An 11-option choice is divided by 0.1006, which
sharpens the distribution dramatically. `multilingual` ships an empty table and flat 1.0
temperatures, so it applies no scaling at all. A port that ignores `temperature_by_options`
produces plausible-looking but wrong probabilities on two of the three checkpoints.

## Options for other languages

### A. Keep Python, cross the boundary over HTTP — already built

This server exposes a Jev-compatible API, so any language with an HTTP client already works.
Zero fidelity risk, no artifact to keep in sync with the checkpoint. The cost is a process
boundary and a Python runtime in the deployment. **This is the baseline any port has to beat.**

### B. ONNX export plus an ONNX Runtime binding — recommended

Export `DecisionModel.forward` once from Python, then run it from Java/Kotlin/Scala, C#, Rust
(`ort`), C/C++, Go, Node/TypeScript (and browsers via ORT-WASM), Swift/Objective-C, Ruby or
Julia. Per-language work is pieces 1 and 3 above plus wiring; the 28-layer encoder stays inside
the graph.

For the JVM specifically:

- `com.microsoft.onnxruntime:onnxruntime` for `OrtEnvironment` / `OrtSession`.
- `ai.djl.huggingface:tokenizers` — a JNI wrapper over the same Rust `tokenizers` crate Python
  uses, loading the checkpoint's `tokenizer.json` unmodified (3.5 MB for English, 34 MB for
  multilingual). Using the identical tokenizer implementation removes the largest single source
  of drift for free.
- Or DJL end to end if its `Translator` and batching abstractions are wanted.

### C. Native reimplementation

Reimplement encoder and head directly against `model.safetensors` — Rust with Candle (verify
current ModernBERT coverage first), or Swift with MLX on Apple silicon only. Much more work.
Justified only to eliminate the ONNX dependency or to train outside Python.

### D. TorchScript / LibTorch

`torch.jit.trace` run from C++, JVM or `tch-rs`. Tracing ModernBERT is riskier than the ONNX
export and LibTorch is a heavy native dependency. No advantage over B here.

### Non-starters

| Runtime | Why not |
| --- | --- |
| llama.cpp / GGUF / Ollama | Decoder-oriented. Even where ModernBERT embeddings are supported, there is no way to express the typed decision head or the marker gather. |
| vLLM / TGI / SGLang | Built for autoregressive generation. Laya is a single non-autoregressive forward pass. |
| transformers.js alone | Runs the encoder, not the custom head. Needs the ONNX export from B anyway. |

## Spike results

Run on macOS (Darwin 27), Python 3.12.8, torch 2.14.0, transformers 5.17.0, onnx 1.23.0,
onnxruntime 1.30.0, onnxscript 0.7.2. CPU execution provider, fp32.

| Checkpoint | Export | ONNX size | max abs logit diff | max abs prob diff | Payload parity |
| --- | --- | --- | --- | --- | --- |
| `english` | dynamo, opset 18 | 1689 MB | 1.6e-05 | 2.3e-06 | 6/6 exact |
| `typed-decisions` | dynamo, opset 18 | 1689 MB | 6.5e-06 | 1.3e-06 | 6/6 exact |
| `multilingual` | dynamo, opset 18 | 1291 MB | 2.1e-05 | 3.1e-06 | 6/6 exact |

Each checkpoint was exported at one shape and verified at two: the traced shape and a
deliberately different one (more questions, more options, longer sequence). That is what proves
the axes are genuinely dynamic rather than frozen.

**Payload parity** is the stronger claim. `scripts/verify_parity.py` reimplements collate and
postprocess in plain numpy with no torch, runs the exported graph, and compares the complete
rounded response against `Agent.system_one` — `choice`, `score`, `noul`, every `probabilities`
map, `confidence`, and the `action` block. All six cases matched exactly on all three
checkpoints: short and long text, dict and plain-string state, an 11-option choice (exercising
the `choice:11+` bucket), CJK and accented Unicode, and empty input.

### Performance

ONNX Runtime CPU is roughly twice as fast as PyTorch CPU for a 3-question triage call:

| Runtime | median | p10 | p90 |
| --- | --- | --- | --- |
| PyTorch CPU | 280.7 ms | 252.6 ms | 311.1 ms |
| ONNX Runtime CPU | 140.7 ms | 104.8 ms | 232.0 ms |

The in-process win is therefore not only deployment simplicity.

## Export gotchas

Three traps, each of which fails in a way that looks like something else.

1. **The legacy TorchScript exporter silently produces a shape-frozen graph.** It exports
   without error and passes at the traced shape, then fails at any other sequence length with
   `Reshape ... Input shape:{512,5,1024}, requested shape:{68,80,64}`. ModernBERT's attention
   reshapes are baked in as constants; the exporter says so in `TracerWarning`s that are easy to
   scroll past. Use `dynamo=True` (the `torch.export`-based exporter) with
   `torch.export.Dim.AUTO`. Passing `dynamic_axes` to the legacy exporter is not enough.
2. **The opset must be 18 or higher.** ModernBERT's GeGLU MLP lowers to `Split` with the
   `num_outputs` attribute, which is opset 18+. At opset 17 the export appears to succeed and
   ONNX Runtime then rejects the file with
   `INVALID_GRAPH: Unrecognized attribute: num_outputs for operator Split`.
3. **`onnxscript` is a separate dependency** from `onnx`, and without it the dynamo exporter
   fails over to the legacy one — straight back into trap 1.

One more, already handled upstream: ModernBERT's `torch.compile` decorators block
`torch.onnx.export`, which is what Optimum's ModernBERT support had to work around. Laya
already disables this at `agent.py:190` (`reference_compile = False`) for unrelated latency
reasons, so the prerequisite is satisfied by the library as shipped.

## Reproducing

```bash
uv sync
uv run python scripts/export_onnx.py                    # all three, ~20s export each
uv run python scripts/export_onnx.py --only english     # one checkpoint
uv run python scripts/verify_parity.py --only english   # end-to-end payload diff
```

`scripts/export_onnx.py` checks logits at two shapes and writes `build/onnx/report.json`.
`scripts/verify_parity.py` checks the full API payload and writes `build/onnx/parity.json`.
Artifacts land in `build/onnx/<checkpoint>/` as `model.onnx` plus `model.onnx.data` (external
data format), about 4.5 GB in total for all three. `build/` is gitignored.

The numpy postprocess in `verify_parity.py` is written deliberately as a **port reference** — it
is torch-free and meant to be transliterated line for line into Kotlin, Rust or Go.

## Remaining risks

- **`build_sequence` is still unproven outside Python.** The spike reused Laya's own Python
  implementation in order to isolate the ONNX question, so it proves the graph and the
  postprocess port cleanly and says nothing about a reimplementation of the sequence builder.
  That remains the highest-risk piece, and it fails quietly: an off-by-one in the token budget
  produces slightly wrong probabilities, not a crash. Port it first and test it against golden
  vectors before wiring up ONNX Runtime at all.
- **Artifact size.** The exports are fp32 and total about 4.5 GB. Quantization is a separate
  step that needs its own validation pass — do not assume parity survives it.
- **`act_logits` drift is ~1e-3**, against ~1e-5 for the decision logits. It changed no rounded
  `act_probability` in the test corpus, and `translate.py` strips the block before the response
  leaves the server, but nothing load-bearing should be built on it without wider testing.
- **Checkpoint and export must stay in sync.** The ONNX file is a derived artifact. If the
  upstream checkpoint is updated, the export and the parity run have to be repeated.

## If this goes further

1. Port `build_sequence` to the target language and test it against golden vectors first.
2. Use `verify_parity.py`'s numpy postprocess as the reference implementation.
3. Decide how the ~4.5 GB of ONNX artifacts are hosted and versioned. That is a deployment
   question, not a code question, and it is the main ongoing cost of option B over option A.
