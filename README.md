# Mixed-Precision Quantization for Qwen3 / Qwen3.5 on MLX

High-quality mixed-bit quantization recipes **specifically tuned for modern Qwen3.5 VLMs** (Qwen3.5-9B, 27B, 35B-A3B, etc.) as well as classic Qwen3 / Qwen2.5 text models.

The predicates correctly handle:
- Nested language towers (`model.language_model.layers.*`)
- Qwen3.5 linear attention (Gated DeltaNet style `linear_attn.*` projections)
- MTP (Multi-Token Prediction) heads
- Vision / visual towers (kept at 8-bit by default for VLM quality)

Excellent results on Apple Silicon, especially **M4 Pro / Max machines with 48GB unified memory**.

## Why This Exists

Uniform 4-bit quantization of 32B–72B+ class models often produces a noticeable quality cliff:

- Degraded long-context coherence and instruction following
- Increased reasoning errors on hard tasks
- Weaker performance on summarization and creative writing

The root cause is that **not all weights are equally important**. Embeddings, the language modeling head, value projections in attention, and down projections in the FFN carry disproportionate signal. Protecting a small strategic subset at 6-bit or 8-bit while keeping the bulk at 4-bit or 5-bit recovers most of the lost quality with only modest memory cost.

This approach mirrors the philosophy behind advanced GGUF K-quants and other calibration-aware techniques, but is implemented as clean, fast MLX conversion predicates that run on Apple Silicon without heavy calibration datasets.

The techniques here will become even more valuable as models scale to 70B–200B+ parameter regimes where every saved gigabyte matters and full 8-bit or 6-bit inference is impractical.

## Quick Start — Qwen3.5-9B on M4 Pro 48GB

```bash
pip install mlx-lm huggingface_hub

# Best quality option for 48GB machines (recommended) -> MLX-Q5_K6_L
python qwen_mixed_quant.py --variant D --model Qwen/Qwen3.5-9B

# Fast but still very good -> MLX-Q4_K6_L
python qwen_mixed_quant.py --variant C --model Qwen/Qwen3.5-9B

# Direct analog of llama.cpp Q6_K_L (6-bit body + 8-bit embeds/output) -> MLX-Q6_K_L
python qwen_mixed_quant.py --variant E --model Qwen/Qwen3.5-9B
```

Output directory names follow a **GGUF-parallel naming convention** (e.g. `Qwen3.5-9B-MLX-Q6_K_L`).

### Naming convention

The MLX suffix mirrors llama.cpp's `Q<bits>_K_<S|M|L>` GGUF tags:

```
MLX-Q<body>_K[<protect>]_<embed>
       │        │          └─ embeddings/output (+vision/MTP): L = 8-bit ("large")
       │        └─ protected-projection tier (v/down/out_proj in bands), in bits;
       │           omitted when it collapses to the body (uniform body)
       └─ bulk/body bit-width (the dominant tier)
_K = MLX group-wise quantization (the analog of GGUF K-quant superblocks)
```

So **`MLX-Q6_K_L`** = 6-bit body + 8-bit embeddings/output — the direct analog of GGUF **`Q6_K_L`**.
`MLX-Q5_K6_L` = 5-bit body with a 6-bit protected-projection tier and 8-bit embeddings (no plain GGUF equivalent — this mid-tier protection is MLX-specific).

Test with images or pure text using `mlx_lm` or the `mlx-vlm` package.

## Recommended Variants for M4 Pro / Max 48GB

| Variant | Output suffix      | Bulk | Protected bands          | Vision / MTP | Approx. size (Qwen3.5-9B) | Quality on 48GB          | Best for on M4 Pro                  |
|---------|--------------------|------|--------------------------|--------------|---------------------------|--------------------------|-------------------------------------|
| A       | `MLX-Q4_K_L`       | 4-bit| none (8-bit I/O only)    | 8-bit        | ~5.2 GB                   | Good                     | Fast chat, high throughput          |
| C       | `MLX-Q4_K6_L`      | 4-bit| 6-bit (linear_attn+down) | 8-bit        | ~5.8 GB                   | Very good / Excellent    | Daily driver, coding, RAG, images   |
| D       | `MLX-Q5_K6_L`      | 5-bit| 6-bit (linear_attn+down) | 8-bit        | ~6.8 GB                   | **Best practical**       | Long context, reasoning, summarization |
| E       | `MLX-Q6_K_L`       | 6-bit| none (uniform body)      | 8-bit        | ~6.5 GB (~9.6 GB peak)    | **Q6_K_L-equivalent**    | Matching llama.cpp Q6_K_L 1:1       |

**Strong recommendation**: start with **Variant D (`MLX-Q5_K6_L`)** — the best practical quality that still leaves comfortable headroom on a 48GB machine. Reach for the others when:

- **C (`MLX-Q4_K6_L`)** — you want maximum speed, or plan to run multiple models / very large batches.
- **E (`MLX-Q6_K_L`)** — you want 1:1 parity with a llama.cpp `Q6_K_L` build (same footprint and quality).

### Memory and context headroom

Qwen3.5-9B at Variant D uses **~6.5–7.5 GB** for weights, leaving an M4 Pro 48GB plenty of room for:

- 64k–128k context (with KV cache)
- Vision encoding
- Running alongside other tools / browsers

For even larger context, enable MLX KV-cache quantization at run time (see mlx-lm docs).

### Scaling to larger Qwen3.5 models

The script auto-detects vision / MTP / linear-attention characteristics, so the same variants apply to bigger models:

| Model | Suggested variant | Notes |
|---|---|---|
| `Qwen/Qwen3.5-9B` | D | 32k–128k context + images, excellent quality |
| `Qwen/Qwen3.5-27B` | C or D | ~16k–32k context (aggressive 4/5-bit + 8-bit vision) |
| `Qwen/Qwen3.5-35B-A3B` (MoE) | D | very promising — low active parameter count |

For even larger or unusual architectures, copy or subclass `get_qwen_mixed_predicate` and customize the protection rules.

## The Variants (Technical)

| Variant | Key behavior for Qwen3.5 VLMs |
|---------|-------------------------------|
| A       | Pure speed. 4-bit everywhere except critical I/O and vision tower kept at 8-bit. |
| B       | Uses mlx-lm's built-in `mixed_4_6` string recipe (good baseline, less tuned for linear_attn + MTP). |
| C (`MLX-Q4_K6_L`) | 4-bit bulk + 6-bit on the most important projections in protected layers + 8-bit vision/MTP/embeds. Excellent balance. |
| D (`MLX-Q5_K6_L`) | Same protection as C but 5-bit bulk. Currently the highest quality recipe that still runs very comfortably on 48GB. |
| E (`MLX-Q6_K_L`)  | Uniform 6-bit body + 8-bit embeddings/output (+vision/MTP) — the direct analog of llama.cpp's `Q6_K_L`: the 6-bit floor mirrors the Q6_K body, the 8-bit embeddings/output mirror the `_L` Q8_0 bump. ~6.56 bpw effective, footprint-matched to the real GGUF. For a heavier "Q6_K_L+" that also lifts the band-critical projections to 8-bit, set `high_bits=8` in `variant_e_q6_k_l`. |

**Protected projections in modern Qwen3.5 models** (in the selected layer bands):
- Classic: `v_proj`, `down_proj`
- Qwen3.5 linear attention: `in_proj_qkv`, `out_proj`, `in_proj_z`, `in_proj_a/b`, `down_proj`

The layer selection heuristic (first 1/8 + last 1/8 + every 3rd) is applied to the language tower only.

## Installation

```bash
pip install mlx-lm
# Optional but recommended for faster downloads
pip install huggingface_hub[cli]
```

The script itself has no additional dependencies beyond `mlx-lm`.

## Full CLI Reference

```
python qwen_mixed_quant.py \
  --variant {A,B,C,D,E} \
  --model MODEL_ID_OR_PATH \
  --output-dir ./mlx_models \
  --num-layers N \
  --trust-remote-code
```

- `--variant`: Quantization strategy (default: D). See recommendations above.
- `--model`: Hugging Face repo ID or local path (default: Qwen/Qwen3.5-9B). Works great with Qwen3.5-9B, 27B, Qwen3-32B, etc.
- `--num-layers`: Almost always auto-detected now (works for Qwen3.5 VLMs via `text_config`). You only need to pass it for very unusual models.
- `--trust-remote-code`: Rarely needed for official Qwen models.

The output directory name is derived automatically from the model name and chosen variant.

## Programmatic Use

All variants (including the new `get_qwen_mixed_predicate`) are simple callables compatible with `mlx_lm.convert(..., quant_predicate=...)`.

```python
from qwen_mixed_quant import get_qwen_mixed_predicate

pred = get_qwen_mixed_predicate(
    num_layers=32,
    high_bits=6,
    low_bits=5,
    vision_bits=8,
    protect_mtp=True,
)
```

```python
from qwen_mixed_quant import variant_d_high_fidelity, get_mixed_layer_predicate

# Direct use
from mlx_lm import convert
convert(
    hf_path="Qwen/Qwen3-32B",
    mlx_path="./Qwen3-32B-MLX-Q5_K6_L",
    quantize=True,
    quant_predicate=lambda p, l: variant_d_high_fidelity(p, l, num_layers=64),
)

# Or build a custom predicate (e.g. protect more layers for a 72B model)
pred = get_mixed_layer_predicate(num_layers=80, high_bits=6, low_bits=5, group_size=64)
```

## How the Modern Predicate Works

The primary function is now `get_qwen_mixed_predicate` (the old one still exists for compatibility).

It understands the full modern Qwen3.5 family structure:

- Correctly parses `model.language_model.layers.N....` (VLM nesting)
- Protects the right tensors inside **linear_attn** modules (`in_proj_qkv`, `out_proj`, etc.)
- Keeps the entire vision tower at 8-bit (or skips quantization) by default
- Gives the MTP head 8-bit protection
- Applies the proven early/late/periodic band heuristic only to the language tower

This makes the tool dramatically more effective for the actual Qwen3.5-9B, 27B, and MoE models people want to run on Apple Silicon today.

## Evaluating Quantized Models

After conversion, good quick checks include:

```bash
# Perplexity on a small validation set (use mlx_lm or your own loop)
python -m mlx_lm.generate --model <path> --prompt "..." --max-tokens 512

# Long context test: feed a 20k–32k token document and ask for faithful summary
# Reasoning test: GSM8K, MATH, or custom hard prompts
```

Compare against:
- The unquantized BF16 reference (if you have the VRAM)
- The official `mlx-lm` 4-bit or `mixed_4_6` baseline

## Project Structure

```
qwen_mixed_quant.py   # All variants + CLI
README.md             # This file
```

The entire quantization strategy lives in one well-commented file so it is easy to audit, modify, and vendor into larger pipelines.

## License

MIT

## Acknowledgments

- Built on top of the excellent `mlx-lm` conversion infrastructure from Apple and the MLX community.
- Quantization design ideas draw from GGUF K-quant patterns, SmoothQuant literature, and practical experience shipping large models on Apple Silicon.

---

If you use these recipes for 32B+ models or discover better layer/bit combinations, please open an issue or PR. The goal is to make high-quality quantized Qwen (and future large models) practical for everyone running on laptops and workstations.
