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

# Best quality option for 48GB machines (recommended)
python qwen_mixed_quant.py --variant D --model Qwen/Qwen3.5-9B

# Fast but still very good
python qwen_mixed_quant.py --variant C --model Qwen/Qwen3.5-9B
```

The output directory names are now descriptive (e.g. `Qwen3.5-9B-MLX-5bit-vlm-high-fidelity`).

Test with images or pure text using `mlx_lm` or the `mlx-vlm` package.

## Recommended Variants for M4 Pro / Max 48GB

| Variant | Output suffix (example)              | Bulk | Protected bands          | Vision / MTP | Approx. size (Qwen3.5-9B) | Quality on 48GB          | Best for on M4 Pro                  |
|---------|--------------------------------------|------|--------------------------|--------------|---------------------------|--------------------------|-------------------------------------|
| A       | 4bit-8bit-vision                     | 4-bit| 8-bit I/O only           | 8-bit        | ~5.2 GB                   | Good                     | Fast chat, high throughput          |
| C       | 4bit-vlm-balanced                    | 4-bit| 6-bit (linear_attn+down) | 8-bit        | ~5.8 GB                   | Very good / Excellent    | Daily driver, coding, RAG, images   |
| D       | 5bit-vlm-high-fidelity               | 5-bit| 6-bit (linear_attn+down) | 8-bit        | ~6.8 GB                   | **Best practical**       | Long context, reasoning, summarization |

**Strong recommendation for 48GB machines**: Start with **Variant D**. On an M4 Pro 48GB you will still have plenty of headroom for 32k–128k context + vision encoding.

Variant C is the better choice if you want maximum speed or plan to run multiple models / very large batches.

## The Four Variants (Technical)

| Variant | Key behavior for Qwen3.5 VLMs |
|---------|-------------------------------|
| A       | Pure speed. 4-bit everywhere except critical I/O and vision tower kept at 8-bit. |
| B       | Uses mlx-lm's built-in `mixed_4_6` string recipe (good baseline, less tuned for linear_attn + MTP). |
| C       | 4-bit bulk + 6-bit on the most important projections in protected layers + 8-bit vision/MTP/embeds. Excellent balance. |
| D       | Same protection as C but 5-bit bulk. Currently the highest quality recipe that still runs very comfortably on 48GB. |

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
  --variant {A,B,C,D} \
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

## Using on M4 Pro 48GB (Practical Guidance)

Qwen3.5-9B quantized with Variant D typically uses **~6.5–7.5 GB** for the weights. This leaves very comfortable headroom on a 48GB M4 Pro for:

- 64k–128k context (with KV cache)
- Vision encoding
- Running alongside other tools / browsers

**Typical comfortable setups on 48GB:**

- `Qwen3.5-9B` (Variant D) → 32k–128k context, images, excellent quality
- `Qwen3.5-27B` (Variant C or D, aggressive) → possible with 16k–32k context
- `Qwen3.5-35B-A3B` MoE (Variant D) → very promising because of low active parameters

For even larger context on 48GB, consider also enabling MLX KV cache quantization when running (see mlx-lm docs).

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
    mlx_path="./Qwen3-32B-MLX-5bit-mixed",
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

## Adapting for Larger Qwen3.5 Models on 48GB

This repo is now optimized for exactly the models that fit well on M4 Pro 48GB:

- `Qwen/Qwen3.5-9B` → Variant D (excellent)
- `Qwen/Qwen3.5-27B` → Variant C or D (aggressive 4/5-bit with 8-bit vision)
- `Qwen/Qwen3.5-35B-A3B` (MoE) → Very interesting target — low active parameter count

For these models the script now auto-detects many characteristics and the predicate makes the right decisions for vision + MTP + linear attention.

For even larger or very different architectures you can still subclass or copy `get_qwen_mixed_predicate` and customize the protection rules.

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
