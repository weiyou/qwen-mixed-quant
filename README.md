# Mixed-Precision Quantization for Qwen on MLX

High-quality mixed-bit quantization recipes for Qwen3 and Qwen3.5 models (and extensible to other large decoder-only models). Designed to deliver near-BF16 quality at 4-bit and 5-bit memory footprints.

## Why This Exists

Uniform 4-bit quantization of 32B–72B+ class models often produces a noticeable quality cliff:

- Degraded long-context coherence and instruction following
- Increased reasoning errors on hard tasks
- Weaker performance on summarization and creative writing

The root cause is that **not all weights are equally important**. Embeddings, the language modeling head, value projections in attention, and down projections in the FFN carry disproportionate signal. Protecting a small strategic subset at 6-bit or 8-bit while keeping the bulk at 4-bit or 5-bit recovers most of the lost quality with only modest memory cost.

This approach mirrors the philosophy behind advanced GGUF K-quants and other calibration-aware techniques, but is implemented as clean, fast MLX conversion predicates that run on Apple Silicon without heavy calibration datasets.

The techniques here will become even more valuable as models scale to 70B–200B+ parameter regimes where every saved gigabyte matters and full 8-bit or 6-bit inference is impractical.

## Quick Start

```bash
pip install mlx-lm

# Recommended high-fidelity variant (best quality/speed trade-off)
python qwen_mixed_quant.py --variant D --model Qwen/Qwen3.5-9B

# Test the result
python -m mlx_lm.generate --model ./mlx_models/Qwen3.5-9B-MLX-5bit-enhanced-mixed \
  --prompt "Explain the key differences between Qwen3 and Qwen2.5 in one paragraph." \
  --max-tokens 256
```

## The Four Variants

| Variant | Name in output                  | Embed / lm_head | Bulk bits | Key projections (selected layers) | Relative memory | Quality level       | Recommended use case                          |
|---------|---------------------------------|-----------------|-----------|-----------------------------------|-----------------|---------------------|-----------------------------------------------|
| A       | 4bit-8bit-embed                 | 8-bit           | 4-bit     | —                                 | Lowest          | Good                | Maximum speed / edge deployment               |
| B       | mixed-4-6 (official)            | mixed           | 4-bit     | 6-bit (mlx-lm recipe)             | Low             | Very good           | Drop-in replacement for built-in mixed_4_6    |
| C       | 4bit-enhanced-mixed             | 8-bit           | 4-bit     | 6-bit                             | Low +           | Excellent           | Daily driver, chat, coding, RAG               |
| D       | 5bit-enhanced-mixed             | 8-bit           | 5-bit     | 6-bit                             | Medium          | Highest (recommended) | Long-form writing, summarization, analysis    |

**Notes**
- "Selected layers" = first 1/8 + last 1/8 + every 3rd layer in the middle band.
- Only `v_proj` / `v_a_proj` / `v_b_proj` (value) and `down_proj` (FFN) receive the higher precision inside those layers.
- Variant D is the sweet spot for most serious use. The extra ~0.5 bit on the bulk gives a surprisingly large quality lift for modest memory increase.

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

- `--variant`: Quantization strategy (default: C). See table above.
- `--model`: Hugging Face repo ID or local path (default: Qwen/Qwen3.5-9B).
- `--num-layers`: Number of transformer layers. Required for variants C and D. Common values:
  - 9B / 8B class → 32
  - 32B class (Qwen3-32B etc.) → 64
  - Larger models → check `config.json` → `num_hidden_layers`
- `--trust-remote-code`: Pass through to transformers (needed for some custom Qwen variants).

The output directory name is derived automatically from the model name and chosen variant.

## Programmatic Use

All variants are simple callables compatible with `mlx_lm.convert(..., quant_predicate=...)`.

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

## How Layer Selection Works

The core helper `get_mixed_layer_predicate` implements a robust, model-architecture-aware heuristic:

1. Walks the module path string (e.g. `model.layers.23.mlp.down_proj`) to discover the layer index without hard-coded assumptions.
2. Handles both standard attention and Qwen3's hybrid Gated DeltaNet layers (`v_a_proj`, `v_b_proj`).
3. Applies elevated precision when the layer index falls in the protected bands.

This pattern (early + late + periodic) is cheap to compute and empirically effective. It does not require a calibration dataset.

## Adapting for Larger Models and Other Families

This repository was built with scale in mind.

**Qwen3 / Qwen3.5 family**
- 9B-class: `--num-layers 32`
- 32B-class: `--num-layers 64`
- Future 72B-class: check the config (typically 80–96 layers)

**Other model families**
The predicate logic is already fairly general for any Llama-style or Qwen-style decoder. For heavy customization:

- Fork `get_mixed_layer_predicate` and change the `use_high_bits` condition
- Protect additional tensors (`q_proj`, `k_proj`, `gate_proj`, `up_proj`, etc.)
- Use different bit allocations per band (early layers vs late layers)

**Future work ideas** (contributions welcome)
- Auto-detect `num_hidden_layers` + architecture from the HF config before conversion
- Support for Qwen2.5, Llama-3.1/4, Mistral, etc. with family-specific defaults
- Optional lightweight importance scoring using a tiny calibration set
- Integration with `mlx_lm` official recipes as composable building blocks

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
