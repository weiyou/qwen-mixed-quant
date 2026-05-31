#!/usr/bin/env python3
"""
Qwen Mixed-Bit Quantization for MLX  —  LEGACY v0 (ARCHIVED)

This is the original version of the tool, kept only for reproducibility.
It is the recipe that produced the `*-enhanced-mixed` model series
(e.g. Qwen3.5-9B-MLX-4bit-enhanced-mixed / -5bit-enhanced-mixed).

It predates the VLM-aware predicate in ../qwen_mixed_quant.py and treats
Qwen3.5-9B as a classic text model. Compared to the current tool it:
  - protects only classic projections (v_proj / down_proj), NOT the Qwen3.5
    linear-attention projections (in_proj_qkv / out_proj / in_proj_z/a/b)
  - does NOT keep the vision tower at higher precision (quantizes it at bulk bits)
  - gives the MTP head no special treatment
  - hardcodes num_layers=32 and the Qwen3.5-9B output names

For any new work use ../qwen_mixed_quant.py instead. For pure-text Qwen3/Qwen2.5
models the current tool's get_mixed_layer_predicate reproduces this behavior;
this file is needed only to regenerate the VLM `enhanced-mixed` baselines exactly.

Creates high-quality mixed-precision quantized versions of Qwen3 / Qwen3.5 models.

Variants focus on protecting the layers that matter most for generation quality
(especially embeddings, lm_head, and certain attention/FFN projections) while
aggressively quantizing the rest for speed.

Usage:
    python qwen_mixed_quant_v0.py --variant D --model Qwen/Qwen3.5-9B
"""

import argparse
from pathlib import Path
from typing import Callable

from mlx_lm import convert


def get_mixed_layer_predicate(
    num_layers: int | None = None,
    high_bits: int = 6,
    low_bits: int = 4,
    group_size: int = 64,
) -> Callable:
    """
    Returns a predicate that applies higher precision to v_proj and down_proj
    in selected layers (first 1/8, last 1/8, and every 3rd in the middle).

    This replicates the logic from mlx-lm's built-in mixed_4_6 recipe.
    """
    def predicate(path: str, layer) -> dict | bool:
        if not hasattr(layer, "to_quantized"):
            return False

        # Dynamically find the layer index location (more robust than hardcoding)
        # Example path: model.layers.7.self_attn.v_proj
        layer_idx = None
        parts = path.split(".")
        for i, part in enumerate(parts):
            if part.isdigit():
                # Heuristic: the first numeric part that is followed by 'self_attn' or 'mlp'
                # is almost always the layer index for Qwen-style models.
                if i + 1 < len(parts) and parts[i + 1] in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm"):
                    layer_idx = int(part)
                    break
                # Fallback: just take the first numeric segment
                if layer_idx is None:
                    layer_idx = int(part)

        if layer_idx is None:
            layer_idx = 0

        # Determine if this layer should get higher precision on key projections
        use_high_bits = (
            layer_idx < num_layers // 8
            or layer_idx >= 7 * num_layers // 8
            or (layer_idx - num_layers // 8) % 3 == 2
        ) if num_layers else False

        if use_high_bits and any(x in path for x in ["v_proj", "v_a_proj", "v_b_proj", "down_proj"]):
            return {"bits": high_bits, "group_size": group_size}

        return {"bits": low_bits, "group_size": group_size}

    return predicate


# ---------------------------------------------------------------------------
# Variant Predicates
# ---------------------------------------------------------------------------

def variant_a_fastest(path: str, layer) -> dict | bool:
    """Fastest practical mix: 4-bit bulk + 8-bit I/O."""
    if "embed_tokens" in path or "lm_head" in path:
        return {"bits": 8, "group_size": 64}
    if hasattr(layer, "to_quantized"):
        return {"bits": 4, "group_size": 64}
    return False


def variant_b_official_mixed():
    """Use mlx-lm's official mixed_4_6 recipe (string form)."""
    return "mixed_4_6"


def variant_c_good_quality(path: str, layer, num_layers: int = 32) -> dict | bool:
    """
    Enhanced mix (good balance of quality and speed).

    - 8-bit: embed_tokens + lm_head
    - 6-bit: v_proj + down_proj in selected layers
    - 4-bit: everything else
    """
    if "embed_tokens" in path or "lm_head" in path:
        return {"bits": 8, "group_size": 64}

    if not hasattr(layer, "to_quantized"):
        return False

    pred = get_mixed_layer_predicate(num_layers=num_layers, high_bits=6, low_bits=4)
    return pred(path, layer)


def variant_d_high_fidelity(path: str, layer, num_layers: int = 32) -> dict | bool:
    """
    Higher fidelity mix (recommended for summarization / long-form work).

    - 8-bit : embed_tokens + lm_head
    - 6-bit : v_proj + down_proj in selected layers (same heuristic as K_M style)
    - 5-bit : everything else

    Why not 7-bit on the selected layers?
    The jump from 6-bit → 7-bit on a subset of layers gives only marginal quality
    improvement for a noticeable increase in memory on those tensors. 5-bit bulk
    gives a much better speed/quality trade-off for most users.
    """
    if "embed_tokens" in path or "lm_head" in path:
        return {"bits": 8, "group_size": 64}

    if not hasattr(layer, "to_quantized"):
        return False

    # 6-bit on important projections, 5-bit on the rest
    pred = get_mixed_layer_predicate(num_layers=num_layers, high_bits=6, low_bits=5)
    return pred(path, layer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create mixed-bit quantized MLX versions of Qwen models."
    )
    parser.add_argument(
        "--variant",
        choices=["A", "B", "C", "D"],
        default="C",
        help="Quantization variant to use (default: C)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Hugging Face model ID or local path",
    )
    parser.add_argument(
        "--output-dir",
        default="./mlx_models",
        help="Directory to save the quantized model",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of transformer layers (32 for Qwen3.5-9B)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model",
    )

    args = parser.parse_args()

    variant = args.variant.upper()

    if variant == "A":
        predicate = variant_a_fastest
        name = "Qwen3.5-9B-MLX-4bit-8bit-embed"
        desc = "Variant A: 4-bit bulk + 8-bit embeds/lm_head (fastest)"

    elif variant == "B":
        predicate = variant_b_official_mixed()
        name = "Qwen3.5-9B-MLX-mixed-4-6"
        desc = "Variant B: Official mlx-lm mixed_4_6 recipe"

    elif variant == "C":
        predicate = lambda p, l: variant_c_good_quality(p, l, num_layers=args.num_layers)
        name = "Qwen3.5-9B-MLX-4bit-enhanced-mixed"
        desc = "Variant C: 8-bit I/O + 6-bit key projections + 4-bit bulk"

    else:  # D
        predicate = lambda p, l: variant_d_high_fidelity(p, l, num_layers=args.num_layers)
        name = "Qwen3.5-9B-MLX-5bit-enhanced-mixed"
        desc = "Variant D: 8-bit I/O + 6-bit key projections + 5-bit bulk (high fidelity)"

    out_path = Path(args.output_dir) / name

    print(f"[INFO] {desc}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Output: {out_path}")

    convert(
        hf_path=args.model,
        mlx_path=str(out_path),
        quantize=True,
        quant_predicate=predicate,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"\n[INFO] Done. Model saved to: {out_path}")
    print("[INFO] Test with:")
    print(f"  python -m mlx_lm.generate --model {out_path} --prompt 'Hello' --max-tokens 128")


if __name__ == "__main__":
    main()
