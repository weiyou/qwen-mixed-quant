#!/usr/bin/env python3
"""
Mixed-Precision Quantization Recipes for Qwen3 / Qwen3.5 on MLX

Creates high-quality 4-bit and 5-bit quantized MLX models by strategically
protecting the weights that matter most (embeddings, lm_head, value projections,
and FFN down projections in early / late / periodic layers).

This approach delivers significantly better quality than uniform low-bit
quantization for the same memory budget — and the benefit grows with model size.

See README.md for the full rationale, variant comparison table, and guidance
for scaling to 32B / 72B+ class models.

Usage:
    python qwen_mixed_quant.py --variant D --model Qwen/Qwen3.5-9B
    python qwen_mixed_quant.py --variant D --model Qwen/Qwen3-32B --num-layers 64
"""

import argparse
from pathlib import Path
from typing import Callable

from mlx_lm import convert

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_model_name(model: str) -> str:
    """Convert a HF repo ID or path into a clean directory-friendly name."""
    name = model.rstrip("/\\")
    if "/" in name:
        name = name.split("/")[-1]
    # Remove or replace characters that are awkward in paths
    for ch in [":", " ", "@", "#", "$", "%", "&", "*", "?", "<", ">", "|", '"', "'"]:
        name = name.replace(ch, "-")
    name = name.replace("..", ".")
    return name or "model"


def get_num_layers_from_hf(model: str) -> int | None:
    """Attempt to read num_hidden_layers from the model's config.json on the Hub."""
    try:
        from huggingface_hub import hf_hub_download
        import json

        config_path = hf_hub_download(
            repo_id=model,
            filename="config.json",
            repo_type="model",
            local_files_only=False,
        )
        with open(config_path, "r") as f:
            cfg = json.load(f)
        n = cfg.get("num_hidden_layers") or cfg.get("n_layers") or cfg.get("num_layers")
        if isinstance(n, int) and n > 0:
            return n
    except Exception:
        pass
    return None


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


def variant_c_good_quality(path: str, layer, num_layers: int) -> dict | bool:
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


def variant_d_high_fidelity(path: str, layer, num_layers: int) -> dict | bool:
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
        description="Create high-quality mixed-precision quantized MLX models from Qwen checkpoints.",
        epilog=(
            "Variants:\n"
            "  A  Fastest practical (4-bit bulk + 8-bit I/O)\n"
            "  B  Official mlx-lm mixed_4_6 recipe\n"
            "  C  Excellent balance (4-bit bulk + 6-bit key projections + 8-bit I/O)\n"
            "  D  Highest fidelity (5-bit bulk + 6-bit key projections + 8-bit I/O)  [recommended]\n\n"
            "For 32B-class models use --num-layers 64 (or let auto-detection handle it)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=["A", "B", "C", "D"],
        default="D",
        help="Quantization variant (default: D = highest fidelity)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Hugging Face model ID or local path (default: Qwen/Qwen3.5-9B)",
    )
    parser.add_argument(
        "--output-dir",
        default="./mlx_models",
        help="Base directory for the quantized model (default: ./mlx_models)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of transformer layers. Auto-detected from config.json when omitted.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom modeling code from the model repo",
    )

    args = parser.parse_args()

    variant = args.variant.upper()
    short_name = sanitize_model_name(args.model)

    # Resolve num_layers (needed by C and D)
    num_layers = args.num_layers
    if num_layers is None and variant in ("C", "D"):
        num_layers = get_num_layers_from_hf(args.model)
        if num_layers:
            print(f"[INFO] Auto-detected {num_layers} layers from model config")
        else:
            parser.error(
                "--num-layers is required for variants C and D when auto-detection fails. "
                "Pass e.g. --num-layers 32 for 9B-class or --num-layers 64 for 32B-class models."
            )

    if variant == "A":
        predicate = variant_a_fastest
        name = f"{short_name}-MLX-4bit-8bit-embed"
        desc = "Variant A: 4-bit bulk + 8-bit embeds/lm_head (fastest)"

    elif variant == "B":
        predicate = variant_b_official_mixed()
        name = f"{short_name}-MLX-mixed-4-6"
        desc = "Variant B: Official mlx-lm mixed_4_6 recipe"

    elif variant == "C":
        predicate = lambda p, l: variant_c_good_quality(p, l, num_layers=num_layers)
        name = f"{short_name}-MLX-4bit-enhanced-mixed"
        desc = "Variant C: 8-bit I/O + 6-bit key projections + 4-bit bulk"

    else:  # D
        predicate = lambda p, l: variant_d_high_fidelity(p, l, num_layers=num_layers)
        name = f"{short_name}-MLX-5bit-enhanced-mixed"
        desc = "Variant D: 8-bit I/O + 6-bit key projections + 5-bit bulk (high fidelity)"

    out_path = Path(args.output_dir) / name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] {desc}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Output: {out_path}")
    if num_layers:
        print(f"[INFO] Using layer count: {num_layers}")

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
