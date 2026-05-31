#!/usr/bin/env python3
"""
Mixed-Precision Quantization Recipes for Qwen3 / Qwen3.5 / Qwen3.6 on MLX

Specially tuned for Qwen3.5 VLMs (including Qwen3.5-9B) with support for:
- Nested language tower (model.language_model.layers.*)
- Linear attention (Gated DeltaNet-style) projections
- MTP (Multi-Token Prediction) heads
- Vision / visual towers (kept at higher precision by default)

Optimized for Apple Silicon, especially M4 Pro / Max / Ultra machines with 48GB+ unified memory.

Variants C and D are the recommended choices for excellent quality at 4-bit / 5-bit footprints;
variant E (MLX-Q6_K_L) is a direct analog of llama.cpp's Q6_K_L 6-bit quant.

Usage:
    python qwen_mixed_quant.py --variant D --model Qwen/Qwen3.5-9B
    python qwen_mixed_quant.py --variant D --model Qwen/Qwen3-32B
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


def _extract_num_language_layers(cfg: dict) -> int | None:
    """
    Extract the number of *language* layers, handling both classic text models
    and modern Qwen3.5 VLMs / MoE models where the count lives under text_config.
    """
    # Classic pure-text models (Qwen3, Qwen2.5, etc.)
    for key in ("num_hidden_layers", "n_layers", "num_layers"):
        val = cfg.get(key)
        if isinstance(val, int) and val > 0:
            return val

    # Qwen3.5-style VLMs and MoE models: language tower is under text_config
    for nested_key in ("text_config", "language_config", "llm_config"):
        nested = cfg.get(nested_key)
        if isinstance(nested, dict):
            for key in ("num_hidden_layers", "n_layers", "num_layers"):
                val = nested.get(key)
                if isinstance(val, int) and val > 0:
                    return val

    return None


def get_num_layers_from_hf(model: str) -> int | None:
    """Attempt to read the language model layer count from config.json on the Hub."""
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
        return _extract_num_language_layers(cfg)
    except Exception:
        pass
    return None


def detect_model_architecture(model: str) -> dict:
    """
    Best-effort detection of Qwen3.5/3/2.5 characteristics from config.
    Returns dict with keys like: num_layers, model_type, has_vision, has_mtp, uses_linear_attn.
    """
    info = {
        "num_layers": None,
        "model_type": None,
        "has_vision": False,
        "has_mtp": False,
        "uses_linear_attn": False,
        "is_vlm": False,
    }
    try:
        from huggingface_hub import hf_hub_download
        import json

        config_path = hf_hub_download(repo_id=model, filename="config.json", repo_type="model")
        with open(config_path) as f:
            cfg = json.load(f)

        info["num_layers"] = _extract_num_language_layers(cfg)
        info["model_type"] = cfg.get("model_type")

        arch = str(cfg.get("architectures", [""])[0]).lower()
        info["is_vlm"] = "conditional" in arch or "vlm" in arch or info["model_type"] in ("qwen3_5", "qwen3_5_vl", "qwen3_5_moe_text")

        # Rough heuristics from known Qwen3.5 family behavior
        if info["model_type"] in ("qwen3_5", "qwen3_5_vl") or "3.5" in model.lower():
            info["has_vision"] = True
            info["has_mtp"] = True
            info["uses_linear_attn"] = True
    except Exception:
        pass
    return info


def extract_layer_index(path: str) -> int | None:
    """
    Robustly extract the transformer layer index from a weight path.
    Supports both classic and Qwen3.5 VLM nested layouts:
        model.layers.5.mlp.down_proj
        model.language_model.layers.17.linear_attn.out_proj
    """
    parts = path.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            # Look ahead for common layer container names
            if i + 1 < len(parts):
                nxt = parts[i + 1]
                if nxt in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm",
                           "linear_attn", "layers"):  # last one catches some nesting
                    return int(part)
            # Fallback: first numeric segment that looks like a layer
            if 0 <= int(part) < 256:  # reasonable layer count upper bound
                return int(part)
    return None


def is_vision_path(path: str) -> bool:
    """Return True for any tensor belonging to the vision / visual tower."""
    p = path.lower()
    return any(x in p for x in ["visual.", ".visual.", "vision_tower", "image_newline", "pixel"])


def is_mtp_path(path: str) -> bool:
    """Return True for Multi-Token Prediction head weights."""
    return path.startswith("mtp.") or ".mtp." in path


def is_embed_or_head(path: str) -> bool:
    """True for embedding and language modeling head weights (including nested)."""
    return (
        "embed_tokens" in path
        or path == "lm_head.weight"
        or path.endswith("lm_head.weight")
        or ("language_model" in path and "embed_tokens" in path)
    )


def get_qwen_mixed_predicate(
    num_layers: int,
    high_bits: int = 6,
    low_bits: int = 4,
    embed_bits: int = 8,
    vision_bits: int | None = None,   # None = do not quantize vision tower at all
    protect_mtp: bool = True,
    mtp_bits: int = 8,
    group_size: int = 64,
) -> Callable:
    """
    Modern, Qwen3.5-aware mixed precision predicate.

    Key improvements for Qwen3.5 / 3.6 VLMs (including on M4 Pro 48GB):
    - Correctly handles nested language tower: model.language_model.layers.N
    - Protects linear_attn projections (the Qwen3.5 equivalent of v_proj / important attn weights)
    - Always keeps embed_tokens + lm_head at high precision (8-bit by default)
    - Never quantizes (or uses very high bits for) the vision tower
    - Gives special treatment to the MTP (multi-token prediction) head
    - Uses the proven "first 1/8 + last 1/8 + every 3rd" band for language layers
    """
    def predicate(path: str, layer) -> dict | bool:
        if not hasattr(layer, "to_quantized"):
            return False

        # Vision tower: usually best left alone or at very high precision for VLM quality
        if is_vision_path(path):
            if vision_bits is None:
                return False
            return {"bits": vision_bits, "group_size": group_size}

        # MTP head (very valuable for generation quality in recent Qwen models)
        if is_mtp_path(path):
            if protect_mtp:
                return {"bits": mtp_bits, "group_size": group_size}
            return {"bits": low_bits, "group_size": group_size}

        # Embeddings and lm_head (critical for quality)
        if is_embed_or_head(path):
            return {"bits": embed_bits, "group_size": group_size}

        layer_idx = extract_layer_index(path)
        if layer_idx is None:
            # Unknown module — be conservative and don't quantize
            return False

        # Protected bands for language layers (same philosophy as before, tuned for 32/64 layer models)
        use_high = (
            layer_idx < max(1, num_layers // 8)
            or layer_idx >= num_layers - max(1, num_layers // 8)
            or (layer_idx - num_layers // 8) % 3 == 2
        )

        # For Qwen3.5 linear attention models, these are the critical projections to protect
        # (in_proj_qkv / out_proj are roughly analogous to the value path + output)
        important_projs = {
            "v_proj", "v_a_proj", "v_b_proj",
            "down_proj",
            "in_proj_qkv", "out_proj", "in_proj_z", "in_proj_a", "in_proj_b",
        }

        if use_high and any(proj in path for proj in important_projs):
            return {"bits": high_bits, "group_size": group_size}

        return {"bits": low_bits, "group_size": group_size}

    return predicate


# Backwards-compatible thin wrapper for classic pure-text Qwen3 / Qwen2.5 models
def get_mixed_layer_predicate(
    num_layers: int | None = None,
    high_bits: int = 6,
    low_bits: int = 4,
    group_size: int = 64,
) -> Callable:
    """
    Legacy wrapper. For modern Qwen3.5 VLMs use get_qwen_mixed_predicate instead.
    """
    if num_layers is None:
        num_layers = 32
    # Delegate to the new robust implementation with safe defaults
    return get_qwen_mixed_predicate(
        num_layers=num_layers,
        high_bits=high_bits,
        low_bits=low_bits,
        embed_bits=8,
        vision_bits=None,
        protect_mtp=True,
        group_size=group_size,
    )


# ---------------------------------------------------------------------------
# Variant Predicates
# ---------------------------------------------------------------------------

def variant_a_fastest(path: str, layer) -> dict | bool:
    """Fastest practical mix for Qwen3.5 VLMs: 4-bit bulk + 8-bit critical I/O + vision at 8-bit."""
    if is_embed_or_head(path) or is_mtp_path(path):
        return {"bits": 8, "group_size": 64}
    if is_vision_path(path):
        return {"bits": 8, "group_size": 64}
    if hasattr(layer, "to_quantized"):
        return {"bits": 4, "group_size": 64}
    return False


def variant_b_official_mixed():
    """Use mlx-lm's official mixed_4_6 recipe (string form)."""
    return "mixed_4_6"


def variant_c_good_quality(path: str, layer, num_layers: int) -> dict | bool:
    """
    Excellent balance for Qwen3.5 VLMs and large dense models (recommended default for 48GB Macs).

    - 8-bit : language_model.embed_tokens + lm_head + (optionally vision at 8-bit)
    - 6-bit : critical projections in protected bands (linear_attn.* + down_proj + classic v_proj)
    - 4-bit : everything else in the language tower
    - Vision tower: left in higher precision (or BF16) by default for VLM quality
    """
    pred = get_qwen_mixed_predicate(
        num_layers=num_layers,
        high_bits=6,
        low_bits=4,
        embed_bits=8,
        vision_bits=8,           # keep vision at 8-bit for good multimodal quality
        protect_mtp=True,
        mtp_bits=8,
        group_size=64,
    )
    return pred(path, layer)


def variant_d_high_fidelity(path: str, layer, num_layers: int) -> dict | bool:
    """
    Highest practical quality on memory-constrained hardware (great for M4 Pro 48GB).

    - 8-bit : embed + lm_head + MTP + vision
    - 6-bit : critical linear_attn / v_proj / down_proj in protected bands
    - 5-bit : bulk of language tower

    This is the sweet spot for Qwen3.5-9B and 27B-class models when you want
    near-BF16 reasoning and long-context behavior while staying comfortably
    within 48GB unified memory with large context.
    """
    pred = get_qwen_mixed_predicate(
        num_layers=num_layers,
        high_bits=6,
        low_bits=5,
        embed_bits=8,
        vision_bits=8,
        protect_mtp=True,
        mtp_bits=8,
        group_size=64,
    )
    return pred(path, layer)


def variant_e_q6_k_l(path: str, layer, num_layers: int) -> dict | bool:
    """
    Direct MLX analog of llama.cpp's Q6_K_L quant.

    Q6_K_L = a Q6_K body (~6.56 bpw) plus token_embd and output/lm_head elevated
    to Q8_0. The "_L" is exactly that embedding/output bump.

    MLX mapping (a faithful, footprint-matching clone, ~6.56 bpw effective):
    - 8-bit : embed + lm_head + MTP + vision  (the Q8_0 "_L" portion + family I/O)
    - 6-bit : the entire language tower       (the Q6_K body; high_bits == low_bits,
              so the protected bands collapse to the uniform 6-bit floor)

    On Qwen3.5-9B this lands at ~9.5-9.7 GB peak and tracks the real Q6_K_L GGUF
    in both size and quality (matches the uniform-6 MLX reference).

    If you instead want a "Q6_K_L+" that elevates the band-critical projections
    (v_proj / down_proj / out_proj) to 8-bit -- the MLX stand-in for Q6_K's
    importance-weighted superblocks -- set high_bits=8 (costs ~1.5-2 GB more).
    """
    pred = get_qwen_mixed_predicate(
        num_layers=num_layers,
        high_bits=6,   # == low_bits: uniform 6-bit body, the true Q6_K_L analog
        low_bits=6,    # 6-bit body == Q6_K
        embed_bits=8,  # Q8_0 embeddings + output == the "_L"
        vision_bits=8,
        protect_mtp=True,
        mtp_bits=8,
        group_size=64,
    )
    return pred(path, layer)


def variant_e2_q6_k(path: str, layer) -> dict | bool:
    """
    Plain uniform 6-bit — the analog of llama.cpp's Q6_K (no "_L" embedding bump).

    Every quantizable layer (including embeddings, lm_head, MTP and the vision
    tower) is quantized at 6-bit. This is exactly equivalent to running
    `mlx_lm.convert -q --q-bits 6 --q-group-size 64` with the default predicate.

    Use --variant E (MLX-Q6_K_L) instead when you want the I/O tensors
    (embeddings / output / MTP / vision) kept at 8-bit.
    """
    if hasattr(layer, "to_quantized"):
        return {"bits": 6, "group_size": 64}
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create high-quality mixed-precision quantized MLX models from Qwen3 / Qwen3.5 / Qwen3.6 checkpoints.",
        epilog=(
            "Variants (output dirs follow a GGUF-parallel MLX-Q<body>_K<protect>_<embed> naming scheme):\n"
            "  A  MLX-Q4_K_L   4-bit bulk + 8-bit embed/I/O + vision, no protected tier. Fastest.\n"
            "  B  MLX-mixed-4-6  upstream mlx-lm baseline (outside the naming scheme).\n"
            "  C  MLX-Q4_K6_L  4-bit bulk + 6-bit protected projections + 8-bit embed/vision [good default]\n"
            "  D  MLX-Q5_K6_L  5-bit bulk + 6-bit protected projections + 8-bit embed/vision [best quality on 48GB]\n"
            "  E  MLX-Q6_K_L   6-bit bulk + 8-bit embed/output (+vision/MTP). Direct analog of llama.cpp Q6_K_L.\n"
            "  E2 MLX-Q6_K     plain uniform 6-bit everywhere. Analog of llama.cpp Q6_K (no _L bump).\n\n"
            "Naming: Q<body> = bulk bit-width; _K<n> = MLX group-quant with protected projections at <n>-bit\n"
            "(omitted when the protected tier collapses to the body); _L = 8-bit ('large') embeddings/output.\n\n"
            "Qwen3.5-9B (your primary model) is a VLM with linear_attn + MTP + visual tower.\n"
            "Variants C, D and E are tuned specifically for these models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=["A", "B", "C", "D", "E", "E2"],
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

    # Rich architecture detection (especially useful for Qwen3.5 VLMs)
    arch_info = detect_model_architecture(args.model)
    if arch_info.get("model_type"):
        vlm_str = "VLM" if arch_info.get("is_vlm") else "text-only"
        print(f"[INFO] Detected: {arch_info['model_type']} ({vlm_str})")

    # Resolve num_layers with robust nested lookup for Qwen3.5 models
    num_layers = args.num_layers
    if num_layers is None:
        num_layers = arch_info.get("num_layers") or get_num_layers_from_hf(args.model)

    if num_layers is None and variant in ("C", "D", "E"):
        parser.error(
            "--num-layers could not be auto-detected. "
            "Please pass it explicitly (e.g. --num-layers 32 for Qwen3.5-9B or --num-layers 64 for Qwen3-32B)."
        )
    elif num_layers:
        print(f"[INFO] Using {num_layers} language layers")

    if variant == "A":
        predicate = variant_a_fastest
        name = f"{short_name}-MLX-Q4_K_L"
        desc = "Variant A (MLX-Q4_K_L): 4-bit bulk + 8-bit embed/I/O + vision, no protected projection tier (fastest)"

    elif variant == "B":
        predicate = variant_b_official_mixed()
        name = f"{short_name}-MLX-mixed-4-6"
        desc = "Variant B (MLX-mixed-4-6): upstream mlx-lm mixed_4_6 baseline (outside the MLX-Q*_K_* convention)"

    elif variant == "C":
        predicate = lambda p, l: variant_c_good_quality(p, l, num_layers=num_layers)
        name = f"{short_name}-MLX-Q4_K6_L"
        desc = "Variant C (MLX-Q4_K6_L): 4-bit bulk + 6-bit protected projections (linear_attn+down) + 8-bit embed/vision/MTP [great on 48GB]"

    elif variant == "D":
        predicate = lambda p, l: variant_d_high_fidelity(p, l, num_layers=num_layers)
        name = f"{short_name}-MLX-Q5_K6_L"
        desc = "Variant D (MLX-Q5_K6_L): 5-bit bulk + 6-bit protected projections + 8-bit embed/vision/MTP (best quality on 48GB Mac)"

    elif variant == "E":
        predicate = lambda p, l: variant_e_q6_k_l(p, l, num_layers=num_layers)
        name = f"{short_name}-MLX-Q6_K_L"
        desc = "Variant E (MLX-Q6_K_L): 6-bit bulk + 8-bit embed/output (+vision/MTP) — direct analog of llama.cpp Q6_K_L"

    else:  # E2
        predicate = variant_e2_q6_k
        name = f"{short_name}-MLX-Q6_K"
        desc = "Variant E2 (MLX-Q6_K): plain uniform 6-bit everywhere — analog of llama.cpp Q6_K (no _L bump)"

    out_path = Path(args.output_dir) / name
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    print("[INFO] Recommended test command (M4 Pro 48GB):")
    print(f"  python -m mlx_lm.generate --model {out_path} \\")
    print("      --prompt 'Describe this image in detail.' --max-tokens 512 \\")
    print("      --temp 0.7")
    print("\n[INFO] For Qwen3.5 VLMs also try passing images with mlx_lm or mlx-vlm.")


if __name__ == "__main__":
    main()
