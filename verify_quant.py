#!/usr/bin/env python3
"""
Verification / audit tool for models produced by qwen_mixed_quant.py (or mlx_lm mixed quant).

Usage:
    python verify_quant.py /path/to/Qwen3.5-9B-MLX-Q5_K6_L
    python verify_quant.py /path/to/model1 /path/to/model2 --detailed

It inspects:
- The detailed per-module quantization map in config.json
- Actual dtypes in safetensors (lm_head, embed_tokens, sample layers) when possible
- Reports whether I/O tensors (embed + lm_head) received the intended high precision
- Summarizes bit distribution and tries to identify the recipe/variant

Exit code 0 if everything looks healthy for the detected style; non-zero on clear problems.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx  # type: ignore
    HAS_MLX = True
except Exception:
    HAS_MLX = False

try:
    from safetensors import safe_open  # type: ignore
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False


def load_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_dir}")
    return json.loads(cfg_path.read_text())


def load_index(model_dir: Path) -> dict[str, Any]:
    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"No model.safetensors.index.json found in {model_dir}")
    return json.loads(idx_path.read_text())


def analyze_quantization_map(qcfg: dict[str, Any]) -> dict[str, Any]:
    """Analyze the 'quantization' section of config.json."""
    if not qcfg:
        return {"error": "no quantization section"}

    top_bits = qcfg.get("bits")
    top_gs = qcfg.get("group_size")
    mode = qcfg.get("mode")

    per_module: list[tuple[str, int, int]] = []
    bit_counts: Counter[int] = Counter()
    special: dict[str, dict[str, Any]] = {}

    for key, val in qcfg.items():
        if not isinstance(val, dict) or "bits" not in val:
            continue
        bits = int(val["bits"])
        gs = int(val.get("group_size", 0))
        per_module.append((key, bits, gs))
        bit_counts[bits] += 1

        # Track interesting tensors
        klower = key.lower()
        if "embed_tokens" in klower:
            special["embed_tokens"] = {"bits": bits, "group_size": gs, "path": key}
        if "lm_head" in klower:
            special["lm_head"] = {"bits": bits, "group_size": gs, "path": key}
        if any(v in klower for v in ("visual", "vision", "image_newline")):
            special.setdefault("vision", []).append({"path": key, "bits": bits})
        if "mtp" in klower:
            special.setdefault("mtp", []).append({"path": key, "bits": bits})

    # Compute rough "dominant" bits
    dominant = bit_counts.most_common(1)[0][0] if bit_counts else None

    return {
        "top_level": {"bits": top_bits, "group_size": top_gs, "mode": mode},
        "bit_counts": dict(bit_counts),
        "num_quantized_modules": len(per_module),
        "dominant_bits": dominant,
        "special_tensors": special,
        "has_per_module_overrides": len(bit_counts) > 1 or (dominant != top_bits),
    }


def find_key_in_index(idx: dict[str, Any], substrings: list[str]) -> str | None:
    for k in idx.get("weight_map", {}):
        if any(s in k.lower() for s in substrings):
            return k
    return None


def inspect_tensor_dtypes(model_dir: Path, idx: dict[str, Any]) -> dict[str, Any]:
    """Use MLX (preferred) or safetensors to look at actual stored dtypes for key tensors."""
    results: dict[str, Any] = {}
    if not (HAS_MLX or HAS_SAFETENSORS):
        results["note"] = "Neither mlx nor safetensors available for deep dtype inspection"
        return results

    weight_map = idx.get("weight_map", {})

    targets = {
        "lm_head": ["lm_head.weight", "lm_head"],
        "embed_tokens": ["embed_tokens.weight", "embed_tokens"],
        "sample_layer_0_down": ["layers.0.mlp.down_proj.weight", "layers.0.mlp.down_proj"],
        "sample_layer_31_v": ["layers.31.self_attn.v_proj.weight", "layers.31.self_attn.v_proj"],
    }

    for name, substrs in targets.items():
        key = find_key_in_index(idx, substrs)
        if not key:
            results[name] = {"found": False}
            continue

        shard = weight_map[key]
        shard_path = model_dir / shard
        if not shard_path.exists():
            results[name] = {"found": True, "shard": shard, "error": "shard missing"}
            continue

        try:
            if HAS_MLX:
                weights = mx.load(str(shard_path))
                # Find the actual tensor(s) for this logical key
                matches = {k: str(v.dtype) for k, v in weights.items() if name.split("_")[0] in k.lower() or any(s in k.lower() for s in substrs)}
                if not matches:
                    # fallback: any key containing the last segment
                    last = substrs[0].split(".")[-2] if "." in substrs[0] else substrs[0]
                    matches = {k: str(v.dtype) for k, v in weights.items() if last in k.lower()}
                results[name] = {"found": True, "shard": shard, "tensors": matches or {"note": "no matching tensors in shard"}}
            else:
                # safetensors fallback (numpy dtypes)
                with safe_open(str(shard_path), framework="numpy") as f:  # type: ignore
                    keys = [k for k in f.keys() if any(s in k.lower() for s in substrs)]
                    dtypes = {}
                    for k in keys:
                        t = f.get_tensor(k)
                        dtypes[k] = str(t.dtype)
                    results[name] = {"found": True, "shard": shard, "tensors": dtypes or {"note": "not present as separate tensor"}}
        except Exception as e:
            results[name] = {"found": True, "shard": shard, "error": str(e)}

    return results


def guess_variant(analysis: dict[str, Any]) -> str:
    bc = analysis.get("bit_counts", {})
    special = analysis.get("special_tensors", {})
    embed = special.get("embed_tokens", {})
    lm = special.get("lm_head", {})

    has_8 = 8 in bc
    has_6 = 6 in bc
    has_5 = 5 in bc
    has_4 = 4 in bc

    embed_8 = embed.get("bits") == 8
    lm_8 = lm.get("bits") == 8
    lm_quantized = "lm_head" in special and lm.get("bits", 0) > 0

    if bc == {6: bc.get(6, 0)} or (len(bc) == 1 and 6 in bc):
        return "E2 (uniform 6-bit)"
    if has_8 and has_6 and not has_5 and not has_4:
        if embed_8 and (not lm_quantized or lm_8):
            return "E (Q6_K_L style)"
        return "uniform-6 + 8-bit I/O (partial)"
    if has_8 and has_6 and has_5:
        return "D (Q5_K6_L) or similar 5+6+8 mix"
    if has_8 and has_6 and has_4:
        return "C (Q4_K6_L) or similar 4+6+8 mix"
    if has_8 and has_4 and not has_6:
        return "A (Q4_K_L) style"
    if has_6 and has_4:
        return "mixed 4/6 (possibly mlx-lm built-in)"
    return "custom / unknown mixed"


def verify_model(model_dir: Path, detailed: bool = False) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(str(model_dir))

    cfg = load_config(model_dir)
    idx = load_index(model_dir)

    qcfg = cfg.get("quantization", {})
    analysis = analyze_quantization_map(qcfg)
    variant_guess = guess_variant(analysis)

    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "variant_guess": variant_guess,
        "quantization_analysis": analysis,
        "issues": [],
        "ok": True,
    }

    special = analysis.get("special_tensors", {})
    embed = special.get("embed_tokens")
    lm = special.get("lm_head")

    # Core expectations for the recipes in this repo
    if embed is None:
        report["issues"].append("embed_tokens not found in quantization map (unexpected for _L variants)")
    elif embed.get("bits") != 8:
        report["issues"].append(f"embed_tokens is {embed.get('bits')}-bit (expected 8-bit for _L variants)")

    if lm is None:
        # This is the historic bug we fixed
        report["issues"].append("lm_head NOT present in quantization map (stayed bfloat16) — this was a known issue before the is_embed_or_head fix")
        report["ok"] = False
    else:
        if lm.get("bits") != 8:
            report["issues"].append(f"lm_head is {lm.get('bits')}-bit (expected 8-bit for _L / high-fidelity variants)")
            report["ok"] = False

    if detailed:
        report["dtype_inspection"] = inspect_tensor_dtypes(model_dir, idx)

        # Cross-check: if the .weight tensor for a supposedly-quantized lm_head/embed is still full precision
        dtype_info = report.get("dtype_inspection", {})
        lm_dtype = dtype_info.get("lm_head", {})
        if lm and isinstance(lm, dict) and lm.get("bits", 0) >= 4:
            if "tensors" in lm_dtype:
                for tname, dt in lm_dtype.get("tensors", {}).items():
                    if "weight" in tname.lower() and ("bfloat16" in dt.lower() or "float16" in dt.lower()):
                        report["issues"].append(f"lm_head .weight on disk is {dt} (full precision) while config says it should be quantized to {lm.get('bits')}-bit")
                        report["ok"] = False

    report["summary"] = {
        "bit_distribution": analysis.get("bit_counts"),
        "embed_bits": embed.get("bits") if embed else None,
        "lm_head_bits": lm.get("bits") if lm else None,
        "num_modules_quantized": analysis.get("num_quantized_modules"),
    }

    return report


def print_report(report: dict[str, Any], verbose: bool = False) -> None:
    print(f"\n=== Quantization Audit: {report['model_dir']} ===")
    print(f"Guessed variant style: {report['variant_guess']}")
    print(f"Overall status: {'OK' if report['ok'] else 'ISSUES FOUND'}")

    summ = report["summary"]
    print(f"\nBit distribution: {summ['bit_distribution']}")
    print(f"embed_tokens : {summ['embed_bits']}-bit")
    print(f"lm_head      : {summ['lm_head_bits']}-bit (None = unquantized / bf16)")

    if report["issues"]:
        print("\nIssues / warnings:")
        for iss in report["issues"]:
            print(f"  - {iss}")
    else:
        print("\nNo issues detected for the expected recipe.")

    if verbose and "dtype_inspection" in report:
        print("\n--- Deep dtype inspection (from safetensors) ---")
        for name, info in report["dtype_inspection"].items():
            print(f"  {name}: {info}")

    if verbose:
        qa = report["quantization_analysis"]
        print(f"\nTop-level in config: {qa.get('top_level')}")
        print(f"Has per-module overrides: {qa.get('has_per_module_overrides')}")


def main():
    parser = argparse.ArgumentParser(description="Audit MLX quantized models produced by qwen_mixed_quant or similar.")
    parser.add_argument("model_dirs", nargs="+", help="One or more paths to quantized model directories")
    parser.add_argument("--detailed", "-d", action="store_true", help="Also inspect actual tensor dtypes from safetensors (requires mlx or safetensors)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON instead of human text")
    args = parser.parse_args()

    all_ok = True
    outputs = []

    for raw in args.model_dirs:
        p = Path(raw)
        try:
            rep = verify_model(p, detailed=args.detailed)
            outputs.append(rep)
            if not rep["ok"]:
                all_ok = False
            if not args.json:
                print_report(rep, verbose=args.detailed)
        except Exception as e:
            print(f"ERROR processing {p}: {e}", file=sys.stderr)
            all_ok = False
            outputs.append({"model_dir": str(p), "error": str(e)})

    if args.json:
        print(json.dumps(outputs if len(outputs) > 1 else outputs[0], indent=2))

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
