"""GGUF export and bundle assembly (Step F) for the continual-counsel offline pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def merge_to_fp16(
    adapter_dir: str,
    base_model_id: str,
    output_dir: str,
    base_cfg: Any,
) -> Path:
    """Merge a LoRA adapter into the base model weights at fp16 precision.

    Saves a full HuggingFace model to output_dir. The merged model is suitable
    for GGUF conversion and for offline deployment without PEFT.

    base_cfg is used to resolve the model_id if base_model_id is a profile alias.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    log.info("Loading base model %s for fp16 merge.", base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU merge avoids GPU memory pressure during export
        trust_remote_code=True,
    )

    log.info("Loading adapter from %s.", adapter_dir)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    log.info("Merging adapter weights into base model.")
    merged_model = peft_model.merge_and_unload()
    merged_model = merged_model.half()  # ensure fp16 throughout

    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    log.info("Merged fp16 model saved to %s.", output_dir)
    return output_dir


def convert_to_gguf(
    merged_model_dir: str,
    llama_cpp_dir: str,
    output_path: str,
    quant_level: str,
) -> Path:
    """Convert a HuggingFace model to GGUF format using llama.cpp tooling.

    Runs two subprocesses:
        1. python convert_hf_to_gguf.py <merged_model_dir> --outfile <tmp_f16.gguf>
        2. llama-quantize <tmp_f16.gguf> <output_path> <quant_level>

    Raises RuntimeError with the subprocess stderr if either step fails.
    """
    llama_cpp_dir = Path(llama_cpp_dir)
    merged_model_dir = Path(merged_model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp_dir / "llama-quantize"

    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}")
    if not quantize_bin.exists():
        # Also try without extension and with .exe (Windows).
        quantize_bin_exe = llama_cpp_dir / "llama-quantize.exe"
        if not quantize_bin_exe.exists():
            raise FileNotFoundError(
                f"llama-quantize not found at {quantize_bin}. "
                "Build llama.cpp first (cmake --build build --target llama-quantize)."
            )
        quantize_bin = quantize_bin_exe

    # Step 1: HF -> f16 GGUF.
    tmp_gguf = output_path.parent / (output_path.stem + "_f16.gguf")
    cmd_convert = [
        sys.executable,
        str(convert_script),
        str(merged_model_dir),
        "--outfile", str(tmp_gguf),
        "--outtype", "f16",
    ]
    log.info("Running convert_hf_to_gguf: %s", " ".join(cmd_convert))
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed (exit {result.returncode}):\n{result.stderr}"
        )

    # Step 2: quantise to target quant_level.
    cmd_quant = [str(quantize_bin), str(tmp_gguf), str(output_path), quant_level]
    log.info("Running llama-quantize: %s", " ".join(cmd_quant))
    result = subprocess.run(cmd_quant, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed (exit {result.returncode}):\n{result.stderr}"
        )

    # Clean up the intermediate f16 file.
    if tmp_gguf.exists():
        tmp_gguf.unlink()

    log.info("GGUF export complete: %s", output_path)
    return output_path


def write_manifest(bundle_dir: str, commit_hash: str, python_version: str) -> Path:
    """Compute SHA-256 and byte size for every file in bundle_dir; write manifest.

    Writes two files:
        manifest.json  -- structured JSON with file metadata
        manifest.sha256 -- SHA-256 of the manifest.json content

    Manifest JSON schema:
        {
            "regime": ...,
            "timestamp": ...,
            "commit_hash": ...,
            "python_version": ...,
            "files": [{"path": ..., "sha256": ..., "size_bytes": ...}]
        }

    The "regime" and "timestamp" fields are inferred from the bundle_dir name
    (expected format: <regime>_<timestamp>).
    """
    bundle_dir = Path(bundle_dir)
    dir_name = bundle_dir.name
    # Best-effort parse of regime and timestamp from directory name.
    parts = dir_name.rsplit("_", 1)
    regime = parts[0] if len(parts) == 2 else dir_name
    timestamp = parts[1] if len(parts) == 2 else datetime.now(timezone.utc).isoformat()

    metadata = {
        "regime": regime,
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "python_version": python_version,
    }

    metadata_path = bundle_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log.info("Metadata written to %s", metadata_path)
    return metadata_path


def assemble_bundle(
    regime: str,
    timestamp_str: str,
    adapter_dir: str,
    gguf_path: str,
    basis_checkpoint: str,
    golden_set_dir: str,
    benchmark_results: dict,
    commit_hash: str,
    python_version: str,
    exports_root: str,
) -> Path:
    """Copy all training artifacts into a versioned export bundle directory.

    Bundle layout:
        exports/<regime>_<timestamp>/
            adapter/              <- LoRA adapter weights and config
            model.gguf            <- quantised GGUF file
            basis_checkpoint/     <- OrthonormalBasis checkpoint (if provided)
            golden_set/           <- golden replay set JSONL files
            benchmark_results.json
            manifest.json
            manifest.sha256

    Returns the bundle directory path.
    """
    bundle_dir = Path(exports_root) / f"{regime}_{timestamp_str}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Copy adapter.
    adapter_src = Path(adapter_dir)
    if adapter_src.exists():
        shutil.copytree(str(adapter_src), str(bundle_dir / "adapter"), dirs_exist_ok=True)
    else:
        log.warning("Adapter directory not found: %s", adapter_src)

    # Copy GGUF.
    gguf_src = Path(gguf_path) if gguf_path else None
    if gguf_src and gguf_src.is_file():
        shutil.copy2(str(gguf_src), str(bundle_dir / "model.gguf"))
    else:
        log.warning("GGUF file not found or not produced: %s. Bundle will be incomplete.", gguf_path)

    # Copy basis checkpoint.
    basis_src = Path(basis_checkpoint)
    if basis_src.exists():
        if basis_src.is_dir():
            shutil.copytree(str(basis_src), str(bundle_dir / "basis_checkpoint"), dirs_exist_ok=True)
        else:
            shutil.copy2(str(basis_src), str(bundle_dir / "basis_checkpoint"))
    else:
        log.info("No basis checkpoint at %s; skipping.", basis_src)

    # Copy golden set.
    golden_src = Path(golden_set_dir)
    if golden_src.exists():
        shutil.copytree(str(golden_src), str(bundle_dir / "golden_set"), dirs_exist_ok=True)
    # Write benchmark results.
    bench_path = bundle_dir / "benchmark_results.json"
    serialisable_results = {}
    for name, result in benchmark_results.items():
        if hasattr(result, "__dict__"):
            from dataclasses import asdict
            serialisable_results[name] = asdict(result) if hasattr(result, "__dataclass_fields__") else vars(result)
        else:
            serialisable_results[name] = result
    bench_path.write_text(json.dumps(serialisable_results, indent=2), encoding="utf-8")

    # Write manifest last (so it captures all copied files).
    write_manifest(str(bundle_dir), commit_hash, python_version)

    log.info("Bundle assembled at %s", bundle_dir)
    return bundle_dir
