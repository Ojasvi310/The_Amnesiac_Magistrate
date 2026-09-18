"""pytest configuration and shared fixtures for Continual Counsel."""
from __future__ import annotations

import json
import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "colab_only: marks tests that require Colab GPU environment; skipped by default locally."
    )


def pytest_collection_modifyitems(config, items):
    skip_colab = pytest.mark.skip(reason="requires Colab GPU — run via scripts/run_colab_tests.sh")
    for item in items:
        if "colab_only" in item.keywords:
            item.add_marker(skip_colab)


@pytest.fixture
def tiny_adapter_dirs(tmp_path):
    """Two fake adapter directories with minimal weight files (numpy .npz format for test simplicity)."""
    dirs = []
    rng = np.random.default_rng(42)
    for i in range(2):
        adapter_dir = tmp_path / f"adapter_{i}"
        adapter_dir.mkdir()
        import torch
        weights = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn((8, 16), dtype=torch.float32),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn((16, 8), dtype=torch.float32),
            "base_model.model.layers.1.self_attn.v_proj.lora_A.weight": torch.randn((8, 16), dtype=torch.float32),
            "base_model.model.layers.1.self_attn.v_proj.lora_B.weight": torch.randn((16, 8), dtype=torch.float32),
        }
        torch.save(weights, adapter_dir / "adapter_model.bin")
        # Minimal adapter config
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        }))
        dirs.append(adapter_dir)
    return dirs


def _make_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@pytest.fixture
def sample_manifest(tmp_path):
    """Creates a valid bundle directory with manifest.json and manifest.sha256."""
    bundle = tmp_path / "test_bundle"
    bundle.mkdir()

    # Create some dummy files
    (bundle / "model.gguf").write_bytes(b"fake gguf content " * 10)
    (bundle / "adapter_config.json").write_text('{"peft_type": "LORA"}')

    files_list = []
    for f in bundle.iterdir():
        digest = _make_sha256(f)
        files_list.append({"path": f.name, "sha256": digest, "size_bytes": f.stat().st_size})

    manifest = {
        "regime": "regime_q1",
        "timestamp": "2024-01-15T10:00:00",
        "commit_hash": "abc123def456",
        "python_version": "3.11.0",
        "files": files_list,
    }
    manifest_json = json.dumps(manifest, indent=2)
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(manifest_json)

    manifest_sha256 = hashlib.sha256(manifest_json.encode()).hexdigest()
    (bundle / "manifest.sha256").write_text(manifest_sha256 + "\n")

    return bundle


@pytest.fixture
def sample_index_dir(tmp_path):
    """Creates a tiny FAISS index (5 vectors, dim=384) and matching chunks.json."""
    try:
        import faiss
    except ImportError:
        pytest.skip("faiss-cpu not installed")

    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()

    dim = 384
    n = 5
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    # Normalize for cosine similarity (IndexFlatIP on normalized = cosine)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    chunks = [
        {"text": f"Section {i+1} content about regulation.", "regime": "regime_q1",
         "section": f"Section {i+1}", "effective_date": "2024-01-01"}
        for i in range(n)
    ]
    (index_dir / "chunks.json").write_text(json.dumps(chunks))

    return index_dir


@pytest.fixture
def sample_golden_set(tmp_path):
    """Creates 10 sample golden QA pairs as JSONL."""
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    pairs = [
        {
            "prompt": f"What is the penalty for violation {i} under Section {i+1}?",
            "response": f"The penalty is {(i+1) * 1000} dollars under Section {i+1}.2.",
            "regime": "regime_q1",
            "seed_agreement": True,
            "grounded": True,
        }
        for i in range(10)
    ]
    lines = "\n".join(json.dumps(p) for p in pairs)
    (golden_dir / "golden_pairs.jsonl").write_text(lines)
    return golden_dir
