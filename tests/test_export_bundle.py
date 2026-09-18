"""Tests for export bundle manifest verification in export.py.

A silent partial export is the most dangerous failure mode in the two-machine
architecture — it fails quietly while the inference service runs on stale/partial
weights. These tests verify that the manifest verification logic correctly catches
tampered files, missing files, truncated files, and manifest.sha256 mismatches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


try:
    from src.offline.export import write_manifest, verify_bundle
except ImportError:
    pytest.skip("src.offline.export not importable", allow_module_level=True)


def _build_valid_bundle(tmp_path: Path) -> Path:
    """Create a bundle with correct manifest and sha256, return bundle dir."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.gguf").write_bytes(b"fake gguf content " * 20)
    (bundle / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 16}')
    (bundle / "benchmark_results.json").write_text('{"regime_q1": {"accuracy": 0.85}}')

    write_manifest(bundle, commit_hash="abc123", python_version="3.11.0")
    return bundle


class TestExportBundleVerification:
    def test_valid_bundle_passes(self, tmp_path):
        """A correctly assembled bundle with matching manifest passes verification."""
        bundle = _build_valid_bundle(tmp_path)
        passed, errors = verify_bundle(bundle)
        assert passed, f"Valid bundle should pass verification. Errors: {errors}"
        assert len(errors) == 0

    def test_tampered_file_rejected(self, tmp_path):
        """Modifying a file after manifest is written → verification detects mismatch."""
        bundle = _build_valid_bundle(tmp_path)
        # Tamper with the GGUF file
        (bundle / "model.gguf").write_bytes(b"tampered content!")
        passed, errors = verify_bundle(bundle)
        assert not passed, "Tampered file should cause verification to fail."
        assert any("model.gguf" in e for e in errors), f"Error should mention model.gguf. Errors: {errors}"

    def test_missing_file_rejected(self, tmp_path):
        """Deleting a file listed in the manifest → verification detects missing file."""
        bundle = _build_valid_bundle(tmp_path)
        (bundle / "benchmark_results.json").unlink()
        passed, errors = verify_bundle(bundle)
        assert not passed
        assert any("benchmark_results.json" in e or "missing" in e.lower() for e in errors)

    def test_truncated_file_rejected(self, tmp_path):
        """Writing fewer bytes than manifest records → size mismatch detected."""
        bundle = _build_valid_bundle(tmp_path)
        # Truncate the GGUF file
        original = (bundle / "model.gguf").read_bytes()
        (bundle / "model.gguf").write_bytes(original[:10])  # truncated to 10 bytes
        passed, errors = verify_bundle(bundle)
        assert not passed
        assert any("model.gguf" in e for e in errors)

    def test_manifest_sha256_mismatch_rejected(self, tmp_path):
        """Tampering with manifest.json after writing manifest.sha256 → rejected immediately."""
        bundle = _build_valid_bundle(tmp_path)
        # Tamper with manifest.json content (but don't update manifest.sha256)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["commit_hash"] = "TAMPERED"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        # manifest.sha256 still holds the original hash → mismatch
        passed, errors = verify_bundle(bundle)
        assert not passed
        assert any("manifest" in e.lower() and ("sha256" in e.lower() or "mismatch" in e.lower())
                   for e in errors), f"Should report manifest sha256 mismatch. Errors: {errors}"

    def test_missing_manifest_rejected(self, tmp_path):
        """Bundle with no manifest.json → rejected."""
        bundle = tmp_path / "no_manifest_bundle"
        bundle.mkdir()
        (bundle / "model.gguf").write_bytes(b"content")
        passed, errors = verify_bundle(bundle)
        assert not passed
        assert any("manifest" in e.lower() for e in errors)
