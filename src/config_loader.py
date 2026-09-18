"""Shared config loading utilities — imported by both offline and online code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RetrieverConfig:
    embed_model_id: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5
    confidence_threshold: float = 0.45


@dataclass
class PathsConfig:
    data_regimes: str = "data/regimes"
    replay_cache: str = "data/replay_cache"
    exports: str = "exports"
    runs: str = "runs"
    db: str = "continual_counsel.db"
    llama_cpp_dir: str = "llama.cpp"


@dataclass
class ProfileConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_rank_ceiling: int = 32
    vram_budget_gb: Any = None  # None = detect at runtime
    gguf_quant_level: str = "Q4_K_M"
    model_id_fallback: str | None = None


@dataclass
class BaseConfig:
    active_profile: str = "proxy"
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)
    paths: PathsConfig = field(default_factory=PathsConfig)
    retrieval: RetrieverConfig = field(default_factory=RetrieverConfig)
    python_version_expected: str = "3.11"

    @property
    def profile(self) -> ProfileConfig:
        return self.profiles.get(self.active_profile, ProfileConfig())


def _repo_root() -> Path:
    """Walk up from this file to find the repo root (contains configs/)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Cannot locate repo root — configs/ directory not found.")


def load_base_config(profile: str | None = None, repo_root: Path | None = None) -> BaseConfig:
    root = repo_root or _repo_root()
    raw = yaml.safe_load((root / "configs" / "base.yaml").read_text())

    cfg = BaseConfig()
    cfg.active_profile = profile or raw.get("active_profile", "proxy")
    cfg.python_version_expected = raw.get("python_version_expected", "3.11")

    raw_paths = raw.get("paths", {})
    cfg.paths = PathsConfig(**{k: v for k, v in raw_paths.items() if hasattr(PathsConfig, k) or True})

    raw_ret = raw.get("retrieval", {})
    cfg.retrieval = RetrieverConfig(
        embed_model_id=raw_ret.get("embed_model_id", "BAAI/bge-small-en-v1.5"),
        top_k=raw_ret.get("top_k", 5),
        confidence_threshold=raw_ret.get("confidence_threshold", 0.45),
    )

    for name, pdata in raw.get("profiles", {}).items():
        cfg.profiles[name] = ProfileConfig(
            model_id=pdata.get("model_id", "Qwen/Qwen2.5-1.5B-Instruct"),
            lora_rank_ceiling=pdata.get("lora_rank_ceiling", 32),
            vram_budget_gb=pdata.get("vram_budget_gb"),
            gguf_quant_level=pdata.get("gguf_quant_level", "Q4_K_M"),
            model_id_fallback=pdata.get("model_id_fallback"),
        )

    return cfg


def _resolve_path(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else base / p


def resolve_paths(cfg: BaseConfig, repo_root: Path | None = None) -> dict[str, Path]:
    root = repo_root or _repo_root()
    return {
        field_name: _resolve_path(root, getattr(cfg.paths, field_name))
        for field_name in vars(cfg.paths)
    }
