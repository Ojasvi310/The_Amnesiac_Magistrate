"""Training hyperparameter config dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class OrthogonalityConfig:
    lambda_weight: float = 0.1
    lambda_bump_factor: float = 2.0
    lambda_max: float = 0.8


@dataclass
class TrainingLoopConfig:
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    fp16: bool = True
    bf16: bool = False


@dataclass
class ReplayConfig:
    ratio: float = 0.30
    golden_set_min: int = 500
    golden_set_max: int = 2000
    refresh_enabled: bool = False


@dataclass
class GatesConfig:
    bwt_degradation_threshold: float = 0.03


@dataclass
class MergeConfig:
    cadence: Any = 2            # int or "never"
    ties_keep_top_pct: float = 0.20
    post_merge_bwt_tolerance: float = 0.02


@dataclass
class DistillationConfig:
    enabled: bool = False
    loss_weight: float = 0.5


@dataclass
class DPOConfig:
    enabled: bool = False
    revalidation_mode: str = "gate"   # "gate" = option (a)
    num_epochs: int = 1
    beta: float = 0.1


@dataclass
class QuantizationConfig:
    level: str = "Q4_K_M"


@dataclass
class ExportConfig:
    cadence: str = "every_adapter"   # "every_adapter" | "merge_only"


@dataclass
class EvalConfig:
    held_out_fraction: float = 0.15
    confusion_set_size: int = 100


@dataclass
class TrainingConfig:
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    orthogonality: OrthogonalityConfig = field(default_factory=OrthogonalityConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_training_config(path: Path | None = None) -> TrainingConfig:
    if path is None:
        from src.config_loader import _repo_root
        path = _repo_root() / "configs" / "training.yaml"

    raw = yaml.safe_load(path.read_text())

    def _get(d, key, default):
        return d.get(key, default) if isinstance(d, dict) else default

    r = raw or {}
    lora_raw = r.get("lora", {})
    orth_raw = r.get("orthogonality", {})
    train_raw = r.get("training", {})
    replay_raw = r.get("replay", {})
    gates_raw = r.get("gates", {})
    merge_raw = r.get("merge", {})
    distill_raw = r.get("distillation", {})
    dpo_raw = r.get("dpo", {})
    quant_raw = r.get("quantization", {})
    export_raw = r.get("export", {})
    eval_raw = r.get("eval", {})

    return TrainingConfig(
        lora=LoRAConfig(
            rank=lora_raw.get("rank", 16),
            alpha=lora_raw.get("alpha", 32),
            dropout=lora_raw.get("dropout", 0.05),
            target_modules=lora_raw.get("target_modules", LoRAConfig().target_modules),
            bias=lora_raw.get("bias", "none"),
            task_type=lora_raw.get("task_type", "CAUSAL_LM"),
        ),
        orthogonality=OrthogonalityConfig(
            lambda_weight=orth_raw.get("lambda_weight", 0.1),
            lambda_bump_factor=orth_raw.get("lambda_bump_factor", 2.0),
            lambda_max=orth_raw.get("lambda_max", 0.8),
        ),
        training=TrainingLoopConfig(
            num_epochs=train_raw.get("num_epochs", 3),
            learning_rate=train_raw.get("learning_rate", 2e-4),
            warmup_ratio=train_raw.get("warmup_ratio", 0.03),
            lr_scheduler=train_raw.get("lr_scheduler", "cosine"),
            weight_decay=train_raw.get("weight_decay", 0.01),
            gradient_accumulation_steps=train_raw.get("gradient_accumulation_steps", 4),
            max_seq_length=train_raw.get("max_seq_length", 1024),
            fp16=train_raw.get("fp16", True),
            bf16=train_raw.get("bf16", False),
        ),
        replay=ReplayConfig(
            ratio=replay_raw.get("ratio", 0.30),
            golden_set_min=replay_raw.get("golden_set_min", 500),
            golden_set_max=replay_raw.get("golden_set_max", 2000),
            refresh_enabled=replay_raw.get("refresh_enabled", False),
        ),
        gates=GatesConfig(bwt_degradation_threshold=gates_raw.get("bwt_degradation_threshold", 0.03)),
        merge=MergeConfig(
            cadence=merge_raw.get("cadence", 2),
            ties_keep_top_pct=merge_raw.get("ties_keep_top_pct", 0.20),
            post_merge_bwt_tolerance=merge_raw.get("post_merge_bwt_tolerance", 0.02),
        ),
        distillation=DistillationConfig(
            enabled=distill_raw.get("enabled", False),
            loss_weight=distill_raw.get("loss_weight", 0.5),
        ),
        dpo=DPOConfig(
            enabled=dpo_raw.get("enabled", False),
            revalidation_mode=dpo_raw.get("revalidation_mode", "gate"),
            num_epochs=dpo_raw.get("num_epochs", 1),
            beta=dpo_raw.get("beta", 0.1),
        ),
        quantization=QuantizationConfig(level=quant_raw.get("level", "Q4_K_M")),
        export=ExportConfig(cadence=export_raw.get("cadence", "every_adapter")),
        eval=EvalConfig(
            held_out_fraction=eval_raw.get("held_out_fraction", 0.15),
            confusion_set_size=eval_raw.get("confusion_set_size", 100),
        ),
    )
