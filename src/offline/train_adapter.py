"""O-LoRA training (Steps B and C) for the continual-counsel offline pipeline.

GPU-dependent training loop. Run on Colab only.
Parts that call into CUDA/bitsandbytes are marked @pytest.mark.colab_only.
"""
from __future__ import annotations

# @pytest.mark.colab_only -- portions that instantiate models require CUDA
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from src.config_loader import BaseConfig
from src.training_config import TrainingConfig
from src.offline.distill import hidden_state_distill_loss

log = logging.getLogger(__name__)


class OrthonormalBasis:
    """Tracks an incremental QR-based orthonormal basis across adapter A-matrices.

    Each adapter in the O-LoRA sequence contributes a set of A-matrices (the
    low-rank down-projection). We accumulate an orthonormal basis from their
    columns so that new adapters can be penalised for projecting into the same
    subspace as prior ones (catastrophic forgetting via weight interference).

    The basis is stored as a single matrix Q (cols = basis vectors).
    """

    def __init__(self) -> None:
        # Q: float32 CPU tensor, shape (feature_dim, n_basis_vectors).
        # None until the first update.
        self._Q: torch.Tensor | None = None

    @property
    def rank(self) -> int:
        """Number of basis vectors accumulated so far."""
        return 0 if self._Q is None else self._Q.shape[1]

    def update(self, new_A_matrices: list[torch.Tensor]) -> None:
        """Incorporate a new adapter's A-matrices into the cumulative basis.

        Each A-matrix has shape (lora_rank, feature_dim). We flatten along the
        rank axis, run QR, and extend Q by the new orthogonal directions.

        Silently skips matrices that are fully in the existing span (no new
        directions to add).
        """
        for A in new_A_matrices:
            # A: (r, d) -- rows are the LoRA rank directions.
            cols = A.detach().float().cpu().T  # (d, r)
            if self._Q is None:
                Q, _ = torch.linalg.qr(cols)
                self._Q = Q
            else:
                # Project out the existing span before adding.
                proj = self._Q @ (self._Q.T @ cols)
                residual = cols - proj
                if residual.norm() < 1e-6:
                    # Fully within existing span; nothing new to add.
                    continue
                Q_new, _ = torch.linalg.qr(residual)
                # Filter out near-zero columns produced by QR on a rank-deficient residual.
                norms = Q_new.norm(dim=0)
                Q_new = Q_new[:, norms > 1e-6]
                if Q_new.shape[1] > 0:
                    self._Q = torch.cat([self._Q, Q_new], dim=1)

    def orthogonality_loss(self, new_A_matrices: list[torch.Tensor]) -> torch.Tensor:
        """Penalise the current adapter's A-matrices for overlap with the accumulated basis.

        Loss = mean over all A-matrices of ||Q^T A^T||_F^2, which measures how much
        each new direction projects onto the existing subspace. Zero means fully
        orthogonal (ideal); high values mean significant overlap (forgetting risk).
        """
        if self._Q is None or len(new_A_matrices) == 0:
            # No prior regimes -- nothing to be orthogonal to.
            device = new_A_matrices[0].device if new_A_matrices else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=False)

        device = new_A_matrices[0].device
        Q = self._Q.to(device)
        losses = []
        for A in new_A_matrices:
            # A: (r, d). Q: (d, k).
            # We want ||Q^T @ A^T||_F^2 = how much of A's column space overlaps Q.
            proj = Q.T @ A.T  # (k, r)
            losses.append((proj ** 2).sum())

        return torch.stack(losses).mean()

    def reset_from_merged(self, merged_A_matrices: list[torch.Tensor]) -> None:
        """Rebuild the basis from a TIES-merged adapter's A-matrices.

        Called after a TIES merge event (Step E) so that the basis reflects the
        merged parameter state rather than the cumulative sum of all per-regime
        bases (which would grow monotonically and saturate available rank).
        """
        self._Q = None
        self.update(merged_A_matrices)
        log.info("OrthonormalBasis reset from merged adapter; new rank = %d", self.rank)


def compose_training_batch(
    regime_dir: str,
    replay_cache_dir: str,
    regime_name: str,
    replay_ratio: float,
) -> Dataset:
    """Load new regime documents and replay-cache pairs; return a HuggingFace Dataset.

    The dataset contains dicts with keys 'prompt' and 'response'.
    regime_dir: path to data/regimes/<regime>/ -- expects a docs.json or similar.
    replay_cache_dir: path to data/replay_cache/ root.
    """
    import glob

    regime_dir = Path(regime_dir)
    replay_cache_dir = Path(replay_cache_dir)

    # Load new-regime QA pairs.
    new_pairs = []
    docs_path = regime_dir / "docs.json"
    qa_path = regime_dir / "qa.json"

    if qa_path.exists():
        new_pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    elif docs_path.exists():
        # Treat each doc chunk as a faux "response"; the prompt is a fixed template.
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        for doc in docs:
            new_pairs.append({
                "prompt": f"Summarise the following {regime_name} compliance requirement:\n{doc.get('text', '')}",
                "response": doc.get("text", ""),
            })
    else:
        log.warning("No QA or docs JSON found in %s. Training on replay only.", regime_dir)

    if not new_pairs:
        log.warning("New regime data is empty for %s.", regime_name)

    # Load replay-cache golden sets.
    replay_pairs = []
    golden_glob = str(replay_cache_dir / "**" / "golden_set.jsonl")
    for path in glob.glob(golden_glob, recursive=True):
        try:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    replay_pairs.append(json.loads(line))
        except Exception as exc:
            log.warning("Failed to load replay file %s: %s", path, exc)

    # Mix: replay_ratio fraction from replay, rest from new regime.
    n_total = len(new_pairs) + len(replay_pairs)
    if n_total == 0:
        raise ValueError("Training batch is completely empty.")

    n_replay = min(len(replay_pairs), int(n_total * replay_ratio))
    n_new = n_total - n_replay

    import random
    sampled_replay = random.sample(replay_pairs, n_replay) if n_replay < len(replay_pairs) else replay_pairs
    sampled_new = random.sample(new_pairs, min(n_new, len(new_pairs)))

    all_pairs = sampled_new + sampled_replay
    random.shuffle(all_pairs)

    return Dataset.from_list([
        {"prompt": p.get("prompt", ""), "response": p.get("response", "")}
        for p in all_pairs
    ])


def detect_gpu_capabilities() -> dict:
    """Detect the current GPU and return capability information.

    Returns a dict with:
        gpu_name  (str)     -- CUDA device name, or "CPU" if no GPU
        vram_gb   (float)   -- total VRAM in GiB (0 if no GPU)
        supports_bf16 (bool) -- True if the GPU's compute capability >= 8.0
                                (Ampere+), which is required for stable bf16.
    """
    if not torch.cuda.is_available():
        log.warning("No CUDA device detected. Training will be extremely slow on CPU.")
        print("GPU capabilities: no CUDA device found.")
        return {"gpu_name": "CPU", "vram_gb": 0.0, "supports_bf16": False}

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    # bf16 requires Ampere (sm_80) or later.
    supports_bf16 = props.major >= 8

    info = {
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "supports_bf16": supports_bf16,
    }
    print(f"GPU capabilities: {info}")
    log.info("GPU capabilities: %s", info)
    return info


def bump_lambda(current_lambda: float, cfg: Any) -> float:
    """Apply bump_factor to current_lambda, capped at lambda_max.

    Called when Step D fails, to increase the orthogonality penalty before retrying.
    """
    new_val = current_lambda * cfg.orthogonality.lambda_bump_factor
    return min(new_val, cfg.orthogonality.lambda_max)


class _OLoRATrainer(Trainer):
    """Custom Trainer subclass that injects the orthogonality loss into the CE loss."""

    def __init__(self, basis: OrthonormalBasis, lambda_weight: float, train_cfg: TrainingConfig, teacher_model=None, **kwargs):
        super().__init__(**kwargs)
        self.basis = basis
        self.lambda_weight = lambda_weight
        self.train_cfg = train_cfg
        self.teacher_model = teacher_model
        self._step_count = 0
        self._run_dir: Path | None = None  # set externally after construction

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract current LoRA A-matrices for orthogonality loss computation.
        current_A_matrices = [
            p for name, p in model.named_parameters()
            if "lora_A" in name and p.requires_grad
        ]

        # Standard cross-entropy loss from the base Trainer.
        if self.train_cfg.distillation.enabled and self.teacher_model is not None:
            # Run student with hidden states.
            outputs = model(**inputs, output_hidden_states=True)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True)

            ce_loss = outputs.loss
            distill_loss = hidden_state_distill_loss(
                teacher_hiddens=list(teacher_outputs.hidden_states),
                student_hiddens=list(outputs.hidden_states),
            )
            loss = ce_loss + self.train_cfg.distillation.loss_weight * distill_loss
        else:
            outputs = model(**inputs)
            loss = outputs.loss

        # Orthogonality penalty -- penalise A-matrices that project onto prior subspace.
        orth_loss = self.basis.orthogonality_loss(current_A_matrices)
        loss = loss + self.lambda_weight * orth_loss

        # Log basis rank to jsonl file every step.
        if self._run_dir is not None:
            rank_log_path = self._run_dir / "basis_rank.jsonl"
            entry = {"step": self._step_count, "basis_rank": self.basis.rank}
            with rank_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        self._step_count += 1

        return (loss, outputs) if return_outputs else loss


def _tokenize_batch(examples, tokenizer, max_seq_length):
    """Tokenize prompt+response pairs and set labels (mask the prompt tokens)."""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for prompt, response in zip(examples["prompt"], examples["response"]):
        full_text = prompt + "\n" + response
        prompt_enc = tokenizer(prompt + "\n", add_special_tokens=False)
        full_enc = tokenizer(
            full_text,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # Labels: -100 for prompt tokens (don't compute loss there), real ids for response.
        prompt_len = len(prompt_enc["input_ids"])
        labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[prompt_len:]
        labels = labels[: len(input_ids)]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def train_lora(
    base_model_id: str,
    training_batch: Dataset,
    basis: OrthonormalBasis,
    train_cfg: TrainingConfig,
    base_cfg: BaseConfig,
    regime_name: str,
    run_dir: str,
) -> Path:
    """Full O-LoRA training loop (Steps B+C).

    Loads the base model in 4-bit (bitsandbytes NF4), wraps it with LoRA via PEFT,
    then trains with CE + lambda * orthogonality_loss. Optionally adds hidden-state
    distillation loss if train_cfg.distillation.enabled.

    Returns the path to the saved adapter directory.
    """
    # @pytest.mark.colab_only -- bitsandbytes 4-bit loading requires CUDA
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu_info = detect_gpu_capabilities()

    # Override fp16/bf16 flags based on actual hardware capability.
    use_bf16 = gpu_info["supports_bf16"]
    use_fp16 = (not use_bf16) and gpu_info["gpu_name"] != "CPU"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # bfloat16 compute dtype is preferred when available; falls back to float16.
        # float32 would cause OOM on a 15 GB Colab T4.
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("Loading base model %s in 4-bit", base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # prepare_model_for_kbit_training casts norms and LM head to fp32 and enables
    # gradient checkpointing, which is necessary before adding LoRA to a quantised model.
    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = train_cfg.lora
    peft_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Optionally load teacher for distillation (same base model, frozen).
    teacher_model = None
    if train_cfg.distillation.enabled:
        log.info("Loading frozen teacher model for hidden-state distillation.")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    # Tokenise the training batch.
    loop_cfg = train_cfg.training
    tokenized = training_batch.map(
        lambda ex: _tokenize_batch(ex, tokenizer, loop_cfg.max_seq_length),
        batched=True,
        remove_columns=training_batch.column_names,
    )

    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=loop_cfg.num_epochs,
        # TODO: make per-GPU dynamic once VRAM detection is wired
        per_device_train_batch_size=4,
        gradient_accumulation_steps=loop_cfg.gradient_accumulation_steps,
        learning_rate=loop_cfg.learning_rate,
        lr_scheduler_type=loop_cfg.lr_scheduler,
        warmup_ratio=loop_cfg.warmup_ratio,
        weight_decay=loop_cfg.weight_decay,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_dir=str(run_dir / "logs"),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = _OLoRATrainer(
        basis=basis,
        lambda_weight=train_cfg.orthogonality.lambda_weight,
        train_cfg=train_cfg,
        teacher_model=teacher_model,
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding=True),
    )
    trainer._run_dir = run_dir

    log.info("Starting O-LoRA training for regime %s", regime_name)
    trainer.train()

    # Update basis with the newly trained A-matrices.
    trained_A_matrices = [
        p.detach().cpu()
        for name, p in model.named_parameters()
        if "lora_A" in name
    ]
    basis.update(trained_A_matrices)
    log.info("Basis updated after training; new rank = %d", basis.rank)

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("Adapter saved to %s", adapter_dir)

    return adapter_dir
