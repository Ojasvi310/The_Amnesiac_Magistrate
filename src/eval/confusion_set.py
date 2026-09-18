"""Cross-regime adversarial evaluation set for Continual Counsel.

The key failure mode being tested: a model that has learned Regulation A's 72-hour
breach notification deadline may apply it to Regulation B scenarios where the
deadline is 48 hours, or vice versa. The confusion set forces the model to correctly
identify which regime governs a given scenario and cite the right rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class ConfusionPair:
    question: str
    correct_answer: str
    trap_regime: str          # the regime whose rule is WRONG for this question
    governing_regime: str     # the regime whose rule is CORRECT
    required_distinction: str # what the model needs to get right to distinguish them


# Hardcoded toy confusion pairs for the two synthetic regimes.
# Real use would extend this programmatically from actual document content.
_TOY_CONFUSION_PAIRS: list[ConfusionPair] = [
    ConfusionPair(
        question=(
            "A hospital's AI diagnostic system incorrectly flagged a patient's test results, "
            "contributing to a delayed diagnosis. When must the hospital report this to the "
            "Regulatory Authority?"
        ),
        correct_answer="Within 48 hours of discovery under Regulation B Section 4.1.",
        trap_regime="regime_q1",
        governing_regime="regime_q2",
        required_distinction=(
            "Regulation B (AI accountability) governs AI system incidents at 48 hours. "
            "Regulation A (data privacy) governs personal data breaches at 72 hours. "
            "The 'clinical diagnosis support' use case is explicitly named in Regulation B Section 1."
        ),
    ),
    ConfusionPair(
        question=(
            "A company's marketing database was accessed by an unauthorized third party, "
            "exposing 600 customer contact records. What is the notification deadline?"
        ),
        correct_answer=(
            "Within 72 hours of discovery under Regulation A Section 4.1, "
            "as this is a personal data breach, not an AI system incident."
        ),
        trap_regime="regime_q2",
        governing_regime="regime_q1",
        required_distinction=(
            "A database breach is a personal data breach under Regulation A, not an AI incident. "
            "The 72-hour clock (Regulation A) applies, not the 48-hour clock (Regulation B)."
        ),
    ),
    ConfusionPair(
        question=(
            "An employment screening AI system has a Conformity Assessment from 18 months ago. "
            "The vendor just retrained it on a new dataset representing 40% new data. "
            "Is a new assessment required, and under which regulation?"
        ),
        correct_answer=(
            "Yes, a new Conformity Assessment is required under Regulation B Section 2.2, "
            "because the retraining used >30% new data (a material change threshold). "
            "Regulation A alone would not require this."
        ),
        trap_regime="regime_q1",
        governing_regime="regime_q2",
        required_distinction=(
            "Regulation B Section 2.2 defines material change thresholds for AI Conformity "
            "Assessments including >30% new training data. Regulation A covers data governance "
            "but not model conformity reassessment."
        ),
    ),
    ConfusionPair(
        question=(
            "A company wants to combine its Regulation A data governance compliance documentation "
            "with a new Regulation B Conformity Assessment. Is this permissible?"
        ),
        correct_answer=(
            "Yes, Regulation B FAQ Q4 explicitly permits incorporating a Regulation A-compliant "
            "data governance section by reference, provided the combined document also covers "
            "bias disaggregation metrics and foreseeable-misuse analysis."
        ),
        trap_regime="regime_q1",
        governing_regime="regime_q2",
        required_distinction=(
            "This is a cross-regime question requiring knowledge of both. Regulation B FAQ Q4 "
            "explicitly addresses this scenario. A model that only knows Regulation A would miss "
            "the Regulation B requirements for bias disaggregation."
        ),
    ),
    ConfusionPair(
        question=(
            "A data controller has implemented a good-faith 72-hour notification after a breach "
            "but the notification was incomplete. Are they shielded from the 'repeated violation' "
            "surcharge under Regulation A?"
        ),
        correct_answer=(
            "Yes, under the Penalty Safe Harbor in Regulation A Enforcement Guidance Q1-2024-EG-001: "
            "a documented incident response plan reviewed within 12 months plus a good-faith "
            "notification effort, even if incomplete, shields against the enhanced surcharge for "
            "a first occurrence."
        ),
        trap_regime="regime_q2",
        governing_regime="regime_q1",
        required_distinction=(
            "This scenario is governed entirely by Regulation A's penalty safe harbor provision "
            "in the enforcement guidance, not Regulation B. A model confused between the two "
            "might cite Regulation B's penalty provisions instead."
        ),
    ),
]


def build_confusion_set(
    regime_dirs: dict[str, str],
    n_pairs: int = 100,
) -> list[ConfusionPair]:
    """Return confusion pairs. For the toy dataset, returns the hardcoded set.

    In real use this would scan regime_dirs for conflicting rule extractions.
    The n_pairs cap is applied; the toy set has 5 pairs.
    """
    return _TOY_CONFUSION_PAIRS[:n_pairs]


def save_confusion_set(pairs: list[ConfusionPair], output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps([vars(p) for p in pairs], indent=2)
    )


def evaluate_on_confusion_set(
    confusion_pairs: list[ConfusionPair],
    answer_fn: Callable[[str], str],
) -> tuple[float, list[dict]]:
    """Run answer_fn on each question, score, return (mean_score, per_pair_results)."""
    from src.eval.benchmarks import score_answer

    per_pair = []
    for pair in confusion_pairs:
        try:
            predicted = answer_fn(pair.question)
        except Exception as e:
            predicted = f"[error: {e}]"

        score = score_answer(predicted, pair.correct_answer, answer_type="multi_clause")
        # Check if the governing regime is mentioned in the answer
        regime_mentioned = pair.governing_regime.lower() in predicted.lower()
        # Check if the trap regime is mistakenly cited as the authority
        trap_cited = pair.trap_regime.lower() in predicted.lower() and not regime_mentioned

        per_pair.append({
            "question": pair.question[:80] + "...",
            "governing_regime": pair.governing_regime,
            "trap_regime": pair.trap_regime,
            "predicted": predicted[:200],
            "score": score,
            "regime_correct": regime_mentioned,
            "fell_into_trap": trap_cited,
        })

    mean_score = sum(r["score"] for r in per_pair) / len(per_pair) if per_pair else 0.0
    return mean_score, per_pair
