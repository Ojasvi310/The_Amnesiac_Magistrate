"""Per-quarter held-out benchmark construction for Continual Counsel evaluation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class QAPair:
    question: str
    answer: str
    regime: str
    section_ref: str
    answer_type: Literal["factual", "multi_clause", "numeric"]


_PENALTY_PATTERN = re.compile(
    r"(?:fine|penalty|penalt(?:y|ies)|surcharge)[^.]*?(\d+(?:\.\d+)?%|\$[\d,]+(?:\s*million)?|\d+\s*(?:million|billion)?)",
    re.IGNORECASE,
)
_TIMEFRAME_PATTERN = re.compile(
    r"(\d+)\s*(hour|day|month|year|business day)s?",
    re.IGNORECASE,
)
_SECTION_PATTERN = re.compile(r"Section\s+(\d+(?:\.\d+)*)", re.IGNORECASE)


def _extract_section_sentences(text: str) -> list[tuple[str, str]]:
    """Return (section_ref, sentence) pairs for sentences that mention a Section."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    results = []
    for sent in sentences:
        m = _SECTION_PATTERN.search(sent)
        if m:
            results.append((f"Section {m.group(1)}", sent.strip()))
    return results


def _factual_pair(section_ref: str, sentence: str, regime: str) -> QAPair | None:
    """Try to generate a factual QA pair from a sentence about a penalty or timeframe."""
    pen = _PENALTY_PATTERN.search(sentence)
    time = _TIMEFRAME_PATTERN.search(sentence)

    if pen:
        value = pen.group(1)
        question = f"What is the penalty amount mentioned in {section_ref} of {regime}?"
        return QAPair(
            question=question,
            answer=value,
            regime=regime,
            section_ref=section_ref,
            answer_type="numeric",
        )
    if time:
        value = f"{time.group(1)} {time.group(2)}s"
        question = f"What timeframe is specified in {section_ref} of {regime}?"
        return QAPair(
            question=question,
            answer=value,
            regime=regime,
            section_ref=section_ref,
            answer_type="numeric",
        )
    return None


def generate_benchmark_from_docs(
    regime_dir: str | Path,
    regime_name: str,
    held_out_fraction: float = 0.15,
) -> list[QAPair]:
    """Read .txt files from regime_dir, extract QA pairs for benchmarking.

    Generates:
    - Numeric/factual pairs: penalty amounts, timeframes extracted by regex
    - Multi-clause pairs: questions requiring combining two section citations
    """
    regime_dir = Path(regime_dir)
    pairs: list[QAPair] = []

    for txt_file in sorted(regime_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        section_sentences = _extract_section_sentences(text)

        for section_ref, sentence in section_sentences:
            pair = _factual_pair(section_ref, sentence, regime_name)
            if pair:
                pairs.append(pair)

        # Multi-clause: pair consecutive section sentences to build combined questions
        for i in range(len(section_sentences) - 1):
            sec_a, sent_a = section_sentences[i]
            sec_b, sent_b = section_sentences[i + 1]
            if sec_a != sec_b:
                question = (
                    f"How do {sec_a} and {sec_b} of {regime_name} interact? "
                    f"What conditions must be satisfied under both?"
                )
                # Answer: combined summary of the two sentences
                answer = f"Under {sec_a}: {sent_a[:80]}... Under {sec_b}: {sent_b[:80]}..."
                pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    regime=regime_name,
                    section_ref=f"{sec_a} + {sec_b}",
                    answer_type="multi_clause",
                ))

    # Held-out split: take the last held_out_fraction as the benchmark
    n_held = max(5, int(len(pairs) * held_out_fraction))
    return pairs[-n_held:]


def save_benchmark(pairs: list[QAPair], regime_dir: str | Path) -> Path:
    out = Path(regime_dir) / "benchmark.json"
    out.write_text(json.dumps([vars(p) for p in pairs], indent=2))
    return out


def load_benchmark(regime_name: str, data_root: str | Path) -> list[QAPair]:
    data_root = Path(data_root)
    bench_path = data_root / "regimes" / regime_name / "benchmark.json"
    if bench_path.exists():
        raw = json.loads(bench_path.read_text())
        return [QAPair(**r) for r in raw]
    # Generate on the fly if missing
    regime_dir = data_root / "regimes" / regime_name
    pairs = generate_benchmark_from_docs(regime_dir, regime_name)
    save_benchmark(pairs, regime_dir)
    return pairs


def score_answer(predicted: str, ground_truth: str, answer_type: str) -> float:
    """Score a predicted answer against ground truth.

    Numeric/factual: exact substring match after normalizing whitespace.
    Multi-clause: keyword overlap (Jaccard) on key terms.
    """
    pred = predicted.strip().lower()
    truth = ground_truth.strip().lower()

    if answer_type in ("factual", "numeric"):
        return 1.0 if truth in pred else 0.0

    # Multi-clause: keyword overlap
    pred_words = set(re.findall(r"\b\w+\b", pred))
    truth_words = set(re.findall(r"\b\w+\b", truth))
    # Remove stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                 "and", "or", "for", "with", "that", "this", "it", "be"}
    pred_words -= stopwords
    truth_words -= stopwords

    if not truth_words:
        return 0.0
    intersection = pred_words & truth_words
    union = pred_words | truth_words
    return len(intersection) / len(union) if union else 0.0
