"""
Self-verification pass for generated answers.

Claims are extracted via pattern matching and then checked for grounding
in the retrieved chunks. An optional LLM forward pass can tighten the check.
No training-stack imports -- callable LLM is passed in by the caller if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# Patterns that indicate legally significant claims worth verifying.
_LEGAL_CITATION_RE = re.compile(
    r"""
    (?:
        Section\s+\d+(?:\.\d+)*       # Section 12.3
      | \u00a7\s*\d+(?:\.\d+)*                # § 4.1
      | Regulation\s+[A-Z][A-Z0-9\-]* # Regulation EC-123
      | Article\s+\d+                 # Article 9
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_NUMERIC_CLAIM_RE = re.compile(
    r"""
    (?:
        \$[\d,]+(?:\.\d+)?            # dollar amounts
      | \d+(?:\.\d+)?\s*%             # percentages
      | \d+\s*(?:days?|months?|years?|hours?|weeks?)  # timeframes
      | penalty\s+of\s+\$?[\d,]+      # penalty amounts
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Sentence boundary -- rough but sufficient for this heuristic.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class VerificationResult:
    verified: bool
    flagged_claims: list[str] = field(default_factory=list)
    explanation: str = ""


def extract_claims(text: str) -> list[str]:
    """
    Return sentences that contain legal citation patterns or numeric values
    likely to be factual claims (penalties, timeframes, etc.).

    Only sentences with at least one such pattern are included; pure prose
    sentences are assumed to be non-verifiable opinion.
    """
    sentences = _SENTENCE_END_RE.split(text.strip())
    claims: list[str] = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if _LEGAL_CITATION_RE.search(s) or _NUMERIC_CLAIM_RE.search(s):
            claims.append(s)
    return claims


def _claim_is_grounded(claim: str, chunks) -> bool:
    """
    Check whether a claim is supported by at least one retrieved chunk via
    substring keyword overlap. We look for the most distinctive tokens
    (length > 3) from the claim in any chunk text.
    """
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and"}
    tokens = [
        w.lower()
        for w in re.findall(r"[A-Za-z0-9]+", claim)
        if len(w) > 3 and w.lower() not in _STOP
    ]
    if not tokens:
        return True  # Nothing specific to verify; give benefit of the doubt.

    for chunk_meta in chunks:
        chunk_lower = chunk_meta.text.lower()
        # Require at least half the tokens to appear in the chunk.
        matches = sum(1 for t in tokens if t in chunk_lower)
        if matches >= max(1, len(tokens) // 2):
            return True
    return False


def verify_answer(
    answer: str,
    retrieved_chunks,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> VerificationResult:
    """
    Verify a generated answer against the retrieved context.

    For each extracted claim:
      1. Check substring/keyword overlap against retrieved chunks.
      2. If llm_fn is provided, also ask it explicitly whether the claim is
         supported by the retrieved text (one forward pass per flagged claim).

    Parameters
    ----------
    answer:
        The LLM-generated answer text.
    retrieved_chunks:
        List of IndexMetadata objects from the retrieval step.
    llm_fn:
        Optional callable(prompt: str) -> str. When provided, a second-opinion
        LLM pass is made for claims that fail the heuristic check.

    Returns
    -------
    VerificationResult
    """
    claims = extract_claims(answer)
    if not claims:
        return VerificationResult(
            verified=True,
            flagged_claims=[],
            explanation="No verifiable claims detected; answer accepted.",
        )

    flagged: list[str] = []
    context_text = "\n---\n".join(c.text for c in retrieved_chunks)

    for claim in claims:
        grounded = _claim_is_grounded(claim, retrieved_chunks)

        if not grounded and llm_fn is not None:
            prompt = (
                "You are a legal fact-checker. Answer only YES or NO.\n"
                f"Is the following claim supported by the provided text?\n\n"
                f"Claim: {claim}\n\n"
                f"Text:\n{context_text}\n\n"
                "Answer (YES/NO):"
            )
            llm_verdict = llm_fn(prompt).strip().upper()
            # If LLM says YES, override the heuristic failure.
            grounded = llm_verdict.startswith("YES")

        if not grounded:
            flagged.append(claim)

    if flagged:
        return VerificationResult(
            verified=False,
            flagged_claims=flagged,
            explanation=(
                f"{len(flagged)} claim(s) could not be grounded in the "
                f"retrieved context and may be hallucinated."
            ),
        )

    return VerificationResult(
        verified=True,
        flagged_claims=[],
        explanation="All verifiable claims are grounded in retrieved context.",
    )


def re_generate_if_flagged(
    result: VerificationResult,
    regenerate_fn: Callable[[], str],
) -> tuple[str, bool]:
    """
    If the verification result contains flagged claims, call regenerate_fn to
    obtain a fresh answer. The regenerated answer is not itself re-verified
    (to avoid infinite loops); callers should log that regeneration occurred.

    Returns
    -------
    (final_answer, was_regenerated)
    """
    if result.flagged_claims:
        new_answer = regenerate_fn()
        return new_answer, True
    # flagged_claims is empty; verification passed
    return "", False  # Caller already holds the original answer
