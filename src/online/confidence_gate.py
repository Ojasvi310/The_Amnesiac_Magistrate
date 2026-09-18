"""
Retrieval confidence gate.

If the best retrieval score falls below a configurable threshold, generation
is blocked entirely and the query is escalated for human review. This is a
hard stop, not a warning -- callers must catch InsufficientGrounding.
"""

from __future__ import annotations


class InsufficientGrounding(Exception):
    """
    Raised when no retrieved chunk meets the minimum confidence threshold.
    Callers must not proceed with LLM generation after catching this --
    instead return an escalation response to the user.
    """

    def __init__(self, max_score: float, threshold: float, query: str) -> None:
        self.max_score = max_score
        self.threshold = threshold
        self.query = query
        super().__init__(
            f"Retrieval confidence {max_score:.4f} below threshold {threshold:.4f} "
            f"for query: {query!r}"
        )


def check_retrieval_confidence(
    top_k_results: list[tuple],
    threshold: float,
) -> list[tuple]:
    """
    Validate that at least one retrieved chunk surpasses the confidence threshold.

    Parameters
    ----------
    top_k_results:
        List of (IndexMetadata, score) tuples as returned by query_index.
    threshold:
        Minimum inner-product score required to proceed with generation.

    Returns
    -------
    top_k_results unchanged, so callers can chain this call inline.

    Raises
    ------
    InsufficientGrounding
        If top_k_results is empty or the maximum score is below threshold.
    """
    if not top_k_results:
        raise InsufficientGrounding(
            max_score=0.0,
            threshold=threshold,
            query="<no results>",
        )

    # Each element is (IndexMetadata, score); score is the second item.
    max_score = max(score for _, score in top_k_results)

    if max_score < threshold:
        # We do not have the original query text here; callers embed it in the
        # exception message when they re-raise or log.
        raise InsufficientGrounding(
            max_score=max_score,
            threshold=threshold,
            query="<see caller context>",
        )

    return top_k_results


def escalation_response(query: str) -> dict:
    """
    Build the structured escalation payload returned to API consumers when
    generation is blocked by InsufficientGrounding.
    """
    return {
        "status": "escalate",
        "message": "Insufficient grounding -- escalate to human review",
        "query": query,
    }
