"""Grounded generation placeholder that only answers from retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass

from legal_rag.retrieval.in_memory import RetrievalResult


@dataclass(frozen=True)
class GeneratedAnswer:
    """Structured answer payload with citations and abstention flag."""

    answer: str
    citations: list[str]
    grounded: bool


class GroundedAnswerGenerator:
    """Return a grounded summary or abstain when retrieval evidence is insufficient."""

    def __init__(
        self,
        minimum_grounding_score: float = 0.6,
        ambiguity_margin: float = 0.08,
    ) -> None:
        self.minimum_grounding_score = minimum_grounding_score
        self.ambiguity_margin = ambiguity_margin

    def answer(
        self, question: str, retrieved_chunks: list[RetrievalResult]
    ) -> GeneratedAnswer:
        if not retrieved_chunks:
            return self._abstain(
                evidence_state="No evidence",
                unknown=(
                    "No retrieved chunk supports an answer to this question."
                )
            )

        top_results = retrieved_chunks[:3]
        strongest_result = top_results[0]
        citations = [_format_reference(result) for result in top_results]

        if strongest_result.score < self.minimum_grounding_score:
            return self._abstain(
                evidence_state="Weak evidence",
                citations=citations[:1],
                unknown=(
                    "The retrieved chunk is too weak to support a reliable legal answer."
                ),
            )

        if self._is_ambiguous(top_results):
            known = " ".join(
                f"{_format_reference(result)} says {self._summarize_chunk(result)}"
                for result in top_results
            )
            return self._abstain(
                evidence_state="Ambiguous evidence",
                citations=citations,
                known=known,
                unknown=(
                    "The retrieved chunks point to different provisions with similar support, "
                    "so I cannot determine a single grounded answer."
                ),
            )

        known = " ".join(
            f"{_format_reference(result)} says {self._summarize_chunk(result)}"
            for result in top_results
        )
        unknown = (
            "Anything beyond that retrieved provision is unknown from the current evidence."
        )

        return GeneratedAnswer(
            answer=(
                "Evidence assessment: Supported evidence\n"
                f"Known from retrieved context: {known}\n"
                f"Unknown from retrieved context: {unknown}\n"
                "This is legal information from the retrieved text, not legal advice."
            ),
            citations=citations,
            grounded=True,
        )

    def _abstain(
        self,
        evidence_state: str,
        unknown: str,
        citations: list[str] | None = None,
        known: str = "No supported legal proposition could be established from the retrieved chunks.",
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=(
                f"Evidence assessment: {evidence_state}\n"
                f"Known from retrieved context: {known}\n"
                f"Unknown from retrieved context: {unknown}\n"
                "I cannot provide legal advice, and I cannot go beyond the retrieved evidence."
            ),
            citations=citations or [],
            grounded=False,
        )

    def _is_ambiguous(self, top_results: list[RetrievalResult]) -> bool:
        if len(top_results) < 2:
            return False

        first, second = top_results[0], top_results[1]
        same_reference = _format_reference(first) == _format_reference(second)
        score_gap = abs(first.score - second.score)
        return not same_reference and score_gap <= self.ambiguity_margin

    def _summarize_chunk(self, result: RetrievalResult) -> str:
        text = " ".join(line.strip() for line in result.chunk.text.splitlines() if line.strip())
        return text.rstrip(".") + "."


def _format_reference(result: RetrievalResult) -> str:
    chunk = result.chunk
    reference = f"{chunk.document_id} Section {chunk.section_id}"
    if chunk.subsection_id:
        reference += f"({chunk.subsection_id})"
    if chunk.paragraph_id:
        reference += f"({chunk.paragraph_id})"
    return reference
