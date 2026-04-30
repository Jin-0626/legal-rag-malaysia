"""Grounded generation placeholder that only answers from retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass
import re

from legal_rag.retrieval.in_memory import RetrievalResult

EMPLOYMENT_AGREEMENT_CORE_CUES = (
    "employment agreement",
    "employment contract",
    "contract of service",
)
EMPLOYMENT_AGREEMENT_ADVICE_CUES = (
    "before signing",
    "signing",
    "protect myself",
    "protect yourself",
    "need to check",
    "should i check",
    "what should i check",
)


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
                direct_answer="There is not enough retrieved legal information to answer that question reliably.",
                legal_basis="No retrieved legal provision supports an answer to this question.",
                practical_meaning=(
                    "You would need more relevant legal sources before drawing a reliable conclusion."
                ),
                important_limits=(
                    "This response is limited by the absence of supporting retrieved evidence."
                ),
            )

        if _looks_like_employment_agreement_question(question):
            checklist_answer = _build_employment_agreement_checklist_answer(question, retrieved_chunks[:5])
            if checklist_answer is not None:
                return checklist_answer

        top_results = retrieved_chunks[:3]
        strongest_result = top_results[0]
        citations = _deduplicate_preserve_order(_format_reference(result) for result in top_results)

        if strongest_result.score < self.minimum_grounding_score:
            return self._abstain(
                direct_answer="The retrieved material is too weak to support a reliable answer.",
                citations=citations[:1],
                legal_basis=(
                    f"The strongest retrieved source was {citations[0]}, but its support is not strong enough to rely on."
                ),
                practical_meaning=(
                    "The system should avoid making a confident legal statement from weak evidence."
                ),
                important_limits=(
                    "This is legal information only, not legal advice, and the evidence is currently insufficient."
                ),
            )

        if self._is_ambiguous(top_results):
            return self._abstain(
                direct_answer="The retrieved sources point to more than one plausible provision, so I cannot give a single grounded answer.",
                citations=citations,
                legal_basis="Relevant retrieved sources include:\n" + "\n".join(
                    f"- { _format_reference(result) }: {_summarize_chunk_brief(result)}"
                    for result in top_results
                ),
                practical_meaning=(
                    "The current evidence suggests related provisions, but not one clearly dominant answer."
                ),
                important_limits=(
                    "This response is limited by ambiguity in the retrieved evidence and should not be treated as legal advice."
                ),
            )

        primary = top_results[0]
        primary_reference = _format_reference(primary)
        primary_summary = _summarize_chunk_brief(primary)
        broad_match = _is_broad_query_with_narrow_evidence(question, primary)
        additional_sources = [
            f"- {_format_reference(result)}: {_summarize_chunk_brief(result)}"
            for result in top_results[1:]
            if _format_reference(result) != primary_reference
        ]

        return GeneratedAnswer(
            answer=_format_structured_answer(
                direct_answer=(
                    (
                        f"The retrieved evidence does not fully answer this broader question. "
                        f"The closest source is {primary_reference}, which addresses {primary_summary.lower()}"
                    )
                    if broad_match
                    else f"The retrieved sources indicate that {primary_summary}"
                ),
                legal_basis=(
                    (
                        f"The closest retrieved provision is {primary_reference}. "
                        f"It addresses {primary_summary.lower()}"
                    )
                    if broad_match
                    else f"{primary_reference} indicates that {primary_summary}"
                    + (
                        "\nAdditional supporting sources:\n" + "\n".join(additional_sources)
                        if additional_sources
                        else ""
                    )
                ),
                practical_meaning=(
                    (
                        "In plain English, the current retrieved material is narrower than the question you asked, "
                        "so it should be treated as only partial legal information."
                    )
                    if broad_match
                    else "In plain English, this is the main point supported by the retrieved legal text."
                ),
                important_limits=(
                    (
                        "This is legal information, not legal advice. More directly relevant provisions may be needed before drawing a reliable conclusion."
                    )
                    if broad_match
                    else "This is legal information, not legal advice. The answer may depend on facts that are not covered in the retrieved sources."
                ),
                citations=citations,
            ),
            citations=citations,
            grounded=True,
        )

    def _abstain(
        self,
        direct_answer: str,
        legal_basis: str,
        practical_meaning: str,
        important_limits: str,
        citations: list[str] | None = None,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=_format_structured_answer(
                direct_answer=direct_answer,
                legal_basis=legal_basis,
                practical_meaning=practical_meaning,
                important_limits=important_limits,
                citations=citations or [],
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


def _format_reference(result: RetrievalResult) -> str:
    chunk = result.chunk
    unit_label = {
        "article": "Article",
        "perkara": "Perkara",
    }.get(chunk.unit_type.lower(), "Section")
    reference = f"{chunk.act_title or chunk.document_id}, {unit_label} {chunk.unit_id or chunk.section_id}"
    if chunk.subsection_id:
        reference += f"({chunk.subsection_id})"
    if chunk.paragraph_id:
        reference += f"({chunk.paragraph_id})"
    return reference


def _summarize_chunk_brief(result: RetrievalResult, limit: int = 220) -> str:
    text = " ".join(line.strip() for line in result.chunk.text.splitlines() if line.strip())
    heading = (result.chunk.section_heading or "").strip()
    if heading:
        escaped_heading = re.escape(heading)
        text = re.sub(rf"^{escaped_heading}\s*", "", text, count=1, flags=re.IGNORECASE)
    sentence = text.split(". ", 1)[0].strip().rstrip(".")
    normalized = sentence or text.strip().rstrip(".")
    normalized = re.sub(r"^(Section|Article|Perkara)\s+[A-Za-z0-9]+[\s.:_-]*", "", normalized, flags=re.IGNORECASE)
    if _looks_like_unit_token(normalized):
        normalized = _heading_topic(heading)
    if not normalized:
        normalized = _heading_topic(heading)
    if len(normalized) > limit:
        normalized = normalized[:limit].rsplit(" ", 1)[0].strip()
        normalized += "..."
    return normalized[:1].upper() + normalized[1:] + "."


def _format_structured_answer(
    *,
    direct_answer: str,
    legal_basis: str,
    practical_meaning: str,
    important_limits: str,
    citations: list[str],
) -> str:
    lines = [
        "Direct Answer:",
        direct_answer.strip(),
        "",
        "Legal Basis:",
        legal_basis.strip(),
        "",
        "Practical Meaning:",
        practical_meaning.strip(),
        "",
        "Important Limits:",
        important_limits.strip(),
        "",
        "Sources:",
    ]
    if citations:
        lines.extend(f"- {citation}" for citation in citations)
    else:
        lines.append("- No supporting sources retrieved.")
    return "\n".join(lines).strip()


def _format_checklist_answer(
    *,
    direct_answer: str,
    checklist_items: list[str],
    legal_basis: list[str],
    important_limits: str,
    citations: list[str],
) -> str:
    lines = [
        "Direct Answer:",
        direct_answer.strip(),
        "",
        "Checklist:",
    ]
    if checklist_items:
        lines.extend(f"- {item}" for item in checklist_items)
    else:
        lines.append("- No grounded checklist items could be established from the retrieved sources.")
    lines.extend(
        [
            "",
            "Legal Basis:",
        ]
    )
    if legal_basis:
        lines.extend(f"- {item}" for item in legal_basis)
    else:
        lines.append("- No retrieved legal provision clearly supports a checklist item.")
    lines.extend(
        [
            "",
            "Important Limits:",
            important_limits.strip(),
            "",
            "Sources:",
        ]
    )
    if citations:
        lines.extend(f"- {citation}" for citation in citations)
    else:
        lines.append("- No supporting sources retrieved.")
    return "\n".join(lines).strip()


def _deduplicate_preserve_order(items: list[str] | tuple[str, ...] | object) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _is_broad_query_with_narrow_evidence(question: str, result: RetrievalResult) -> bool:
    if re.search(r"\b(section|article|perkara)\b", question, re.IGNORECASE):
        return False
    question_terms = _content_terms(question)
    heading_terms = _content_terms(result.chunk.section_heading or "")
    if not heading_terms:
        return False
    overlap = question_terms & heading_terms
    return len(overlap) == 0


def _looks_like_employment_agreement_question(question: str) -> bool:
    lowered = question.lower()
    has_core = any(cue in lowered for cue in EMPLOYMENT_AGREEMENT_CORE_CUES)
    has_advice = any(cue in lowered for cue in EMPLOYMENT_AGREEMENT_ADVICE_CUES)
    token_pattern = (
        "employment" in lowered
        and ("agreement" in lowered or "contract" in lowered)
        and any(token in lowered for token in ("check", "sign", "protect", "rights", "terms"))
    )
    return (has_core and has_advice) or token_pattern


def _build_employment_agreement_checklist_answer(
    question: str,
    retrieved_chunks: list[RetrievalResult],
) -> GeneratedAnswer | None:
    section_results = _deduplicate_results_by_reference(retrieved_chunks)
    if not section_results:
        return None

    categories: dict[str, RetrievalResult] = {}
    for result in section_results:
        chunk = result.chunk
        haystack = f"{chunk.section_heading} {chunk.text}".lower()
        if "contracts to be in writing" in haystack or "contract of service" in haystack:
            categories.setdefault("written_terms", result)
        if "notice of termination" in haystack:
            categories["termination"] = result
        elif "termination" in haystack:
            categories.setdefault("termination", result)
        if "wage" in haystack or "deduction" in haystack:
            categories.setdefault("wages", result)
        if any(term in haystack for term in ("hours of work", "overtime", "rest day")):
            categories.setdefault("hours", result)
        if any(term in haystack for term in ("annual leave", "sick leave", "holiday", "holidays")):
            categories.setdefault("leave", result)
        if any(term in haystack for term in ("trade union", "discrimination", "favourable", "flexible working arrangement")):
            categories.setdefault("rights", result)

    if not categories:
        return None

    citations = _deduplicate_preserve_order(_format_reference(result) for result in section_results)
    checklist_items: list[str] = []
    legal_basis: list[str] = []

    if "written_terms" in categories:
        result = categories["written_terms"]
        checklist_items.append("Written contract terms: check whether the contract is in writing and whether termination terms are stated.")
        legal_basis.append(f"{_format_reference(result)} supports checking whether the contract of service is recorded in writing and includes termination terms.")
    else:
        checklist_items.append("Job scope and key written terms: practical check only; the retrieved legal sources here do not directly set out a complete clause-by-clause contract checklist.")

    if "wages" in categories:
        result = categories["wages"]
        checklist_items.append("Salary, wages, deductions, and benefits: check how wages are paid and whether deductions are lawful.")
        legal_basis.append(f"{_format_reference(result)} supports checking wage and deduction terms before you agree to them.")

    if "hours" in categories:
        result = categories["hours"]
        checklist_items.append("Working hours, overtime, and rest days: check the stated hours, overtime expectations, and rest-day arrangements.")
        legal_basis.append(f"{_format_reference(result)} supports checking working-time protections such as hours of work, overtime, or rest days.")

    if "leave" in categories:
        result = categories["leave"]
        checklist_items.append("Leave and holidays: check annual leave, sick leave, and holiday-related entitlements.")
        legal_basis.append(f"{_format_reference(result)} supports checking leave and holiday entitlements.")

    if "termination" in categories:
        result = categories["termination"]
        checklist_items.append("Termination and notice: check how much notice is required and what happens when the contract ends.")
        legal_basis.append(f"{_format_reference(result)} supports checking notice and termination terms.")

    if "rights" in categories:
        result = categories["rights"]
        checklist_items.append("Employee protections and rights: check whether the terms appear consistent with employee protections recognized in the retrieved provisions.")
        legal_basis.append(f"{_format_reference(result)} supports reviewing employee-protection or workplace-rights issues reflected in the retrieved sources.")

    direct_answer = (
        "Before signing an employment agreement, the retrieved sources support checking whether the contract terms are written clearly, what notice or termination terms apply, and what the agreement says about wages, leave, working conditions, and employee protections."
    )
    important_limits = (
        "This is legal information, not legal advice. The retrieved provisions support only part of a broader employment-agreement checklist, so practical items like job scope or negotiation points should not be treated here as legal conclusions unless a retrieved source covers them."
    )

    return GeneratedAnswer(
        answer=_format_checklist_answer(
            direct_answer=direct_answer,
            checklist_items=checklist_items,
            legal_basis=legal_basis,
            important_limits=important_limits,
            citations=citations,
        ),
        citations=citations,
        grounded=True,
    )


def _content_terms(text: str) -> set[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "under",
        "what",
        "which",
        "how",
        "do",
        "does",
        "i",
        "my",
        "is",
        "are",
        "be",
        "this",
        "that",
        "agreement",
        "employment",
        "act",
        "law",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _looks_like_unit_token(text: str) -> bool:
    candidate = text.strip()
    return bool(re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?", candidate))


def _heading_topic(heading: str) -> str:
    cleaned = re.sub(r"^(Section|Article|Perkara)\s+[A-Za-z0-9]+[\s.:_-]*", "", heading or "", flags=re.IGNORECASE).strip()
    return cleaned or "the retrieved provision"


def _deduplicate_results_by_reference(results: list[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[str] = set()
    deduplicated: list[RetrievalResult] = []
    for result in results:
        reference = _format_reference(result)
        if reference in seen:
            continue
        seen.add(reference)
        deduplicated.append(result)
    return deduplicated
