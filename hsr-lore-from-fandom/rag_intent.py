import re

from rag_types import IntentClassification, IntentType

_ENTITY_PATTERNS = (
    re.compile(r"^\s*who\s+is\b", re.IGNORECASE),
    re.compile(r"^\s*who\s+are\b", re.IGNORECASE),
    re.compile(r"^\s*tell\s+me\s+about\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+is\b", re.IGNORECASE),
    re.compile(r"^\s*what\s+are\b", re.IGNORECASE),
)

_TIMELINE_PATTERNS = (
    re.compile(r"\bwhen\b", re.IGNORECASE),
    re.compile(r"\btimeline\b", re.IGNORECASE),
    re.compile(r"\bhistory\b", re.IGNORECASE),
    re.compile(r"\bchronolog(?:y|ical)\b", re.IGNORECASE),
    re.compile(r"\bversion\s*\d", re.IGNORECASE),
    re.compile(r"\b(before|after|earlier|later|first|last)\b", re.IGNORECASE),
)

_RELATION_PATTERNS = (
    re.compile(r"\brelationship\b", re.IGNORECASE),
    re.compile(r"\bbetween\b", re.IGNORECASE),
    re.compile(r"\bcompared\s+to\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bversus\b|\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bconnected\b|\bconnection\b", re.IGNORECASE),
    re.compile(r"\bally\b|\benemy\b|\bfriend\b", re.IGNORECASE),
)


def classify_query_intent(query: str) -> IntentClassification:
    text = str(query or "").strip()

    if _matches_any(text, _RELATION_PATTERNS):
        return {"label": "relation_query", "confidence": 0.95}

    if _matches_any(text, _TIMELINE_PATTERNS):
        return {"label": "timeline_query", "confidence": 0.95}

    if _matches_any(text, _ENTITY_PATTERNS):
        return {"label": "entity_lookup", "confidence": 0.9}

    return {"label": "other", "confidence": 0.5}


def retrieval_top_k_for_intent(intent: IntentType, default_top_k: int) -> int:
    if intent == "timeline_query":
        return max(default_top_k, 5)
    if intent == "relation_query":
        return max(default_top_k, 5)
    return default_top_k


def retrieval_score_adjustment(intent: IntentType, query: str, title: str, text: str) -> float:
    q = query.lower()
    t = title.lower()
    body = text.lower()

    if intent == "entity_lookup":
        title_tokens = [token for token in re.findall(r"[a-z0-9]+", q) if len(token) > 2]
        if title_tokens and all(token in t for token in title_tokens[:3]):
            return 0.35
        if any(token in t for token in title_tokens):
            return 0.15
        return 0.0

    if intent == "timeline_query":
        if re.search(r"\b\d{3,4}\b", q) and re.search(r"\b\d{3,4}\b", body + " " + t):
            return 0.2
        if any(k in body + " " + t for k in ("version", "era", "timeline", "history", "before", "after")):
            return 0.1
        return 0.0

    if intent == "relation_query":
        relation_terms = ("with", "between", "ally", "enemy", "friend", "against", "related")
        hits = sum(1 for term in relation_terms if term in body + " " + t)
        return min(0.2, 0.04 * hits)

    return 0.0


def build_intent_addendum(intent: IntentType) -> str:
    if intent == "entity_lookup":
        return (
            "INTENT FOCUS (ENTITY LOOKUP): Prefer direct identity/profile facts about the requested entity. "
            "If multiple entities are mentioned, answer only the one explicitly asked in <user_question>."
        )

    if intent == "timeline_query":
        return (
            "INTENT FOCUS (TIMELINE): Prioritize chronological ordering and time/version anchors explicitly present "
            "in <retrieved_knowledge>. Do not infer missing dates or sequence steps."
        )

    if intent == "relation_query":
        return (
            "INTENT FOCUS (RELATION): Focus on explicit relationships between entities. "
            "Do not imply connections unless the retrieved text directly states them."
        )

    return "INTENT FOCUS (OTHER): Provide the most direct grounded answer using only explicit retrieved facts."


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)
