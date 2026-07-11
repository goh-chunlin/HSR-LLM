import re
import os
from typing import Any, cast

from rag_intent import retrieval_score_adjustment
from rag_runtime import RuntimeState
from rag_types import IntentType, RetrievedChunk


_QUERY_STOPWORDS = frozenset(
    {
        "who",
        "what",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "for",
        "at",
        "by",
        "from",
        "with",
        "and",
        "or",
        "not",
        "be",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "how",
        "why",
        "when",
        "where",
        "this",
        "that",
        "which",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "i",
        "about",
        "tell",
        "give",
        "year",
        "years",
        "day",
        "days",
        "time",
        "times",
        "page",
        "pages",
        "total",
        "many",
        "much",
        "some",
        "any",
        "no",
    }
)

MAX_USER_QUERY_CHARS = 1200
DEFAULT_OVERLAY_SCORE_BONUS = 0.4

_stemmer: Any | None = None


def _read_overlay_score_bonus() -> float:
    raw_value = os.getenv("HSR_LORE_OVERLAY_SCORE_BONUS", str(DEFAULT_OVERLAY_SCORE_BONUS)).strip()
    if not raw_value:
        return DEFAULT_OVERLAY_SCORE_BONUS

    try:
        return float(raw_value)
    except ValueError:
        print(
            f"[STARTUP WARNING] Invalid HSR_LORE_OVERLAY_SCORE_BONUS={raw_value!r}; defaulting to {DEFAULT_OVERLAY_SCORE_BONUS}.",
            flush=True,
        )
        return DEFAULT_OVERLAY_SCORE_BONUS


OVERLAY_SCORE_BONUS = _read_overlay_score_bonus()


def tokenize_text(text: str) -> list[str]:
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer  # type: ignore[import-untyped]

        _stemmer = PorterStemmer()

    clean = re.sub(r'[#\?!\.,:;\(\)\[\]"\'\-\/]', " ", text.lower())
    tokens: list[str] = []
    for token in clean.split():
        token_strip = token.strip()
        if token_strip:
            tokens.append(cast(str, _stemmer.stem(token_strip)))  # type: ignore[reportUnknownMemberType]
    return tokens


def normalize_user_query(query: str, max_chars: int = MAX_USER_QUERY_CHARS) -> str:
    text = str(query).strip()
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _extract_context_pairs(query: str) -> list[tuple[str, str]]:
    q = query.lower()
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(kw: str, d: str) -> None:
        key = (kw, d)
        if key not in seen:
            pairs.append(key)
            seen.add(key)

    for kw, d in re.findall(r"([a-z]+)\s*#?\s*(\d+)", q):
        if kw not in _QUERY_STOPWORDS and len(kw) > 1:
            _add(kw, d)

    for d, kw in re.findall(r"(\d+)[a-z]{0,2}\s+([a-z]+)", q):
        if kw not in _QUERY_STOPWORDS and len(kw) > 1:
            _add(kw, d)

    for d in re.findall(r"#\s*(\d+)", q):
        _add("#", d)

    return pairs


def _digit_in_context(text: str, keyword: str, digit: str) -> int | None:
    d = re.escape(digit)

    if keyword == "#":
        m = re.search(r"#\s*\b" + d + r"\b", text)
        return m.start() if m else None

    kw = re.escape(keyword)
    m = re.search(r"\b" + kw + r"\b[^\n]{0,30}?\b" + d + r"\b", text)
    if m:
        return m.start()

    m = re.search(r"\b" + d + r"[a-z]{0,2}\b[^\n]{0,30}?\b" + kw + r"\b", text)
    if m:
        return m.start()

    return None


def _normalize_title_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", text.lower())).strip()


def _title_match_bonus(query: str, title: str) -> float:
    normalized_query = _normalize_title_text(query)
    normalized_title = _normalize_title_text(title)

    if not normalized_query or not normalized_title:
        return 0.0

    if normalized_title == normalized_query:
        return 0.6

    if normalized_title in normalized_query:
        return 0.45

    query_tokens = set(normalized_query.split())
    title_tokens = normalized_title.split()
    if title_tokens and all(token in query_tokens for token in title_tokens):
        return 0.3

    return 0.0


def retrieve_lore_hybrid(
    query: str,
    runtime: RuntimeState,
    top_k: int = 3,
    intent_label: IntentType = "other",
) -> list[RetrievedChunk]:
    if (
        not runtime.runtime_ready
        or runtime.init_error is not None
        or runtime.embed_model is None
        or runtime.index is None
        or runtime.bm25 is None
    ):
        return []

    import faiss
    import numpy as np

    query_tokens = tokenize_text(query)
    query_digits = re.findall(r"\d+", query)
    query_context_pairs = _extract_context_pairs(query)

    bm25_scores = cast(np.ndarray, runtime.bm25.get_scores(query_tokens))  # type: ignore[reportUnknownMemberType]
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

    query_vector = runtime.embed_model.encode([query], convert_to_numpy=True)  # type: ignore[call-overload]
    faiss.normalize_L2(query_vector)
    faiss_scores, faiss_indices = runtime.index.search(query_vector, 30)

    combined_scores: dict[int, float] = {}

    for idx, score in enumerate(bm25_scores):
        if score > 0:
            combined_scores[idx] = combined_scores.get(idx, 0.0) + 0.5 * (score / max_bm25)

    for score, idx in zip(faiss_scores[0], faiss_indices[0]):
        if idx < 0 or idx >= len(runtime.text_metadata):
            continue
        combined_scores[idx] = combined_scores.get(idx, 0.0) + 0.5 * float(score)

    final_filtered_scores: dict[int, float] = {}
    for idx, total_score in combined_scores.items():
        raw_text = str(runtime.text_metadata[idx].get("text", "")).lower()
        title = str(runtime.text_metadata[idx].get("title", "")).lower()
        source = str(runtime.text_metadata[idx].get("source", "base")).lower()
        title_bonus = _title_match_bonus(query, title)

        if title_bonus > 0:
            total_score += title_bonus

        if source == "overlay":
            total_score += OVERLAY_SCORE_BONUS

        if query_context_pairs:
            combined_text = raw_text + " " + title
            matched_pos = None
            for kw, d in query_context_pairs:
                pos = _digit_in_context(combined_text, kw, d)
                if pos is not None:
                    matched_pos = pos
                    break

            if matched_pos is None:
                final_filtered_scores[idx] = 0.0
                continue

            proximity_bonus = 0.3 * max(0.0, 1.0 - matched_pos / 400.0)
            total_score += proximity_bonus
        elif query_digits:
            if not any(digit in raw_text or digit in title for digit in query_digits):
                final_filtered_scores[idx] = 0.0
                continue

        total_score += retrieval_score_adjustment(
            intent=intent_label,
            query=query,
            title=title,
            text=raw_text,
        )

        final_filtered_scores[idx] = total_score

    sorted_docs = sorted(
        final_filtered_scores.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )[:top_k]

    results: list[RetrievedChunk] = []
    for idx, final_score in sorted_docs:
        results.append(
            {
                "title": runtime.text_metadata[idx]["title"],
                "text": runtime.text_metadata[idx]["text"],
                "score": final_score,
            }
        )

    return results
