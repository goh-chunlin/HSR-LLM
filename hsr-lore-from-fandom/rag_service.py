import time
import re
from typing import cast
from urllib.parse import parse_qs, urlparse

from observability import (
    OTEL_CAPTURE_CONTENT as _OTEL_CAPTURE_CONTENT,
    answer_chars_hist as _answer_chars_hist,
    fingerprint_text as _fingerprint_text,
    request_latency_ms_hist as _request_latency_ms_hist,
    requests_counter as _requests_counter,
    tracer as _tracer,
)
from rag_generation import generate_answer
from rag_intent import classify_query_intent, retrieval_top_k_for_intent
from rag_retrieval import normalize_user_query, retrieve_lore_hybrid, MAX_USER_QUERY_CHARS
from rag_runtime import RuntimeState
from rag_types import RetrievedChunk


GalleryItem = tuple[str, str]


def _format_single_line(value: object) -> str:
    return " ".join(str(value).strip().split())


def _normalize_optional_str(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _format_single_line(value)


def _normalize_title_for_dedup(value: str) -> str:
    lowered = value.lower()
    stripped = re.sub(r"[^\w\s]+", " ", lowered)
    return " ".join(stripped.split())


def _display_dedup_key(match: RetrievedChunk) -> str | None:
    reference = match.get("reference")
    if isinstance(reference, dict):
        page_id = _normalize_optional_str(reference.get("pageId"))
        if page_id:
            return f"pageId:{page_id}"

        source_url = _normalize_optional_str(reference.get("sourceUrl"))
        if source_url:
            return f"sourceUrl:{source_url}"

    title = _normalize_optional_str(match.get("title"))
    normalized_title = _normalize_title_for_dedup(title)
    if normalized_title:
        return f"title:{normalized_title}"

    return None


def _dedupe_matches_for_display(matches: list[RetrievedChunk]) -> list[RetrievedChunk]:
    # Keep only the highest-scoring representative for each page-like key.
    best_by_key: dict[str, RetrievedChunk] = {}
    pass_through: list[RetrievedChunk] = []

    for match in matches:
        key = _display_dedup_key(match)
        if key is None:
            pass_through.append(match)
            continue

        existing = best_by_key.get(key)
        if existing is None or float(match["score"]) > float(existing["score"]):
            best_by_key[key] = match

    deduped = list(best_by_key.values()) + pass_through
    deduped.sort(key=lambda item: float(item["score"]), reverse=True)
    return deduped


def _is_probably_video_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith((".mp4", ".webm", ".ogg", ".mov", ".m4v"))


def _extract_youtube_video_id(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    host = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.strip("/")

    if host == "youtu.be" and path:
        return path.split("/")[0]

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if path == "watch":
            query = parse_qs(parsed.query)
            video_ids = query.get("v", [])
            if video_ids:
                return video_ids[0]

        if path.startswith("shorts/"):
            parts = path.split("/", 1)
            if len(parts) == 2 and parts[1]:
                return parts[1]

        if path.startswith("embed/"):
            parts = path.split("/", 1)
            if len(parts) == 2 and parts[1]:
                return parts[1]

    return None


def _youtube_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _youtube_thumbnail_url(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def _gallery_caption(match: RetrievedChunk, entry: object) -> str:
    entry_dict = cast(dict[str, object], entry)
    parts: list[str] = []

    title = _format_single_line(entry_dict.get("title", ""))
    description = _format_single_line(entry_dict.get("description", ""))
    attribution_url = _format_single_line(entry_dict.get("attributionUrl", ""))
    copyright_or_license = _format_single_line(entry_dict.get("copyrightOrLicense", ""))

    if title:
        parts.append(title)
    if description:
        parts.append(description)
    if attribution_url:
        parts.append(f"Attribution: {attribution_url}")
    if copyright_or_license:
        parts.append(f"Rights: {copyright_or_license}")

    source_title = _normalize_optional_str(match.get("title"))
    if source_title:
        parts.append(f"Source: {source_title}")

    return " | ".join(parts)


def _collect_gallery_items(matches: list[RetrievedChunk], max_media: int = 10) -> list[GalleryItem]:
    gallery_items: list[GalleryItem] = []
    seen_urls: set[str] = set()

    for match in matches:
        media = match.get("media")
        if media is None or not media:
            continue

        for entry in media[:max_media]:
            media_type = _format_single_line(entry.get("type", "")) or "unknown"
            url = _format_single_line(entry.get("url", ""))
            if not url or url in seen_urls:
                continue

            if media_type == "image" or (media_type == "video" and _is_probably_video_url(url)):
                gallery_items.append((url, _gallery_caption(match, entry)))
                seen_urls.add(url)

    return gallery_items


def _append_reference_block(lines: list[str], match: RetrievedChunk) -> None:
    reference = match.get("reference")
    if not isinstance(reference, dict):
        return

    reference_dict = cast(dict[str, object], reference)

    source_name = _format_single_line(reference_dict.get("sourceName", ""))
    source_url = _format_single_line(reference_dict.get("sourceUrl", ""))

    if source_name and source_url:
        lines.append(f"  - Source: [{source_name}]({source_url})")
    elif source_name:
        lines.append(f"  - Source: {source_name}")
    elif source_url:
        lines.append(f"  - Source URL: {source_url}")

    for key, label in (
        ("pageId", "Page ID"),
        ("revisionId", "Revision ID"),
        ("retrievedAt", "Retrieved At"),
        ("license", "License"),
    ):
        value = _format_single_line(reference_dict.get(key, ""))
        if value:
            lines.append(f"  - {label}: {value}")


def _append_media_block(lines: list[str], match: RetrievedChunk, max_media: int = 10) -> None:
    media = match.get("media")
    if media is None or not media:
        return

    gallery_count = 0
    youtube_lines: list[str] = []
    other_lines: list[str] = []

    for entry in media[:max_media]:
        media_type = _format_single_line(entry.get("type", "")) or "unknown"
        url = _format_single_line(entry.get("url", ""))
        title = _format_single_line(entry.get("title", ""))
        description = _format_single_line(entry.get("description", ""))
        attribution_url = _format_single_line(entry.get("attributionUrl", ""))
        copyright_or_license = _format_single_line(entry.get("copyrightOrLicense", ""))
        entry_lines: list[str] | None = None

        label = title or url or "(missing URL)"
        if media_type == "image" and url:
            gallery_count += 1
        elif media_type == "video" and url:
            youtube_video_id = _extract_youtube_video_id(url)
            if youtube_video_id is not None:
                watch_url = _youtube_watch_url(youtube_video_id)
                thumb_url = _youtube_thumbnail_url(youtube_video_id)
                entry_lines = [
                    f"    - YouTube: [{label}]({watch_url})",
                    f"      [![{label}]({thumb_url})]({watch_url})",
                    "      Click the thumbnail to watch on YouTube.",
                ]
            if _is_probably_video_url(url):
                gallery_count += 1
            elif youtube_video_id is None:
                entry_lines = [
                    f"    - Video: [{label}]({url})",
                    "      Open the link to watch this video.",
                ]
        elif url:
            entry_lines = [f"    - {media_type}: [{label}]({url})"]
        else:
            entry_lines = [f"    - {media_type}: {label}"]

        if description:
            if entry_lines is not None:
                entry_lines.append(f"      - Description: {description}")
        if attribution_url:
            if entry_lines is not None:
                entry_lines.append(f"      - Attribution: {attribution_url}")
        if copyright_or_license:
            if entry_lines is not None:
                entry_lines.append(f"      - Rights: {copyright_or_license}")

        if entry_lines is not None:
            if media_type == "video" and url and _extract_youtube_video_id(url) is not None:
                youtube_lines.extend(entry_lines)
            else:
                other_lines.extend(entry_lines)

    if gallery_count == 0 and not youtube_lines and not other_lines:
        return

    lines.append("  - Media Preview:")
    if gallery_count:
        suffix = "item" if gallery_count == 1 else "items"
        lines.append(f"    - Gallery: {gallery_count} {suffix} shown below.")
    lines.extend(youtube_lines)
    lines.extend(other_lines)


def hsr_rag_interface(user_query: str, runtime: RuntimeState) -> tuple[str, list[GalleryItem]]:
    request_started = time.perf_counter()
    request_status = "unknown"
    answer_len = 0

    with _tracer.start_as_current_span("hsr_rag_interface") as span:
        raw_query = str(user_query or "")
        span.set_attribute("app.query.length", len(raw_query))
        span.set_attribute("app.query.sha256", _fingerprint_text(raw_query))
        if _OTEL_CAPTURE_CONTENT:
            span.set_attribute("app.query.preview", raw_query[:MAX_USER_QUERY_CHARS])

        try:
            runtime.initialize()

            if runtime.init_error is not None:
                request_status = "init_error"
                return (
                    "### Runtime initialization failed.\n"
                    "Required artifacts may be missing in this deployment.\n\n"
                    f"**Error:** `{runtime.init_error}`"
                    ,
                    [],
                )

            normalized_query = normalize_user_query(user_query)
            if not normalized_query:
                request_status = "empty_query"
                return "### Please enter a lore question.", []

            span.set_attribute("app.query.normalized_length", len(normalized_query))

            intent = classify_query_intent(normalized_query)
            intent_label = intent["label"]
            intent_confidence = float(intent["confidence"])
            span.set_attribute("app.intent.label", intent_label)
            span.set_attribute("app.intent.confidence", intent_confidence)

            top_k = retrieval_top_k_for_intent(intent_label, default_top_k=4)

            with _tracer.start_as_current_span("retrieve_lore_hybrid") as retrieval_span:
                matches = retrieve_lore_hybrid(
                    normalized_query,
                    runtime=runtime,
                    top_k=top_k,
                    intent_label=intent_label,
                )
                retrieval_span.set_attribute("app.retrieval.matches", len(matches))
                retrieval_span.set_attribute("app.intent.label", intent_label)

            if not matches:
                request_status = "no_match"
                return "### I couldn't find any documents matching that query.", []

            ai_response = generate_answer(normalized_query, matches, tracer=_tracer, intent_label=intent_label)
            answer_len = len(ai_response)
            span.set_attribute("app.answer.length", answer_len)
            if _OTEL_CAPTURE_CONTENT:
                span.set_attribute("app.answer.preview", ai_response[:500])

            request_status = "ok"

            final_output = f"## 💬 Answer\n{ai_response}\n\n"
            final_output += "---\n### 🔍 Retrieved Reference Sources\n"
            display_matches = _dedupe_matches_for_display(matches)
            gallery_items = _collect_gallery_items(display_matches)
            for match in display_matches:
                lines = [f"- **{match['title']}** (Score: {match['score']:.4f})"]
                _append_reference_block(lines, match)
                _append_media_block(lines, match)
                final_output += "\n".join(lines) + "\n"

            return final_output, gallery_items
        except Exception as e:
            request_status = "exception"
            span.record_exception(e)
            span.set_attribute("app.error.type", type(e).__name__)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - request_started) * 1000.0
            metric_attrs = {"status": request_status}

            _requests_counter.add(1, metric_attrs)
            _request_latency_ms_hist.record(elapsed_ms, metric_attrs)
            if answer_len > 0:
                _answer_chars_hist.record(answer_len, metric_attrs)

            span.set_attribute("app.request.status", request_status)
            span.set_attribute("app.request.latency_ms", elapsed_ms)
