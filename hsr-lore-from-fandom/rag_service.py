import time
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


def _format_single_line(value: object) -> str:
    return " ".join(str(value).strip().split())


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


def _append_media_block(lines: list[str], match: RetrievedChunk, max_media: int = 3) -> None:
    media = match.get("media")
    if media is None or not media:
        return

    lines.append("  - Media Preview:")
    for entry in media[:max_media]:
        media_type = _format_single_line(entry.get("type", "")) or "unknown"
        url = _format_single_line(entry.get("url", ""))
        title = _format_single_line(entry.get("title", ""))
        description = _format_single_line(entry.get("description", ""))
        attribution_url = _format_single_line(entry.get("attributionUrl", ""))
        copyright_or_license = _format_single_line(entry.get("copyrightOrLicense", ""))

        label = title or url or "(missing URL)"
        if media_type == "image" and url:
            lines.append(f"    - Image: {label}")
            lines.append(f"      ![{label}]({url})")
        elif media_type == "video" and url:
            youtube_video_id = _extract_youtube_video_id(url)
            if youtube_video_id is not None:
                watch_url = _youtube_watch_url(youtube_video_id)
                thumb_url = _youtube_thumbnail_url(youtube_video_id)
                lines.append(f"    - YouTube: [{label}]({watch_url})")
                lines.append(f"      [![{label}]({thumb_url})]({watch_url})")
                lines.append("      Click the thumbnail to watch on YouTube.")
            else:
                lines.append(f"    - Video: [{label}]({url})")
            if _is_probably_video_url(url):
                lines.append(f"      <video src=\"{url}\" controls width=\"480\"></video>")
                lines.append("      If your browser blocks inline playback, open the link above.")
            elif youtube_video_id is None:
                lines.append("      Open the link to watch this video.")
        elif url:
            lines.append(f"    - {media_type}: [{label}]({url})")
        else:
            lines.append(f"    - {media_type}: {label}")

        if description:
            lines.append(f"      - Description: {description}")
        if attribution_url:
            lines.append(f"      - Attribution: {attribution_url}")
        if copyright_or_license:
            lines.append(f"      - Rights: {copyright_or_license}")

def hsr_rag_interface(user_query: str, runtime: RuntimeState) -> str:
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
                )

            normalized_query = normalize_user_query(user_query)
            if not normalized_query:
                request_status = "empty_query"
                return "### Please enter a lore question."

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
                return "### I couldn't find any documents matching that query."

            ai_response = generate_answer(normalized_query, matches, tracer=_tracer, intent_label=intent_label)
            answer_len = len(ai_response)
            span.set_attribute("app.answer.length", answer_len)
            if _OTEL_CAPTURE_CONTENT:
                span.set_attribute("app.answer.preview", ai_response[:500])

            request_status = "ok"

            final_output = f"## 💬 Answer\n{ai_response}\n\n"
            final_output += "---\n### 🔍 Retrieved Reference Sources\n"
            for match in matches:
                lines = [f"- **{match['title']}** (Score: {match['score']:.4f})"]
                _append_reference_block(lines, match)
                _append_media_block(lines, match)
                final_output += "\n".join(lines) + "\n"

            return final_output
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
