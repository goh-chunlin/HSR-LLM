import time

from observability import (
    OTEL_CAPTURE_CONTENT as _OTEL_CAPTURE_CONTENT,
    answer_chars_hist as _answer_chars_hist,
    fingerprint_text as _fingerprint_text,
    request_latency_ms_hist as _request_latency_ms_hist,
    requests_counter as _requests_counter,
    tracer as _tracer,
)
from rag_generation import generate_answer
from rag_retrieval import normalize_user_query, retrieve_lore_hybrid, MAX_USER_QUERY_CHARS
from rag_runtime import RuntimeState

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

            with _tracer.start_as_current_span("retrieve_lore_hybrid") as retrieval_span:
                matches = retrieve_lore_hybrid(normalized_query, runtime=runtime, top_k=2)
                retrieval_span.set_attribute("app.retrieval.matches", len(matches))

            if not matches:
                request_status = "no_match"
                return "### I couldn't find any documents matching that query."

            ai_response = generate_answer(normalized_query, matches, tracer=_tracer)
            answer_len = len(ai_response)
            span.set_attribute("app.answer.length", answer_len)
            if _OTEL_CAPTURE_CONTENT:
                span.set_attribute("app.answer.preview", ai_response[:500])

            request_status = "ok"

            final_output = f"## 💬 Answer\n{ai_response}\n\n"
            final_output += "---\n### 🔍 Retrieved Reference Sources\n"
            for match in matches:
                final_output += f"- **{match['title']}** (Score: {match['score']:.4f})\n"

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
