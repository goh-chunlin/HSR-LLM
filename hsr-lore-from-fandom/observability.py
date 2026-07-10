import hashlib
import importlib
import os
from typing import Any


OTEL_CAPTURE_CONTENT = os.getenv("HSR_OTEL_CAPTURE_CONTENT", "false").strip().lower() == "true"


def setup_observability() -> tuple[Any | None, Any | None, Any | None, Any | None]:
    """
    Initialize OpenTelemetry for traces and metrics.
    Falls back to no-op behavior if OTel isn't available or misconfigured.
    """
    try:
        otel_module = importlib.import_module("opentelemetry")
        metrics = otel_module.metrics
        trace = otel_module.trace
        OTLPMetricExporter = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.metric_exporter"
        ).OTLPMetricExporter
        OTLPSpanExporter = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        ).OTLPSpanExporter
        MeterProvider = importlib.import_module("opentelemetry.sdk.metrics").MeterProvider
        PeriodicExportingMetricReader = importlib.import_module(
            "opentelemetry.sdk.metrics.export"
        ).PeriodicExportingMetricReader
        Resource = importlib.import_module("opentelemetry.sdk.resources").Resource
        TracerProvider = importlib.import_module("opentelemetry.sdk.trace").TracerProvider
        BatchSpanProcessor = importlib.import_module(
            "opentelemetry.sdk.trace.export"
        ).BatchSpanProcessor

        resource = Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", "hsr-lore-rag-space"),
            "service.version": os.getenv("SPACE_BUILD_VERSION", "unknown"),
            "deployment.environment": os.getenv("OTEL_ENV", "prod"),
        })

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer("hsr.rag.app")

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=10000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter("hsr.rag.app")

        requests_counter = meter.create_counter(
            name="rag_requests_total",
            description="Total number of user queries handled",
            unit="1",
        )
        request_latency_ms = meter.create_histogram(
            name="rag_request_latency_ms",
            description="End-to-end request latency",
            unit="ms",
        )
        answer_chars_hist = meter.create_histogram(
            name="rag_answer_chars",
            description="Length of generated answers",
            unit="1",
        )

        print("=== OBSERVABILITY: OpenTelemetry initialized ===", flush=True)
        return tracer, requests_counter, request_latency_ms, answer_chars_hist
    except Exception as e:
        print(f"[OTEL INIT WARNING] Telemetry disabled: {e}", flush=True)
        return None, None, None, None


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


tracer, requests_counter, request_latency_ms_hist, answer_chars_hist = setup_observability()
