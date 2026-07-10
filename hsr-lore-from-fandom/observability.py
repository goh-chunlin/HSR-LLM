import hashlib
import os
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer
from opentelemetry.metrics import Counter, Histogram


OTEL_CAPTURE_CONTENT = os.getenv("HSR_OTEL_CAPTURE_CONTENT", "false").strip().lower() == "true"


def setup_observability() -> tuple[Tracer, Counter, Histogram, Histogram]:
    """
    Initialize OpenTelemetry for traces and metrics.
    """
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


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


tracer, requests_counter, request_latency_ms_hist, answer_chars_hist = setup_observability()
