import json

from rag_runtime import RuntimeState

runtime = RuntimeState()
runtime.initialize()

print("--- Runtime summary ---")
print(f"Overlay source: {runtime.overlay_source}")
print(f"Overlay path: {runtime.overlay_path}")
print(f"Overlay records: {runtime.overlay_record_count}")
print(f"Overlay replacements: {runtime.overlay_replacement_count}")
print(f"Metadata records: {len(runtime.text_metadata)}")

with open(runtime.chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("--- Printing all database chunks mentioning '83' ---")
for i, c in enumerate(chunks):
    if "83" in c["text"]:
        print(f"\n[Index {i}] Source Document: {c['title']}")
        print(f"Text Content:\n{c['text']}")