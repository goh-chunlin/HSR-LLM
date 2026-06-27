import json

with open("hsr_v1_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("--- Printing all database chunks mentioning '83' ---")
for i, c in enumerate(chunks):
    if "83" in c["text"]:
        print(f"\n[Index {i}] Source Document: {c['title']}")
        print(f"Text Content:\n{c['text']}")