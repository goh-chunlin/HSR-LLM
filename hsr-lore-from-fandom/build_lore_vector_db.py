import os
# Force multi-threading libraries to run on a single thread to stop the memory crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import sys
from typing import Final, TypedDict, cast
from sentence_transformers import SentenceTransformer
import faiss

# Configuration
INPUT_JSONL: Final[str] = "hsr_v1_raw_lore.jsonl"
OUTPUT_FAISS: Final[str] = "my_hsr_1.0_index.faiss"
OUTPUT_CHUNKS: Final[str] = "hsr_v1_chunks.json"

CHUNK_SIZE: Final[int] = 500  # Character length per chunk
CHUNK_OVERLAP: Final[int] = 100  # Overlap to prevent splitting context in half

MIN_SUPPORTED_PYTHON: Final[tuple[int, int]] = (3, 10)
MAX_SUPPORTED_PYTHON: Final[tuple[int, int]] = (3, 13)

BANNED_PREFIXES = (
    "hoyolab/", 
    "version/", 
    "media/", 
    "update/", 
    "news/", 
    "notice/", 
    "community/"
)

class ChunkMetadata(TypedDict):
    title: str
    text: str


def validate_runtime() -> bool:
    """Return False for Python versions known to crash with faiss-cpu wheels."""
    current = (sys.version_info.major, sys.version_info.minor)

    if current < MIN_SUPPORTED_PYTHON:
        print(
            f"Error: Python {sys.version_info.major}.{sys.version_info.minor} is too old. "
            f"Use Python {MIN_SUPPORTED_PYTHON[0]}.{MIN_SUPPORTED_PYTHON[1]} or newer."
        )
        return False

    if current > MAX_SUPPORTED_PYTHON:
        print(
            "Error: Python 3.14+ is not reliable with current faiss-cpu wheels on macOS. "
            "Please run this script in a Python 3.11 or 3.12 environment."
        )
        return False

    return True


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into overlapping chunks to preserve semantic context."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def build_database() -> None:
    if not validate_runtime():
        return

    # Prevent tokenizers/joblib worker over-allocation noise in local runs.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not os.path.exists(INPUT_JSONL):
        print(f"Error: Cannot find {INPUT_JSONL}. Run your extractor first.")
        return

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    # This model fits completely inside memory and runs lightning fast on CPU
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    all_chunks: list[str] = []
    all_metadata: list[ChunkMetadata] = []
    
    print("Reading and chunking lore data...")
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if not isinstance(data, dict):
                continue

            record = cast(dict[str, object], data)
            title_value = record.get("title", "Unknown Source")
            content_value = record.get("content", "")
            title = title_value if isinstance(title_value, str) else "Unknown Source"
            content = content_value if isinstance(content_value, str) else ""
            
            # Skip empty entries
            if not content.strip():
                continue

            # CRITICAL: Drop out-of-universe marketing noise before it touches the chunker
            title_low = title.lower().strip()
            if title_low.startswith(BANNED_PREFIXES) or "hoyolab" in title_low:
                continue  # Skip this file completely, do not chunk or index it
                
            # Break down large wiki pages into manageable pieces
            text_chunks = chunk_text(content)
            for chunk in text_chunks:
                context_anchored_text = f"Source Document: {title}\nContent:\n{chunk}"

                all_chunks.append(context_anchored_text)
                # Keep track of where this piece of info came from
                all_metadata.append({
                    "title": title,
                    "text": chunk
                })

    print(f"Total text chunks created: {len(all_chunks)}")
    if not all_chunks:
        print("Error: No chunks were produced from the input JSONL.")
        return

    print("Generating dense vector embeddings... (This will utilize your Mac's CPU/GPU efficiently)")
    
    # Encode text chunks to a numpy matrix
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)  # pyright: ignore[reportUnknownMemberType]
    
    # MiniLM-L6-v2 outputs 384-dimensional vectors
    dimension = embeddings.shape[1]
    
    print(f"Initializing FAISS index with dimension layout: {dimension}")
    # Using IndexFlatIP (Inner Product) paired with normalized vectors gives Cosine Similarity
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize vectors for accurate cosine similarity matching
    faiss.normalize_L2(embeddings)
    
    # Inject vectors into the index structure
    index.add(embeddings)
    
    # Save the vector index file
    print(f"Saving vector database binary to: {OUTPUT_FAISS}")
    faiss.write_index(index, OUTPUT_FAISS)
    
    # Save the raw string chunks matching the vector positions
    print(f"Saving raw text lookup database to: {OUTPUT_CHUNKS}")
    with open(OUTPUT_CHUNKS, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
        
    print("\nInitialization completely successful! Ready for Hugging Face deployment.")

if __name__ == "__main__":
    build_database()