import argparse
from lore_dump_utils import OUTPUT_JSONL, write_clean_lore_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream the HSR Fandom XML dump into cleaned lore JSONL."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after writing this many matched lore pages. Useful for small debug runs.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output JSONL path for the cleaned lore extraction. "
            f"Defaults to {OUTPUT_JSONL}, or to inspect_outputs when --limit is used."
        ),
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer.")

    write_clean_lore_jsonl(limit=args.limit, output_path=args.output)