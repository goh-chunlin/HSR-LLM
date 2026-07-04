import argparse

from lore_dump_utils import DEBUG_OUTPUT_JSONL, write_debug_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compact raw vs cleaned lore snapshots for debugging."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Export pages whose title or content contains this text.",
    )
    parser.add_argument(
        "--query-scope",
        choices=["title", "content", "raw", "any"],
        default="content",
        help="Choose whether to search page titles, cleaned content, raw wiki text, or any of them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Stop after writing this many matching pages.",
    )
    parser.add_argument(
        "--debug-output",
        help=(
            "Output JSONL path for raw and cleaned debug pages. "
            f"Defaults to {DEBUG_OUTPUT_JSONL}."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be a positive integer.")

    write_debug_pages(
        query=args.query,
        limit=args.limit,
        output_path=args.debug_output,
        query_scope=args.query_scope,
    )