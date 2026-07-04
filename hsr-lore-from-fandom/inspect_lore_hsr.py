import argparse

from lore_dump_utils import DEBUG_OUTPUT_JSONL, write_debug_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compact raw vs cleaned lore snapshots for debugging."
    )
    parser.add_argument(
        "--inspect-title",
        required=True,
        help="Export only pages whose title contains this text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Stop after writing this many matching pages.",
    )
    parser.add_argument(
        "--debug-output",
        default=DEBUG_OUTPUT_JSONL,
        help="Output JSONL path for raw and cleaned debug pages.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be a positive integer.")

    write_debug_pages(
        title_query=args.inspect_title,
        limit=args.limit,
        output_path=args.debug_output,
    )