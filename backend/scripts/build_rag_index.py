"""
Build or query the RAG vector index for the SA Income Tax Act.

Usage (run from backend/):
  python scripts/build_rag_index.py --build
  python scripts/build_rag_index.py --query "What is a deductible expense?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from app.services.rag_service import build_index, query


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SA Income Tax Act -- RAG Engine (local Hugging Face)"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build / rebuild the vector index from the law text file.",
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Ask a question against the indexed law document.",
    )
    args = parser.parse_args()

    if args.build:
        build_index()

    if args.query:
        print("\nRetrieving and generating answer ...\n")
        answer = query(args.query)
        print(answer)

    if not args.build and not args.query:
        parser.print_help()


def _suppress_shutdown_resource_tracker_error(unraisable):
    """Suppress known Windows/multiprocess shutdown bug (RLock teardown order)."""
    if unraisable.exc_type is AttributeError and "_recursion_count" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)


if __name__ == "__main__":
    sys.unraisablehook = _suppress_shutdown_resource_tracker_error
    main()
