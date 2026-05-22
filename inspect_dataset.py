#!/usr/bin/env python3
"""Inspect SWE-bench text datasets saved with HuggingFace save_to_disk."""

from __future__ import annotations

import argparse
import statistics
import textwrap
from pathlib import Path

from datasets import load_from_disk


DEFAULT_DATASET = "./base_datasets/SWE-bench_Verified__style-3__fs-oracle"
PREVIEW_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "problem_statement",
    "text",
    "patch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to a dataset directory created by create_text_dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to inspect",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based row index to display in detail",
    )
    parser.add_argument(
        "--instance_id",
        type=str,
        default=None,
        help="Display the row with this instance_id instead of --index",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=800,
        help="Number of characters to show for long string fields",
    )
    parser.add_argument(
        "--show_full",
        action="store_true",
        help="Print full text and patch instead of truncated previews",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print text length statistics for the whole split",
    )
    return parser.parse_args()


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... [{len(text) - limit} chars omitted] ...\n\n"
        + text[-half:]
    )


def resolve_index(dataset, index: int, instance_id: str | None) -> int:
    if instance_id is None:
        if index < 0 or index >= len(dataset):
            raise IndexError(
                f"index {index} out of range for split with {len(dataset)} rows"
            )
        return index

    for i, row in enumerate(dataset):
        if row["instance_id"] == instance_id:
            return i
    raise ValueError(f"instance_id not found: {instance_id}")


def print_summary(dataset, split: str, dataset_path: str) -> None:
    print("=== Dataset Summary ===")
    print(f"path:   {Path(dataset_path).resolve()}")
    print(f"split:  {split}")
    print(f"rows:   {len(dataset)}")
    print(f"fields: {dataset.column_names}")
    print()


def print_stats(dataset) -> None:
    text_lens = [len(row["text"]) for row in dataset]
    patch_lens = [len(row["patch"]) for row in dataset]
    print("=== Length Statistics ===")
    print(
        "text:  "
        f"min={min(text_lens)}, max={max(text_lens)}, "
        f"avg={int(statistics.mean(text_lens))}, "
        f"median={int(statistics.median(text_lens))}"
    )
    print(
        "patch: "
        f"min={min(patch_lens)}, max={max(patch_lens)}, "
        f"avg={int(statistics.mean(patch_lens))}, "
        f"median={int(statistics.median(patch_lens))}"
    )
    print()


def print_row(row: dict, head: int, show_full: bool) -> None:
    print("=== Sample Detail ===")
    for field in PREVIEW_FIELDS:
        value = row.get(field, "")
        if field in {"text", "patch", "problem_statement"}:
            print(f"--- {field} ({len(value)} chars) ---")
            body = value if show_full else truncate(value, head)
            print(body)
        else:
            print(f"{field}: {value}")
        print()


def main() -> None:
    args = parse_args()

    dataset_dict = load_from_disk(args.dataset_path)
    if args.split not in dataset_dict:
        available = ", ".join(dataset_dict.keys())
        raise KeyError(f"split '{args.split}' not found; available: {available}")

    dataset = dataset_dict[args.split]
    print_summary(dataset, args.split, args.dataset_path)

    if args.stats:
        print_stats(dataset)

    index = resolve_index(dataset, args.index, args.instance_id)
    row = dataset[index]
    print(f"showing index={index}, instance_id={row['instance_id']}")
    print()
    print_row(row, args.head, args.show_full)


if __name__ == "__main__":
    main()
