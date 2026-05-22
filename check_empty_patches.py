#!/usr/bin/env python3
"""
检查 predict_result.json 中 model_patch 为空的条目数量
"""

import argparse
import json
from pathlib import Path


def is_empty_patch(patch) -> bool:
    if patch is None:
        return True
    if not isinstance(patch, str):
        return True
    return patch.strip() == ""


def check_empty_patches(predictions_path: str, list_empty: bool = False) -> dict:
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(path, encoding="utf-8") as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        raise ValueError(f"Expected a JSON array in {predictions_path}")

    empty_ids = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        if is_empty_patch(item.get("model_patch")):
            instance_id = item.get("instance_id", "<unknown>")
            empty_ids.append(instance_id)

    total = len(predictions)
    empty_count = len(empty_ids)
    non_empty_count = total - empty_count

    result = {
        "predictions_path": str(path),
        "total": total,
        "empty_count": empty_count,
        "non_empty_count": non_empty_count,
        "empty_ratio": empty_count / total if total else 0.0,
        "empty_instance_ids": empty_ids,
    }

    print(f"Predictions file: {path}")
    print(f"Total: {total}")
    print(f"Empty patches: {empty_count}")
    print(f"Non-empty patches: {non_empty_count}")
    print(f"Empty ratio: {result['empty_ratio']:.2%}")

    if list_empty and empty_ids:
        print("\nEmpty instance_ids:")
        for instance_id in empty_ids:
            print(f"  - {instance_id}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="./predict_result.json",
        help="预测结果文件路径（默认: ./predict_result.json）",
    )
    parser.add_argument(
        "--list-empty",
        action="store_true",
        help="列出所有 patch 为空的 instance_id",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_empty_patches(
        predictions_path=args.predictions_path,
        list_empty=args.list_empty,
    )
