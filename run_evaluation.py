#!/usr/bin/env python3
"""
运行 SWE-bench 评估脚本

流程:
1. 预构建本地 Docker 镜像（默认 arm64）
2. 运行评估（评估结束后保留镜像，不清理）
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> int:
    print(f"Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


def prepare_images(
    dataset_name: str,
    max_workers: int,
    instance_ids: list[str] | None = None,
    namespace: str = "none",
    arch: str = "arm64",
) -> int:
    """预构建本地 Docker 镜像"""
    cmd = [
        "python", "-m", "swebench.harness.prepare_images",
        "--dataset_name", dataset_name,
        "--split", "test",
        "--max_workers", str(max_workers),
        "--namespace", namespace,
        "--tag", "latest",
        "--env_image_tag", "latest",
        "--arch", arch,
    ]
    if instance_ids:
        cmd.extend(["--instance_ids"] + instance_ids)
    return run_command(cmd)


def run_evaluation(
    dataset_name: str,
    predictions_path: str,
    max_workers: int = 12,
    run_id: str = "evaluation",
    instance_ids: list[str] | None = None,
    namespace: str = "none",
    cache_level: str = "instance",
    clean: bool = False,
    arch: str = "arm64",
) -> int:
    """运行评估"""
    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", predictions_path,
        "--max_workers", str(max_workers),
        "--run_id", run_id,
        "--namespace", namespace,
        "--cache_level", cache_level,
        "--clean", str(clean),
        "--arch", arch,
    ]
    if instance_ids:
        cmd.extend(["--instance_ids"] + instance_ids)
    return run_command(cmd)


def load_instance_ids(predictions_path: str) -> list[str]:
    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)
    return [
        item["instance_id"]
        for item in predictions
        if isinstance(item, dict) and item.get("instance_id")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="./princeton-nlp___swe-bench_verified",
        help="数据集名称",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="./predict_result.json",
        help="预测结果文件路径",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="并行工作进程数",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="evaluation",
        help="运行标识符",
    )
    parser.add_argument(
        "--instance_ids",
        type=str,
        nargs="+",
        default=None,
        help="指定要评估的 instance_id 列表（默认使用 predictions 文件中的全部 instance）",
    )
    parser.add_argument(
        "--skip_prepare",
        action="store_true",
        help="跳过预构建镜像步骤",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="none",
        help='Docker 镜像 namespace，本地构建使用 "none"',
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="arm64",
        choices=["x86_64", "arm64"],
        help="Docker 镜像 CPU 架构（Mac Studio 默认 arm64）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not Path(args.predictions_path).exists():
        print(f"Error: Predictions file not found: {args.predictions_path}")
        sys.exit(1)

    instance_ids = args.instance_ids or load_instance_ids(args.predictions_path)

    print(f"Predictions file: {args.predictions_path}")
    print(f"Instances: {len(instance_ids)}")
    print(f"Max workers: {args.max_workers}")
    print(f"Run ID: {args.run_id}")
    print(f"Namespace: {args.namespace}")
    print(f"Arch: {args.arch}")
    print()

    if not args.skip_prepare:
        print("=== Step 1: Prepare Docker images ===")
        rc = prepare_images(
            dataset_name=args.dataset_name,
            max_workers=args.max_workers,
            instance_ids=instance_ids,
            namespace=args.namespace,
            arch=args.arch,
        )
        if rc != 0:
            sys.exit(rc)
        print()

    print("=== Step 2: Run evaluation ===")
    rc = run_evaluation(
        dataset_name=args.dataset_name,
        predictions_path=args.predictions_path,
        max_workers=args.max_workers,
        run_id=args.run_id,
        instance_ids=instance_ids,
        namespace=args.namespace,
        cache_level="instance",
        clean=False,
        arch=args.arch,
    )
    sys.exit(rc)
