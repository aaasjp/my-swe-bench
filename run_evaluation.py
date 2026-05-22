#!/usr/bin/env python3
"""
运行 SWE-bench 评估脚本

根据预测结果评估模型生成的 patch 是否能正确解决问题
"""

import argparse
import subprocess
import json
from pathlib import Path


def run_evaluation(
    dataset_name: str,
    predictions_path: str,
    max_workers: int = 12,
    run_id: str = "evaluation",
    instance_ids: list = None,
) -> dict:
    """运行评估"""

    # 构建命令
    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", predictions_path,
        "--max_workers", str(max_workers),
        "--run_id", run_id,
    ]

    # 如果指定了 instance_ids，添加参数
    if instance_ids:
        cmd.extend(["--instance_ids"] + instance_ids)

    print(f"Running: {' '.join(cmd)}")
    print()

    # 执行命令
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="./princeton-nlp___swe-bench_verified",
        help="数据集名称"
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="./predict_result.json",
        help="预测结果文件路径"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="并行工作进程数"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="evaluation",
        help="运行标识符"
    )
    parser.add_argument(
        "--instance_ids",
        type=str,
        nargs="+",
        default=None,
        help="指定要评估的 instance_id 列表（可选）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 验证 predictions 文件存在
    if not Path(args.predictions_path).exists():
        print(f"Error: Predictions file not found: {args.predictions_path}")
        exit(1)

    # 读取并验证 predictions 文件
    with open(args.predictions_path, 'r') as f:
        predictions = json.load(f)

    print(f"Predictions file: {args.predictions_path}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Max workers: {args.max_workers}")
    print(f"Run ID: {args.run_id}")
    print()

    run_evaluation(
        dataset_name=args.dataset_name,
        predictions_path=args.predictions_path,
        max_workers=args.max_workers,
        run_id=args.run_id,
        instance_ids=args.instance_ids,
    )