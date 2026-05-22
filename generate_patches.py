#!/usr/bin/env python3
"""
使用 Claude 生成 SWE-bench 问题的 patch

流程:
1. 读取数据集的 text 字段
2. 调用 claude -p 命令，让 Claude 直接生成 JSON 格式结果
3. 保存到 evaluate_result.json
"""

import json
import subprocess
import argparse
import re
from pathlib import Path
from datasets import load_from_disk
from typing import Optional


JSON_OUTPUT_PROMPT = """
CRITICAL: After solving the issue, you MUST output ONLY a JSON block, nothing else. The JSON must be the ONLY content in your response.

```json
{{
  "instance_id": "{instance_id}",
  "model_name_or_path": "claudecode-swe-bench",
  "model_patch": "<patch content here>"
}}
```

Rules:
1. Output ONLY the JSON block, no explanations
2. Replace "<patch content here>" with the diff content
3. model_patch must be a valid JSON string (escape newlines as \\n, quotes as \\")
4. Do not include <patch> tags in model_patch, just the raw diff text

Start your response with the JSON now:"""


def call_claude(text: str, instance_id: str) -> str:
    """调用 claude CLI 生成 patch 并写入 JSON 文件"""
    prompt = text + JSON_OUTPUT_PROMPT.format(instance_id=instance_id)

    cmd = [
        "claude",
        "-p", prompt,
        "--allow-dangerously-skip-permissions"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        print(result.stdout + result.stderr)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT: Command timed out after 5 minutes"
    except Exception as e:
        return f"ERROR: {str(e)}"


def parse_json_output(output: str, instance_id: str, model_name: str) -> Optional[dict]:
    """从 Claude 输出中解析 JSON"""
    # 移除<think>和</think>标签
    output = re.sub(r'<start_turn>.*?<end_turn>', '', output, flags=re.DOTALL)
    output = re.sub(r'<invoke_claude.*?</invoke_claude>', '', output, flags=re.DOTALL)
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)

    # 尝试多种模式匹配 JSON

    # 模式1: ```json ... ``` 代码块
    match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 模式2: 直接的 { ... } 对象
    match = re.search(r'(\{[\s\S]*?"instance_id"[\s\S]*?"model_patch"[\s\S]*?"model_name_or_path"[\s\S]*?\})', output)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 模式3: 尝试找最外层的 { }
    try:
        # 找到第一个 { 和对应的 }
        start = output.find('{')
        if start != -1:
            # 从 start 开始尝试递增地解析
            for end in range(len(output), start, -1):
                try:
                    parsed = json.loads(output[start:end])
                    if "instance_id" in parsed and "model_patch" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    continue
    except:
        pass

    return None


def generate_predictions(
    dataset_path: str,
    output_path: str,
    model_name: str = "claudecode-swe-bench",
    max_instances: Optional[int] = None,
    start_index: int = 0
) -> dict:
    """生成预测结果"""

    # 加载数据集
    dataset_dict = load_from_disk(dataset_path)

    predictions = []
    results = {
        "total_instances": 0,
        "completed": 0,
        "errors": 0,
        "results": []
    }

    # 确定要处理哪些 splits
    splits = ["test"] if "test" in dataset_dict else list(dataset_dict.keys())

    for split in splits:
        dataset = dataset_dict[split]

        # 确定处理的实例范围
        end_index = min(len(dataset), start_index + (max_instances or len(dataset)))

        for i in range(start_index, end_index):
            row = dataset[i]
            instance_id = row["instance_id"]
            text = row["text"]

            results["total_instances"] += 1
            print(f"[{results['total_instances']}/{max_instances or 'all'}] Processing: {instance_id}")

            # 调用 Claude
            output = call_claude(text, instance_id)

            # 解析 JSON 输出
            pred = parse_json_output(output, instance_id, model_name)

            if pred:
                predictions.append(pred)
                results["completed"] += 1
            else:
                # 如果没找到 JSON，创建空的 prediction
                predictions.append({
                    "instance_id": instance_id,
                    "model_name_or_path": model_name,
                    "model_patch": ""
                })
                results["errors"] += 1
                print(f"  Warning: Could not parse JSON from output")

            results["results"].append({
                "instance_id": instance_id,
                "output_length": len(output)
            })

            # 每 10 个保存一次中间结果
            if results["total_instances"] % 10 == 0:
                save_predictions(predictions, output_path)

    # 保存最终结果
    save_predictions(predictions, output_path)

    print(f"\n=== Summary ===")
    print(f"Total: {results['total_instances']}")
    print(f"Completed: {results['completed']}")
    print(f"Errors: {results['errors']}")

    return results


def save_predictions(predictions: list, output_path: str):
    """保存预测结果到 JSON 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./base_datasets/SWE-bench_Verified__style-3__fs-oracle",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluate_result.json",
        help="Output path for predictions JSON"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="claudecode-swe-bench",
        help="Model name to record in results (default: claudecode-swe-bench)"
    )
    parser.add_argument(
        "-n", "--num",
        type=str,
        default="1",
        help='Number of instances to process: "1" (default), "all", or an integer like "10"'
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index in dataset"
    )
    return parser.parse_args()


def parse_num(num_str: str) -> Optional[int]:
    """解析 -n 参数"""
    if num_str.lower() == "all":
        return None
    return int(num_str)


if __name__ == "__main__":
    args = parse_args()

    max_instances = parse_num(args.num)

    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"Model: {args.model_name}")
    print(f"Num: {args.num if max_instances is None else args.num}")
    print()

    generate_predictions(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        model_name=args.model_name,
        max_instances=max_instances,
        start_index=args.start_index
    )