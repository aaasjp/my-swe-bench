#!/usr/bin/env python3
"""调试脚本：查看 Claude 的实际输出"""

import subprocess
from datasets import load_from_disk

dataset_dict = load_from_disk('./base_datasets/SWE-bench_Verified__style-3__fs-oracle')
text = dataset_dict['test'][0]['text']
instance_id = dataset_dict['test'][0]['instance_id']

# 保存 prompt
with open('/tmp/claude_prompt.txt', 'w') as f:
    f.write(text)
print(f"Saved text to /tmp/claude_prompt.txt")

# 截取 text 的前 1000 字符作为测试 prompt
test_prompt = text[:1000]

cmd = [
    "claude",
    "-p", test_prompt + "\n\nPlease respond with a simple test JSON: {\"test\": true}",
    "--allow-dangerously-skip-permissions"
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
print("=== STDOUT ===")
print(result.stdout[:2000] if len(result.stdout) > 2000 else result.stdout)
print("=== STDERR ===")
print(result.stderr[:2000] if len(result.stderr) > 2000 else result.stderr)