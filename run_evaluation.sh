#!/bin/bash
set -euo pipefail

DATASET_NAME="./princeton-nlp___swe-bench_verified"
PREDICTIONS_PATH="predict_result.json"
MAX_WORKERS=12
RUN_ID="test_evaluation"
ARCH="arm64"

# Step 1: 预构建本地 Docker 镜像（arm64）
python -m swebench.harness.prepare_images \
    --dataset_name "${DATASET_NAME}" \
    --split test \
    --max_workers "${MAX_WORKERS}" \
    --namespace none \
    --tag latest \
    --env_image_tag latest \
    --arch "${ARCH}"

# Step 2: 运行评估（本地 arm64 镜像，评估结束后不清理）
python -m swebench.harness.run_evaluation \
    --dataset_name "${DATASET_NAME}" \
    --predictions_path "${PREDICTIONS_PATH}" \
    --max_workers "${MAX_WORKERS}" \
    --run_id "${RUN_ID}" \
    --namespace none \
    --cache_level instance \
    --clean False \
    --arch "${ARCH}"
