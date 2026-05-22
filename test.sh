export GITHUB_TOKEN=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
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path princeton-nlp/SWE-bench_Verified \
    --output_dir ./base_datasets \
    --splits test \
    --prompt_style style-3 \
    --file_source oracle


python -m swebench.harness.run_evaluation \
    --predictions_path gold \
    --max_workers 1 \
    --instance_ids sympy__sympy-20590 \
    --run_id validate-gold


python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path evaluate_result.json \
    --max_workers 12 \
    --run_id full_evaluation