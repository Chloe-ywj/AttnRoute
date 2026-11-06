#!/bin/bash
#SBATCH --job-name=gen_completions
#SBATCH --output=logs/gen_completions_%j.out
#SBATCH --error=logs/gen_completions_%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ywjenv

export NCCL_DEBUG=INFO  # 调试用
export PYTORCH_ALLOC_CONF=expandable_segments:True


nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
cd /data/users/fanxin/yiwenjing/HarmBench
#/data/users/fanxin/yiwenjing/HarmBench/results/HEx-PHI/llama/pair/convert/HEx-PHI-llama-3.json
#./results/harmbench/llama/extracted_samples/pair/samples_converted.json

python generate_completions.py \
    --model_name llama3_8b \
    --models_config_file ./configs/model_configs/models.yaml \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test_cases_path ./results/HEx-PHI/llama/extracted_samples/gcg/samples_converted.json \
    --save_path ./results/module_ablation/llama-HEx-PHI_gcg_wo.json \
    --max_new_tokens 512 \
    --generate_with_vllm \
    --incremental_update
sleep 10

python generate_completions.py \
    --model_name llama3_8b \
    --models_config_file ./configs/model_configs/models.yaml \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test_cases_path ./results/HEx-PHI/llama/extracted_samples/autodan/samples_converted.json \
    --save_path ./results/module_ablation/llama-HEx-PHI_autodan_wo.json \
    --max_new_tokens 512 \
    --generate_with_vllm \
    --incremental_update
sleep 10

python generate_completions.py \
    --model_name llama3_8b \
    --models_config_file ./configs/model_configs/models.yaml \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test_cases_path ./results/HEx-PHI/llama/extracted_samples/pair/samples_converted.json \
    --save_path ./results/module_ablation/llama-HEx-PHI_pair_wo.json \
    --max_new_tokens 512 \
    --generate_with_vllm \
    --incremental_update
sleep 10

python generate_completions.py \
    --model_name llama3_8b \
    --models_config_file ./configs/model_configs/models.yaml \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test_cases_path ./results/harmbench/llama/extracted_samples/gcg/samples_converted.json \
    --save_path ./results/module_ablation/llama-harmbench_gcg_wo.json \
    --max_new_tokens 512 \
    --generate_with_vllm \
    --incremental_update
sleep 10

python generate_completions.py \
    --model_name llama3_8b \
    --models_config_file ./configs/model_configs/models.yaml \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test_cases_path ./results/harmbench/llama/extracted_samples/autodan/samples_converted.json \
    --save_path ./results/module_ablation/llama-harmbench_autodan_wo.json \
    --max_new_tokens 512 \
    --generate_with_vllm \
    --incremental_update
sleep 10

python generate_completions.py \
    --model_name llama3_8b \
    --models_config_file ./configs/model_configs/models.yaml \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test_cases_path ./results/harmbench/llama/extracted_samples/pair/samples_converted.json \
    --save_path ./results/module_ablation/llama-harmbench_pair_wo.json \
    --max_new_tokens 512 \
    --generate_with_vllm \
    --incremental_update
sleep 10

python generate_completions.py \
    --model_name llama3_8b \
    --models_config_file ./configs/model_configs/models.yaml \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test_cases_path ./results/advbench/llama/extracted_samples/pair/samples_converted.json \
    --save_path ./results/module_ablation/llama-advbench_pair_wo.json \
    --max_new_tokens 512 \
    --generate_with_vllm \
    --incremental_update
sleep 10

python generate_completions.py \
    --model_name llama3_8b \
    --models_config_file ./configs/model_configs/models.yaml \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test_cases_path ./results/advbench/llama/extracted_samples/autodan/samples_converted.json \
    --save_path ./results/module_ablation/llama-advbench_autodan_wo.json \
    --max_new_tokens 512 \
    --generate_with_vllm \
    --incremental_update
sleep 10

python generate_completions.py \
    --model_name llama3_8b \
    --models_config_file ./configs/model_configs/models.yaml \
    --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --test_cases_path ./results/advbench/llama/extracted_samples/gcg/samples_converted.json \
    --save_path ./results/module_ablation/llama-advbench_gcg_wo.json \
    --max_new_tokens 512 \
    --generate_with_vllm \
    --incremental_update
sleep 10

echo "作业完成时间: $(date)"
