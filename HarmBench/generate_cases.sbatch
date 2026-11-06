#!/bin/bash
#SBATCH --job-name=generate_cases
#SBATCH --output=logs/generate_cases_%j.out
#SBATCH --error=logs/generate_cases_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu 

# 设置 CUDA 环境
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ywjenv

export RAY_local_mode=True     # 解决 Ray 超时最有效的办法
export PYTHONUNBUFFERED=1      # 让日志实时输出，方便观察卡在哪里
nvidia-smi
echo "检查GPU状态..."
python -c "import torch; print(f'第一次检查: {torch.cuda.is_available()}')"

cd /data/users/fanxin/yiwenjing/HarmBench

python generate_test_cases.py \
  --method_name AutoDAN \
  --experiment_name mistral_7b_v2 \
  --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --save_dir ./results/adaptive_attnroute/mistral_autodan_harmbench
sleep 10

# Vicuna‑7B‑AttnRoute

python generate_test_cases.py \
  --method_name AutoDAN \
  --experiment_name vicuna_7b_v1_5 \
  --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --save_dir ./results/adaptive_attnroute/vicuna_autodan_harmbench
sleep 10

# Llama‑3‑8B‑AttnRoute

echo "作业完成时间: $(date)"