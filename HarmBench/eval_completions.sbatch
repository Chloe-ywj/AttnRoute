#!/bin/bash
#SBATCH --job-name=eval_completions
#SBATCH --output=logs/eval_completions_%j.out
#SBATCH --error=logs/eval_completions_%j.err
#SBATCH --time=12:00:00
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

#export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True
nvidia-smi

cd /data/users/fanxin/yiwenjing/HarmBench

# 运行前自检
echo "Checking GPU status..."
python -c "import torch; print(f'>>> CUDA is_available: {torch.cuda.is_available()}')"

# 执行评估程序
python batch_asr.py

echo "作业完成时间: $(date)"
