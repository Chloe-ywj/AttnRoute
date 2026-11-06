#!/bin/bash
#SBATCH --job-name=run_attnroute
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_attnroute_%j.out
#SBATCH --error=logs/run_attnroute_%j.err

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ywjenv

export NCCL_DEBUG=INFO  # 调试用
export PYTORCH_ALLOC_CONF=expandable_segments:True

nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

cd /data/users/fanxin/yiwenjing/AttnRoute

python run_attnroute.py \
  --model_name mistral \
  --hparams_fname Mistral-7B-Instruct-v0.2.json \
  --out_name module_ablation/static-rank128/Mistral-7B-Instruct-v0.2 \
  --model_path "../models/mistral/Mistral-7B-Instruct-v0.2" \
  --edit_data_file "data/merged_output_mistral.csv" \
  --fixed_target_layers "12,13,14,16,20" \
  --norm_ref_file "./results/module_ablation/Mistral-7B-Instruct-v0.2/update_norms.json" \
  --edit_mode attnroute \
  --cov_mode full \
  --rank_mode "fixed" \
  --fixed_rank 128
sleep 10

echo ""
echo "=== 作业结束: $(date) ==="

