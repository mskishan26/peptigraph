#!/bin/bash
#SBATCH --partition=gpu                             # Use the main GPU queue
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1                           # Accept any GPU (A100 or H200)
#SBATCH --time=08:00:00                             # Job runtime   (needs to be updated, need to find the approx time for a file and then change this)
#SBATCH --job-name=ingestion_pipeline               # Job name
#SBATCH --mem=16GB                                  # Job memory
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --cpus-per-task=8                           # Number of cpus per task
#SBATCH --output=run_peptide%j.out          # Output file
#SBATCH --error=run_peptide%j.err           # Error file
#SBATCH --mail-user=sathishbabu.ki@northeastern.edu  # Email address
#SBATCH --mail-type=ALL                             # Mail type

# 1. Load modules
module load anaconda3/2024.06
module load cuda/12.1.1

# 1.1. Set up environment variables
# Python path
export WANDB_MODE="offline"

# 1.2. Function to monitor GPU usage
monitor_gpu_usage() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="./gpu_usage/gpu_usage_${timestamp}.csv"

    mkdir -p gpu_usage

    # CSV Header
    echo "timestamp,memory_used_mb,memory_total_mb,memory_util_pct,gpu_util_pct" > "$log_file"

    echo "Monitoring GPU usage (CSV). Logging to: $log_file"

    while true; do
        local now=$(date +"%Y-%m-%d %H:%M:%S")

        # Query nvidia-smi for a machine-readable output
        local line=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.memory,utilization.gpu \
                                --format=csv,noheader,nounits)

        # line format example: "1024, 11178, 10, 25"
        echo "$now,$line" >> "$log_file"

        sleep 10
    done &
    GPU_MONITOR_PID=$!
}

# Start GPU monitoring
monitor_gpu_usage

# Ensure GPU monitoring stops when the script exits
trap "kill $GPU_MONITOR_PID" EXIT

# 2. Activate marker environment
source activate /scratch/sathishbabu.ki/dl_peptides

# DO NOT FORGET: you need to be careful about this: change/resume the resume_from each time
python train.py --epochs 300 --resume_from /home/sathishbabu.ki/peptides/checkpoints/transformer_h128_l4_heads8_laplacian_20251203_023521/latest_checkpoint.pt --use_wandb

# 6. Deactivate environment
conda deactivate
