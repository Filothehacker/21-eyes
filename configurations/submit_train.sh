#!/bin/bash
#SBATCH --account=3160155
#SBATCH --job-name=finetune_yolo
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nv-1080:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=ai

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Directory: $(pwd)"

# First, check what modules are available
echo "Available modules:"
module avail

# Check for NVIDIA GPUs
nvidia-smi

# Run the training script
echo "Starting card detection training..."
python src/train_yolo_v1.py

echo "End: $(date)"