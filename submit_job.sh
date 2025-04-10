#!/bin/bash
#SBATCH --job-name=card_detection
#SBATCH --output=card_detection_%j.out
#SBATCH --error=card_detection_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=384G
#SBATCH --partition=gpu
#SBATCH --constraint=Ampere  # Add this if you know the GPUs are NVIDIA Ampere architecture

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Directory: $(pwd)"

# Load necessary modules (adjust these according to your HPC setup)
module purge
module load cuda/11.7
module load python/3.9

# Activate virtual environment
echo "Activating virtual environment venv-cv..."
source venv-cv/bin/activate

# Check Python and pip versions
which python
python --version
which pip
pip --version

# Install required packages if not already in your virtual environment
# Uncomment the following line if you need to install packages
pip install sklearn 

# Create models directory
mkdir -p models

# Run the training script
echo "Starting card detection training..."
python train_card_detector.py --yaml hpc_data.yaml --model-size m --epochs 100 --batch 128 --imgsz 640 --device 0,1,2,3

# Run the testing script after training
echo "Starting card detection testing..."
python test_card_detector.py --yaml hpc_data.yaml --model models/best_card_detector.pt --imgsz 640 --conf 0.25 --batch 64 --device 0

echo "End: $(date)"