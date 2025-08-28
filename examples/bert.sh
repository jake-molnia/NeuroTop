#!/bin/bash
#SBATCH -N 1                          
#SBATCH --ntasks-per-node=1      
#SBATCH --mem=256g                    
#SBATCH --gres=gpu:1                  
#SBATCH -p short                    
#SBATCH -t 24:00:00              
#SBATCH -J "bert_wikipedia"
#SBATCH --output=bert_%A_%a.out 
#SBATCH --error=bert_%A_%a.err   
#SBATCH --mail-type=BEGIN,END,FAIL                             
#SBATCH --mail-user=jrmolnia@wpi.edu                          

# Exit on any error
set -e

echo "Starting job at $(date)"
echo "Running on node: $SLURM_NODELIST"

# Check if Python script exists
if [[ ! -f "bert_wikipedia.py" ]]; then
    echo "ERROR: bert_wikipedia.py not found!"
    exit 1
fi

# Load modules and check they loaded
echo "Loading modules..."
module load python || { echo "Failed to load python"; exit 1; }
module load uv || { echo "Failed to load uv"; exit 1; }
module load cudnn8.9-cuda12.3/8.9.7.29 || { echo "Failed to load cudnn"; exit 1; }

# Quick GPU check
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: No GPU available!"
    exit 1
fi

echo "GPU found: $(nvidia-smi -L)"

# Run the script
echo "Starting bert_wikipedia.py at $(date)"
uv run bert_wikipedia.py

echo "Job completed at $(date)"