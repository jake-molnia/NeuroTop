#!/bin/bash
#SBATCH -N 1                          
#SBATCH --ntasks-per-node=1      
#SBATCH --mem=256g                    
#SBATCH --gres=gpu:1                  
#SBATCH -p short                    
#SBATCH -t 24:00:00              
#SBATCH -J "bert_train_ablate"
#SBATCH --output=bert_train_ablate%A.out 
#SBATCH --error=bert_train_ablate%A.err   
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=jrmolnia@wpi.edu

# Exit on any error
set -e

# --- Environment Setup ---
echo "Starting job at $(date)"
echo "Running on node: $SLURM_NODELIST"

# Suppress HuggingFace Tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Load necessary modules
echo "Loading modules..."
module load python
module load uv
module load cudnn8.9-cuda12.3/8.9.7.29

# Verify GPU is available
echo "GPU found: $(nvidia-smi -L)"

# Define a common output directory for this run
OUTPUT_DIR="./outputs/bert_train_and_ablate_run"
echo "Using output directory: ${OUTPUT_DIR}"


# --- Step 1: Train the Model ---
echo -e "\n--- STEP 1: TRAINING MODEL FOR 5 EPOCHS ---"
uv run examples/bert.py train \
    --dataset-name cola \
    --epochs 500 \
    --output-dir "${OUTPUT_DIR}"

echo "Training complete. Model and analysis data saved to ${OUTPUT_DIR}"


# --- Step 2: Ablate the Trained Model ---
echo -e "\n--- STEP 2: RUNNING ITERATIVE ABLATION ---"
uv run examples/bert.py ablate \
    --model-path "${OUTPUT_DIR}/final_model" \
    --analysis-path "${OUTPUT_DIR}/topology_evolution.npz" \
    --dataset-name cola \
    --strategy iterative \
    --component attention \
    --step-size 1 \
    --output-dir "${OUTPUT_DIR}"

echo "Iterative ablation complete. Performance plot saved to ${OUTPUT_DIR}"
echo "Job finished at $(date)"