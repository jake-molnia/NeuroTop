#!/bin/bash
#SBATCH -N 1                          
#SBATCH --ntasks-per-node=1      
#SBATCH --mem=256g                    
#SBATCH --gres=gpu:1                  
#SBATCH -p short                    
#SBATCH -t 24:00:00              
#SBATCH -J "glue_experiments"
#SBATCH --output=glue_experiments_%A.out 
#SBATCH --error=glue_experiments_%A.err   
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=jrmolnia@wpi.edu

# Exit on any error
set -e

# --- Environment Setup ---
echo "Starting GLUE comprehensive experiments at $(date)"
echo "Running on node: $SLURM_NODELIST"

# Suppress HuggingFace Tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Load necessary modules
echo "Loading modules..."
module load python
module load uv
module load cudnn8.9-cuda12.3/8.9.7.29

export UV_LINK_MODE=copy

# Verify GPU is available
echo "GPU found: $(nvidia-smi -L)"

# Define output directories
RESULTS_DIR="./results"
MODELS_DIR="./trained_models"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "Trained models cached in: ${MODELS_DIR}"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${MODELS_DIR}"

# Define experimental parameters
DATASETS=("cola" "qqp" "stsb" "mrpc" "rte")
MODELS=("bert-base-uncased" "bert-large-uncased")
DISTANCE_METRICS=("euclidean" "manhattan" "cosine")
EPOCHS=50

echo "=== EXPERIMENTAL MATRIX ==="
echo "Datasets: ${DATASETS[@]}"
echo "Models: ${MODELS[@]}" 
echo "Pruning Methods: Random, RF (${DISTANCE_METRICS[@]})"
echo "Total Experiments: $((${#DATASETS[@]} * ${#MODELS[@]} * 4)) experiments"
echo "=========================="

# Function to run single experiment with error handling
run_single_experiment() {
    local dataset=$1
    local model=$2
    local method=$3
    local metric=$4
    local csv_file="${RESULTS_DIR}/${dataset}_results.csv"
    
    echo "Running: $model on $dataset with $method pruning $([ "$metric" != "none" ] && echo "($metric)")"
    
    if [ "$method" = "random" ]; then
        uv run examples/bert.py \
            --dataset "$dataset" \
            --model-name "$model" \
            --pruning-method random \
            --results-csv "$csv_file" \
            --models-dir "$MODELS_DIR" \
            --epochs "$EPOCHS"
    else
        uv run examples/bert.py \
            --dataset "$dataset" \
            --model-name "$model" \
            --pruning-method rf \
            --distance-metric "$metric" \
            --results-csv "$csv_file" \
            --models-dir "$MODELS_DIR" \
            --epochs "$EPOCHS"
    fi
}

# Main experimental loop
experiment_count=0
total_experiments=$((${#DATASETS[@]} * ${#MODELS[@]} * 4))

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=== DATASET: ${dataset^^} ==="
    
    for model in "${MODELS[@]}"; do
        echo ""
        echo "--- Model: $model ---"
        
        # Run random pruning baseline
        ((experiment_count++))
        echo "[$experiment_count/$total_experiments] Random pruning..."
        if ! run_single_experiment "$dataset" "$model" "random" "none"; then
            echo "ERROR: Failed random pruning for $model on $dataset"
            continue
        fi
        
        # Run RF pruning with different distance metrics
        for metric in "${DISTANCE_METRICS[@]}"; do
            ((experiment_count++))
            echo "[$experiment_count/$total_experiments] RF pruning with $metric distance..."
            if ! run_single_experiment "$dataset" "$model" "rf" "$metric"; then
                echo "ERROR: Failed RF pruning ($metric) for $model on $dataset"
                continue
            fi
        done
        
        echo "Completed all pruning methods for $model on $dataset"
    done
    
    echo "=== COMPLETED DATASET: ${dataset^^} ==="
    echo "Results saved to: ${RESULTS_DIR}/${dataset}_results.csv"
done

# --- Final Summary ---
echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================"
echo "Job finished at $(date)"
echo ""
echo "Results Summary:"
for dataset in "${DATASETS[@]}"; do
    csv_file="${RESULTS_DIR}/${dataset}_results.csv"
    if [ -f "$csv_file" ]; then
        echo "- ${dataset^^}: $(wc -l < "$csv_file") rows in ${csv_file}"
    else
        echo "- ${dataset^^}: No results file found!"
    fi
done

echo ""
echo "Trained models cached in: $MODELS_DIR"
echo "Results ready for paper tables!"
echo "========================================"