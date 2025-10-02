#!/bin/bash
#SBATCH -N 1                          
#SBATCH --ntasks-per-node=1      
#SBATCH --mem=256g                    
#SBATCH --gres=gpu:1                  
#SBATCH -p short                    
#SBATCH -t 24:00:00              
#SBATCH -J "glue_component_pruning"
#SBATCH --output=glue_component_%A.out 
#SBATCH --error=glue_component_%A.err   
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=jrmolnia@wpi.edu

set -e

echo "Starting component-wise pruning experiments at $(date)"
echo "Running on node: $SLURM_NODELIST"

export TOKENIZERS_PARALLELISM=false

module load python
module load uv
module load cudnn8.9-cuda12.3/8.9.7.29
export UV_LINK_MODE=copy

echo "GPU found: $(nvidia-smi -L)"

RESULTS_DIR="./results"
MODELS_DIR="./trained_models"
mkdir -p "${RESULTS_DIR}" "${MODELS_DIR}"

DATASETS=("cola" "mrpc" "rte")
MODELS=("bert-base-uncased" "bert-large-uncased")
DISTANCE_METRICS=("euclidean" "manhattan" "cosine")
TRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

echo "=== EXPERIMENTAL MATRIX ==="
echo "Datasets: ${DATASETS[@]}"
echo "Models: ${MODELS[@]}" 
echo "Methods: Random + RF (${DISTANCE_METRICS[@]})"
echo "Mode: Component-wise only"
echo "Total: $((${#DATASETS[@]} * ${#MODELS[@]} * 4)) experiments"
echo "=========================="

run_experiment() {
    local dataset=$1
    local model=$2
    local method=$3
    local metric=$4
    local csv_file="${RESULTS_DIR}/${dataset}_results.csv"
    
    local desc="$model on $dataset with $method"
    [ "$metric" != "none" ] && desc="$desc ($metric)"
    
    echo "[$((++experiment_count))/$total_experiments] $desc"
    
    local cmd="uv run examples/bert.py \
        --dataset $dataset \
        --model-name $model \
        --pruning-method $method \
        --results-csv $csv_file \
        --models-dir $MODELS_DIR \
        --train-epochs $TRAIN_EPOCHS \
        --finetune-epochs $FINETUNE_EPOCHS \
        --component-wise"
    
    [ "$metric" != "none" ] && cmd="$cmd --distance-metric $metric"
    
    if ! $cmd; then
        echo "ERROR: Failed $desc"
        return 1
    fi
}

experiment_count=0
total_experiments=$((${#DATASETS[@]} * ${#MODELS[@]} * 4))

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=== DATASET: ${dataset^^} ==="
    
    for model in "${MODELS[@]}"; do
        echo ""
        echo "--- Model: $model ---"
        
        # Random baseline
        if ! run_experiment "$dataset" "$model" "random" "none"; then
            echo "WARNING: Continuing despite error..."
        fi
        
        # RF with all distance metrics
        for metric in "${DISTANCE_METRICS[@]}"; do
            if ! run_experiment "$dataset" "$model" "rf" "$metric"; then
                echo "WARNING: Continuing despite error..."
            fi
        done
        
        echo "Completed $model on $dataset"
    done
    
    echo "=== COMPLETED DATASET: ${dataset^^} ==="
done

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
        row_count=$(($(wc -l < "$csv_file") - 1))
        expected=$((${#MODELS[@]} * 4))
        echo "- ${dataset^^}: $row_count/$expected experiments"
    else
        echo "- ${dataset^^}: NO RESULTS FILE"
    fi
done

echo ""
echo "Trained models cached in: $MODELS_DIR"
echo "========================================"