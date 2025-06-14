# 02b_run_ablation_suite.py

import click
from pathlib import Path
import subprocess
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
import numpy as np
import itertools

# --- Define the suite of experiments to run ---
# Each dictionary defines one full experiment run.
EXPERIMENT_SUITE = [
    {
        'model': 'MNIST_MLP',
        'dataset': 'mnist',
        'train_epochs': 15  # MNIST trains fast
    },
]

# --- Define the hyperparameter grid for MNIST_MLP ---
HYPERPARAMETER_GRID = {
    'dropout_rates': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
    'use_batchnorm': [True, False]
}

# --- Define the fixed settings for the ablation study ---
ABLATION_SETTINGS = {
    'removal-strategy': 'homology-degree',
    'removal-steps': 100,
    'distance-metric': 'euclidean',
}

def run_command(command):
    """Helper function to run a command and print its output."""
    ic(f"Executing: {' '.join(command)}")
    try:
        # Using capture_output=True to hide intermediate script outputs for a cleaner log
        result = subprocess.run(command, check=True, text=True, capture_output=False)
        ic(f"Successfully finished command.")
    except subprocess.CalledProcessError as e:
        ic(f"Command failed with exit code {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise  # Stop the script if any step fails

def generate_model_filename(dataset, model, dropout_rate=None, use_batchnorm=None):
    """Generate filename with hyperparameters for MNIST_MLP, standard filename for others."""
    if model == 'MNIST_MLP' and dropout_rate is not None and use_batchnorm is not None:
        bn_suffix = 'bn1' if use_batchnorm else 'bn0'
        return f'{dataset}_{model}_dropout{dropout_rate}_{bn_suffix}.pth'
    else:
        return f'{dataset}_{model}.pth'

def generate_activations_filename(model, dataset, dropout_rate=None, use_batchnorm=None, split='train'):
    """Generate filename with hyperparameters for MNIST_MLP, standard filename for others."""
    if model == 'MNIST_MLP' and dropout_rate is not None and use_batchnorm is not None:
        bn_suffix = 'bn1' if use_batchnorm else 'bn0'
        return f'{model}_{dataset}_dropout{dropout_rate}_{bn_suffix}_{split}.npz'
    else:
        return f'{model}_{dataset}_{split}.npz'

def generate_result_filename(model_name, dataset_name, strategy, dropout_rate=None, use_batchnorm=None):
    """Generate result filename with hyperparameters for MNIST_MLP, standard filename for others."""
    if model_name == 'MNIST_MLP' and dropout_rate is not None and use_batchnorm is not None:
        bn_suffix = 'bn1' if use_batchnorm else 'bn0'
        return f'perf_degrad_{model_name}_{dataset_name}_dropout{dropout_rate}_{bn_suffix}_{strategy}'
    else:
        return f'perf_degrad_{model_name}_{dataset_name}_{strategy}'

def create_hyperparameter_label(model_name, dataset_name, dropout_rate=None, use_batchnorm=None):
    """Create a human-readable label for hyperparameter combinations."""
    if model_name == 'MNIST_MLP' and dropout_rate is not None and use_batchnorm is not None:
        bn_text = 'with BN' if use_batchnorm else 'no BN'
        return f"MNIST_MLP (dropout={dropout_rate}, {bn_text})"
    else:
        return f"{model_name} on {dataset_name}"

def plot_comparison(results_files, output_path, group_by=None):
    """Loads all results and plots comparison graphs - single overlay plot showing all models."""
    ic(f"Generating comparison plot from {len(results_files)} result files...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Generate colors and line styles for better differentiation
    colors = plt.cm.tab20(np.linspace(0, 1, len(results_files)))
    line_styles = ['-', '--', '-.', ':'] * (len(results_files) // 4 + 1)
    
    # Separate BatchNorm vs No BatchNorm for different styling
    bn_results = {}
    no_bn_results = {}
    
    for label, path in results_files.items():
        with open(path, 'rb') as f:
            results_data = pickle.load(f)
        
        percentages = [r['percent_removed'] for r in results_data]
        accuracies = [r['accuracy'] for r in results_data]
        
        # Determine color and style based on BatchNorm setting
        if 'with BN' in label:
            if 'with_bn' not in bn_results:
                bn_results['with_bn'] = []
            dropout_rate = label.split('dropout=')[1].split(',')[0]
            clean_label = f"Dropout {dropout_rate} (with BN)"
            color_idx = len(bn_results['with_bn'])
            bn_results['with_bn'].append((clean_label, percentages, accuracies, color_idx))
        elif 'no BN' in label:
            if 'no_bn' not in no_bn_results:
                no_bn_results['no_bn'] = []
            dropout_rate = label.split('dropout=')[1].split(',')[0]
            clean_label = f"Dropout {dropout_rate} (no BN)"
            color_idx = len(no_bn_results['no_bn'])
            no_bn_results['no_bn'].append((clean_label, percentages, accuracies, color_idx))
        else:
            # For non-MNIST_MLP models
            ax.plot(percentages, accuracies, marker='', linestyle='-', label=label, 
                   color=colors[0], linewidth=2)
    
    # Plot BatchNorm models with solid lines
    for clean_label, percentages, accuracies, color_idx in bn_results.get('with_bn', []):
        ax.plot(percentages, accuracies, marker='', linestyle='-', label=clean_label, 
               color=colors[color_idx], linewidth=2.5, alpha=0.8)
    
    # Plot No BatchNorm models with dashed lines
    for clean_label, percentages, accuracies, color_idx in no_bn_results.get('no_bn', []):
        ax.plot(percentages, accuracies, marker='', linestyle='--', label=clean_label, 
               color=colors[color_idx + 6], linewidth=2.5, alpha=0.8)

    ax.set_title('MNIST_MLP: Performance vs. Neuron Ablation\nComparing Dropout Rates & Batch Normalization\n(Strategy: Homology Degree)', 
                fontsize=16, pad=20)
    ax.set_xlabel('Percentage of Neurons Removed (%)', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    
    # Create custom legend
    import matplotlib.lines as mlines
    bn_line = mlines.Line2D([], [], color='black', linestyle='-', label='With BatchNorm')
    no_bn_line = mlines.Line2D([], [], color='black', linestyle='--', label='Without BatchNorm')
    
    # Get handles and labels for dropout rates
    handles, labels = ax.get_legend_handles_labels()
    
    # Add BatchNorm indicators to legend
    legend1 = ax.legend([bn_line, no_bn_line], ['With BatchNorm (solid)', 'Without BatchNorm (dashed)'], 
                       loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(legend1)
    
    # Add main legend with all dropout rates
    ax.legend(handles, labels, title='Dropout Configurations', fontsize=10, 
             bbox_to_anchor=(1.05, 0.75), loc='upper left')
    
    ax.set_ylim(bottom=0, top=100)
    ax.set_xlim(left=-2, right=102)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    ic(f"Saved comparison plot to: {output_path}")
    plt.close(fig)


@click.command()
@click.option('--base-output-dir', type=click.Path(path_type=Path), default='outputs', help="Base directory for all outputs.")
@click.option('--run-grid-search', is_flag=True, default=False, help="Run hyperparameter grid search for MNIST_MLP.")
def main(base_output_dir, run_grid_search):
    """
    Orchestrates a full suite of training, activation extraction, 
    and neuron ablation experiments, with optional hyperparameter grid search.
    """
    results_to_plot = {}

    for exp in EXPERIMENT_SUITE:
        model_name = exp['model']
        dataset_name = exp['dataset']
        epochs = exp['train_epochs']
        
        # Generate hyperparameter combinations for MNIST_MLP
        if model_name == 'MNIST_MLP' and run_grid_search:
            hyperparameter_combinations = list(itertools.product(
                HYPERPARAMETER_GRID['dropout_rates'],
                HYPERPARAMETER_GRID['use_batchnorm']
            ))
            ic(f"Running grid search with {len(hyperparameter_combinations)} combinations")
        else:
            # Single experiment with default hyperparameters
            hyperparameter_combinations = [(0.5, True)]
        
        ic(f"Hyperparameter combinations to run: {hyperparameter_combinations}")
        
        for i, (dropout_rate, use_batchnorm) in enumerate(hyperparameter_combinations):
            ic.enable()
            ic.configureOutput(prefix=f'Suite | {model_name}/{dataset_name} | ')
            
            ic(f"Running combination {i+1}/{len(hyperparameter_combinations)}: dropout={dropout_rate}, use_batchnorm={use_batchnorm}")
            
            if model_name == 'MNIST_MLP':
                bn_text = 'with BN' if use_batchnorm else 'no BN'
                ic(f"Running experiment: dropout={dropout_rate}, {bn_text}")
            
            print("\n" + "="*80)
            ic(f"Starting experiment for {model_name} on {dataset_name}")
            if model_name == 'MNIST_MLP':
                ic(f"Hyperparameters: dropout_rate={dropout_rate}, use_batchnorm={use_batchnorm}")
            print("="*80 + "\n")

            # --- Define all required paths ---
            weights_dir = base_output_dir / 'weights'
            activations_dir = base_output_dir / 'activations'
            ablation_dir = base_output_dir / 'ablation' / f'{model_name}_{dataset_name}_{ABLATION_SETTINGS["removal-strategy"]}'
            
            weights_filename = generate_model_filename(dataset_name, model_name, dropout_rate, use_batchnorm)
            weights_path = weights_dir / weights_filename
            
            activations_filename = generate_activations_filename(model_name, dataset_name, dropout_rate, use_batchnorm, 'train')
            activations_path = activations_dir / activations_filename
            
            ablation_dir.mkdir(parents=True, exist_ok=True)

            # === STEP 1: Train Model (if necessary) ===
            if not weights_path.exists() or epochs > 0:
                if not weights_path.exists():
                    ic(f"Weights not found at {weights_path}. Starting training...")
                else:
                    ic(f"Epochs set to {epochs}, re-training model...")
                
                train_cmd = [
                    'python', '00a_train.py',
                    '--model', model_name,
                    '--dataset', dataset_name,
                    '--epochs', str(epochs),
                    '--save-dir', str(weights_dir),
                    '--dropout-rate', str(dropout_rate)
                ]
                
                if use_batchnorm:
                    train_cmd.append('--use-batchnorm')
                else:
                    train_cmd.append('--no-batchnorm')
                
                run_command(train_cmd)
            else:
                ic(f"Found existing weights and train_epochs is 0: {weights_path}")

            # === STEP 2: Extract Activations (if necessary) ===
            if not activations_path.exists():
                ic(f"Activations not found at {activations_path}. Extracting...")
                extract_cmd = [
                    'python', '01b_extract_activations.py',
                    '--model', model_name,
                    '--weights-path', str(weights_path),
                    '--save-path', str(activations_path),
                    '--dataset', dataset_name,
                    '--dropout-rate', str(dropout_rate)
                ]
                
                if use_batchnorm:
                    extract_cmd.append('--use-batchnorm')
                else:
                    extract_cmd.append('--no-batchnorm')
                
                run_command(extract_cmd)
            else:
                ic(f"Found existing activations: {activations_path}")

            # === STEP 3: Run Neuron Removal Ablation ===
            ic("Starting neuron removal analysis...")
            ablation_cmd = [
                'python', '02a_test_neuron_removal.py',
                '--model-name', model_name,
                '--dataset-name', dataset_name,
                '--weights-path', str(weights_path),
                '--activations-path', str(activations_path),
                '--output-dir', str(ablation_dir),
                '--removal-strategy', ABLATION_SETTINGS['removal-strategy'],
                '--removal-steps', str(ABLATION_SETTINGS['removal-steps']),
                '--distance-metric', ABLATION_SETTINGS['distance-metric'],
                '--dropout-rate', str(dropout_rate)
            ]
            
            if use_batchnorm:
                ablation_cmd.append('--use-batchnorm')
            else:
                ablation_cmd.append('--no-batchnorm')
            
            run_command(ablation_cmd)
            
            # Store the path of the final .pkl file for plotting
            result_filename_base = generate_result_filename(model_name, dataset_name, ABLATION_SETTINGS['removal-strategy'], dropout_rate, use_batchnorm)
            result_file_path = ablation_dir / f'{result_filename_base}.pkl'
            if result_file_path.exists():
                label = create_hyperparameter_label(model_name, dataset_name, dropout_rate, use_batchnorm)
                results_to_plot[label] = result_file_path
            else:
                ic(f"Warning: Could not find result file at {result_file_path}")
            
            print(f"Completed experiment {i+1}/{len(hyperparameter_combinations)}")
            ic.disable() # Disable ic until the next loop iteration

    # === STEP 4: Generate Final Comparison Plot ===
    if results_to_plot:
        ic.enable()
        comparison_plot_path = base_output_dir / 'ablation' / f'comparison_plot_{ABLATION_SETTINGS["removal-strategy"]}_overlay.png'
        plot_comparison(results_to_plot, comparison_plot_path)
    else:
        ic.enable()
        ic("No results were generated to plot.")

    print("\n" + "="*80)
    ic("Ablation suite has completed!")
    ic(f"Total experiments run: {len(results_to_plot)}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()