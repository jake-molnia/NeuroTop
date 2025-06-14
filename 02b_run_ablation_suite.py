# 02b_run_ablation_suite.py

import click
from pathlib import Path
import subprocess
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

# --- Define the suite of experiments to run ---
# Each dictionary defines one full experiment run.
EXPERIMENT_SUITE = [
    #{
    #    'model': 'MLPnet', 
    #    'dataset': 'cifar10',
    #    'train_epochs': 20
    #},
    #{
    #   'model': 'MLPnet', 
    #    'dataset': 'cifar10',
    #   'train_epochs': 0
    #},
    {
        'model': 'resnet18', 
        'dataset': 'cifar10',
        'train_epochs': 5
    },
    {
        'model': 'vgg11_bn',
        'dataset': 'cifar10',
        'train_epochs': 5
    },
    # --- You can add more experiments here! ---
    # Example:
    # {
    #     'model': 'resnet34', 
    #     'dataset': 'cifar100',
    #     'train_epochs': 50
    # },
]

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
        # Print stdout/stderr only if needed for debugging
        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        ic(f"Command failed with exit code {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise  # Stop the script if any step fails

def plot_comparison(results_files, output_path):
    """Loads all results and plots a single comparison graph."""
    ic(f"Generating final comparison plot from {len(results_files)} result files...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_files)))
    
    for i, (label, path) in enumerate(results_files.items()):
        with open(path, 'rb') as f:
            results_data = pickle.load(f)
        
        percentages = [r['percent_removed'] for r in results_data]
        accuracies = [r['accuracy'] for r in results_data]
        
        ax.plot(percentages, accuracies, marker='', linestyle='-', label=label, color=colors[i])

    ax.set_title('Model Performance Degradation vs. Neuron Ablation\n(Strategy: Homology Degree)', fontsize=16)
    ax.set_xlabel('Percentage of Neurons Removed (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.legend(title='Model & Dataset', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=-2, right=102)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    ic(f"Saved final comparison plot to: {output_path}")
    plt.close(fig)


@click.command()
@click.option('--base-output-dir', type=click.Path(path_type=Path), default='outputs', help="Base directory for all outputs.")
def main(base_output_dir):
    """
    Orchestrates a full suite of training, activation extraction, 
    and neuron ablation experiments, then plots a comparison.
    """
    results_to_plot = {}

    for exp in EXPERIMENT_SUITE:
        model_name = exp['model']
        dataset_name = exp['dataset']
        epochs = exp['train_epochs']
        
        ic.enable()
        ic.configureOutput(prefix=f'Suite | {model_name}/{dataset_name} | ')
        print("\n" + "="*80)
        ic(f"Starting experiment for {model_name} on {dataset_name}")
        print("="*80 + "\n")

        # --- Define all required paths ---
        weights_dir = base_output_dir / 'weights'
        activations_dir = base_output_dir / 'activations'
        ablation_dir = base_output_dir / 'ablation' / f'{model_name}_{dataset_name}_{ABLATION_SETTINGS["removal-strategy"]}'
        
        weights_path = weights_dir / f'{dataset_name}_{model_name}.pth'
        activations_path = activations_dir / f'{model_name}_{dataset_name}_train.npz'
        
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # === STEP 1: Train Model (if necessary) ===
        if not weights_path.exists():
            ic(f"Weights not found at {weights_path}. Starting training...")
            train_cmd = [
                'python', '00a_train.py',
                '--model', model_name,
                '--dataset', dataset_name,
                '--epochs', str(epochs),
                '--save-dir', str(weights_dir)
            ]
            run_command(train_cmd)
        else:
            ic(f"Found existing weights: {weights_path}")

        # === STEP 2: Extract Activations (if necessary) ===
        if not activations_path.exists():
            ic(f"Activations not found at {activations_path}. Extracting...")
            extract_cmd = [
                'python', '01b_extract_activations.py',
                '--model', model_name, # Note: 01b needs to support model names
                '--weights-path', str(weights_path),
                '--save-path', str(activations_path),
                '--dataset', dataset_name
            ]
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
        ]
        run_command(ablation_cmd)
        
        # Store the path of the final .pkl file for plotting
        plot_filename_base = f'perf_degrad_{model_name}_{dataset_name}_{ABLATION_SETTINGS["removal-strategy"]}'
        result_file_path = ablation_dir / f'{plot_filename_base}.pkl'
        if result_file_path.exists():
            label = f"{model_name} on {dataset_name}"
            results_to_plot[label] = result_file_path
        else:
            ic(f"Warning: Could not find result file at {result_file_path}")
        
        ic.disable() # Disable ic until the next loop iteration

    # === STEP 4: Generate Final Comparison Plot ===
    if results_to_plot:
        ic.enable()
        comparison_plot_path = base_output_dir / 'ablation' / f'comparison_plot_{ABLATION_SETTINGS["removal-strategy"]}.png'
        plot_comparison(results_to_plot, comparison_plot_path)
    else:
        ic.enable()
        ic("No results were generated to plot.")

    print("\n" + "="*80)
    ic("Ablation suite has completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Need to import numpy for the plotting function
    import numpy as np
    main()
