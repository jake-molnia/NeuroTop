# run_experiment.py
import yaml
import torch
import click
import pandas as pd
from pathlib import Path
from copy import deepcopy
import itertools

from modules.utils import set_seed
from modules.data import get_dataloaders
from modules.models import get_model
from modules.training import train_and_evaluate
from modules.analysis import (
    extract_activations,
    get_ablation_strategy,
    run_ablation_test,
    get_distance_matrix,
    calculate_betti_curves,
    plot_persistence_diagram,
    plot_tsne,
    plot_distance_matrix
)

def run_ablation_on_model(config, model, activations):
    """
    Takes a trained model and its activations and applies a single
    ablation strategy to it, now analyzing all specified layers globally.
    """
    ablation_config = config['analysis']['ablation']
    ablation_strategy = ablation_config['strategy']
    
    model_run_name = config['model']['name']
    output_dir = Path(config['outputs']['base_dir']) / model_run_name / ablation_strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"--- Applying ablation '{ablation_strategy}' to model '{model_run_name}' ---")
    print(f"Output directory: {output_dir}")

    # --- Global Analysis Setup ---
    print("--- Concatenating activations for global analysis ---")
    layers_to_analyze = config['analysis']['activation_extraction']['layers']
    
    concatenated_activations_list = []
    global_to_local_map = [] # To map a global neuron index back to its layer and local index

    for layer_name in layers_to_analyze:
        if layer_name not in activations:
            print(f"Warning: Layer '{layer_name}' not found in extracted activations. Skipping.")
            continue
        layer_acts = activations[layer_name]
        concatenated_activations_list.append(layer_acts)
        # Create mapping for each neuron in this layer
        for i in range(layer_acts.shape[1]):
            global_to_local_map.append((layer_name, i))

    if not concatenated_activations_list:
        print("Error: No valid layers found for concatenation. Aborting ablation.")
        return

    concatenated_activations = torch.cat(concatenated_activations_list, dim=1)
    print(f"Total neurons for global analysis: {concatenated_activations.shape[1]}")

    dist_matrix = get_distance_matrix(concatenated_activations)

    # --- Visualizations on Global Data ---
    viz_config = config['analysis'].get('visualizations', {})
    if viz_config.get('persistence_diagram'):
        diagrams = calculate_betti_curves(dist_matrix, maxdim=1)
        plot_persistence_diagram(diagrams, output_dir / 'persistence_diagram_global.png')
    if viz_config.get('tsne_plot'):
        plot_tsne(concatenated_activations, output_dir / 'tsne_plot_global.png')
    if viz_config.get('distance_matrix'):
        plot_distance_matrix(dist_matrix, output_dir / 'distance_matrix_global.png')

    # --- Global Ablation Test ---
    ablation_strategy_func = get_ablation_strategy(ablation_strategy)
    strategy_params = ablation_config.get('params', {}).get(ablation_strategy, {})
    global_neuron_order = ablation_strategy_func(concatenated_activations, dist_matrix, **strategy_params)
    
    ablation_step = ablation_config.get('step', 5)
    percentages = list(range(0, 101, ablation_step))
    
    device = next(model.parameters()).device
    _, test_loader = get_dataloaders(config)
    ablation_results_df = run_ablation_test(model, test_loader, global_neuron_order, global_to_local_map, percentages, device)
    
    results_path = output_dir / 'ablation_results.csv'
    ablation_results_df.to_csv(results_path, index=False)
    print(f"Ablation results saved to {results_path}")


@click.command()
@click.option('--config', 'config_path', type=click.Path(exists=True), default='configs/grid_search.yaml', help='Path to the grid search configuration file.')
def main(config_path):
    """
    Parses a grid search config, trains each model ONCE, and then applies
    all specified ablation strategies to that single trained model.
    """
    with open(config_path, 'r') as f:
        grid_config = yaml.safe_load(f)
    
    base_config = grid_config.get('base_config', {})
    model_configs = grid_config.get('models', [])
    ablation_configs = grid_config.get('ablations', [])

    if not model_configs or not ablation_configs:
        print("Configuration file must contain lists for 'models' and 'ablations'.")
        return

    print(f"Found {len(model_configs)} model(s) and {len(ablation_configs)} ablation strategie(s).")
    print("Beginning experiment suite...")

    # --- Outer Loop: Iterate over Models ---
    for model_spec in model_configs:
        model_config = deepcopy(base_config)
        model_config['model'] = model_spec
        model_name = model_spec.get('name', 'model')
        
        print(f"\n{'='*60}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'='*60}")
        
        set_seed(model_config['seed'])
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        model_output_dir = Path(model_config['outputs']['base_dir']) / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        model = get_model(model_config['model'])
        model.to(device)

        model_checkpoint_path = model_output_dir / 'model.pth'
        
        if model_config['training']['enabled'] and not model_checkpoint_path.exists():
            print(f"--- Training model: {model_name} ---")
            train_loader, test_loader = get_dataloaders(model_config)
            model = train_and_evaluate(model_config, model, train_loader, test_loader, device)
            if model_config['outputs']['save_model_checkpoint']:
                torch.save(model.state_dict(), model_checkpoint_path)
        else:
            print(f"--- Loading or using existing model: {model_name} ---")
            if model_checkpoint_path.exists():
                model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
                print(f"Loaded checkpoint from {model_checkpoint_path}")
            else:
                print("Warning: No checkpoint found and training is disabled. Using random weights.")
        
        activations_path = model_output_dir / 'activations.pt'
        if not activations_path.exists():
            print(f"--- Extracting activations for {model_name} ---")
            _, test_loader = get_dataloaders(model_config)
            required_layers = model_config['analysis']['activation_extraction']['layers']
            activations = extract_activations(model, test_loader, required_layers, model_config['analysis']['activation_extraction']['max_samples'], device)
            torch.save(activations, activations_path)
        else:
            print(f"--- Loading existing activations for {model_name} ---")
            activations = torch.load(activations_path)

        # --- Inner Loop: Iterate over Ablation Strategies ---
        for ablation_spec in ablation_configs:
            run_config = deepcopy(model_config)
            run_config['analysis']['ablation'] = ablation_spec
            try:
                run_ablation_on_model(run_config, model, activations)
            except Exception as e:
                import traceback
                print(f"\n!!!!!! Ablation strategy '{ablation_spec.get('strategy')}' failed on model '{model_name}' !!!!!!")
                traceback.print_exc()
                print(f"ERROR: {e}")
                print("Continuing to the next ablation...")
                continue

if __name__ == '__main__':
    main()
