# 01c_analyze_topology.py
# Usage: python 01c_analyze_topology.py --activations-path ... [OPTIONS]

import click
from pathlib import Path
import numpy as np
import torch
from icecream import ic
import wandb

from modules import (
    compute_neuron_distances, 
    run_persistent_homology,
    plot_distance_matrix,
    plot_persistence_diagram,
    plot_betti_curves,
    plot_neuron_embedding_2d,
    plot_neuron_embedding_3d
)

@click.command()
# --- Input/Output Options ---
@click.option('--activations-path', type=click.Path(exists=True, path_type=Path), required=True, help='Path to saved activations file (.npz).')
@click.option('--output-dir', type=click.Path(path_type=Path), default='outputs/plots', help='Directory to save analysis plots.')
@click.option('--model-name', type=str, default='MLPnet', help='Name of the model for plot titles.')

# --- Analysis Hyperparameters ---
@click.option('--distance-metric', type=click.Choice(['euclidean', 'cosine']), default='euclidean', help='Metric for neuron distance calculation.')
@click.option('--max-samples', type=int, default=5000, help='Max samples to use for distance calculation.')
@click.option('--reduce-dims', type=int, default=None, help='Reduce neuron dimensions to this number using PCA before distance computation.')
@click.option('--maxdim', type=int, default=2, help='Maximum homology dimension to compute.')
@click.option('--thresh', type=float, default=25.0, help='Filtration threshold for Ripser.')
@click.option('--perplexity', type=int, default=30, help='Perplexity for t-SNE algorithm.')

# --- Plotting Control ---
@click.option('--plot-distance-matrix', is_flag=True, help='Flag to plot the neuron distance matrix.')
@click.option('--plot-betti-curves', is_flag=True, help='Flag to plot the Betti curves.')
@click.option('--plot-embedding-3d', is_flag=True, help='Flag to plot the 3D t-SNE embedding.')

# --- Wandb Logging ---
@click.option('--wandb-project', default='neural-topology', help='Wandb project name.')
@click.option('--wandb-name', default=None, help='Wandb run name.')
@click.option('--log-to-wandb', is_flag=True, help='Flag to enable logging plots to Wandb.')

def main(**kwargs):
    """Perform topological analysis on saved activations with fine-grained control."""
    output_dir = kwargs['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    ic.configureOutput(prefix=f'{kwargs["model_name"]} Analysis | ')
    
    # --- Device Setup ---
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    #device = torch.device('mps' if use_mps else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    ic(f"Using device: {device}")
    
    ic(f"Analyzing activations from: {kwargs['activations_path']}")
    ic(f"Saving plots to: {output_dir}")
    ic(f"Analysis Parameters: { {k: v for k, v in kwargs.items() if k not in ['output_dir', 'activations_path']} }")


    if kwargs['log_to_wandb']:
        wandb.init(project=kwargs['wandb_project'], name=kwargs['wandb_name'], config=kwargs, job_type='analysis')
        ic("Logging to Wandb is enabled.")

    # Load Activations
    activations_data = np.load(kwargs['activations_path'])
    activations_dict = {key: activations_data[key] for key in activations_data.files}
    ic(f"Loaded activation layers: {list(activations_dict.keys())}")

    # Perform Core Analysis
    distance_matrix, neurons, layer_labels, layer_indices = compute_neuron_distances(
        activations_dict, 
        device=device,
        metric=kwargs['distance_metric'], 
        max_samples=kwargs['max_samples'],
        reduce_dims=kwargs['reduce_dims']
    )
    topology_result = run_persistent_homology(
        distance_matrix, 
        maxdim=kwargs['maxdim'], 
        thresh=kwargs['thresh']
    )

    # --- Generate, Save, and Log Plots ---
    plots_to_log = {}

    # 1. Persistence Diagram (default plot)
    diag_path = output_dir / f'persistence_diagram_{kwargs["distance_metric"]}.png'
    plot_persistence_diagram(
        topology_result,
        title=f"Persistence Diagram ({kwargs['model_name']}, {kwargs['distance_metric']})",
        save_path=diag_path
    )
    plots_to_log["persistence_diagram"] = wandb.Image(str(diag_path))

    # 2. 2D t-SNE Embedding (default plot)
    embed_2d_path = output_dir / f'neuron_embedding_2d_perp{kwargs["perplexity"]}.png'
    plot_neuron_embedding_2d(
        neurons,
        layer_labels=layer_labels,
        model_name=kwargs['model_name'],
        perplexity=kwargs['perplexity'],
        save_path=embed_2d_path
    )
    plots_to_log["neuron_embedding_2d"] = wandb.Image(str(embed_2d_path))
    
    # 3. Distance Matrix
    if kwargs['plot_distance_matrix']:
        dist_mat_path = output_dir / f'distance_matrix_{kwargs["distance_metric"]}.png'
        plot_distance_matrix(
            distance_matrix,
            layer_indices=layer_indices,
            layer_names=list(activations_dict.keys()),
            title=f"Neuron Distance Matrix ({kwargs['model_name']}, {kwargs['distance_metric']})",
            save_path=dist_mat_path
        )
        plots_to_log["distance_matrix"] = wandb.Image(str(dist_mat_path))
    
    # 4. Betti Curves
    if kwargs['plot_betti_curves']:
        betti_path = output_dir / f'betti_curves_{kwargs["distance_metric"]}.png'
        plot_betti_curves(
            topology_result,
            thresh=kwargs['thresh'],
            title=f"Betti Curves ({kwargs['model_name']}, {kwargs['distance_metric']})",
            save_path=betti_path
        )
        plots_to_log["betti_curves"] = wandb.Image(str(betti_path))
        
    # 5. 3D t-SNE Embedding
    if kwargs['plot_embedding_3d']:
        embed_3d_path = output_dir / f'neuron_embedding_3d_perp{kwargs["perplexity"]}.png'
        plot_neuron_embedding_3d(
            neurons,
            layer_labels=layer_labels,
            model_name=kwargs['model_name'],
            perplexity=kwargs['perplexity'],
            save_path=embed_3d_path
        )
        plots_to_log["neuron_embedding_3d"] = wandb.Image(str(embed_3d_path))

    # Log all generated plots to Wandb if enabled
    if kwargs['log_to_wandb']:
        ic(f"Logging {len(plots_to_log)} plots to Wandb...")
        wandb.log(plots_to_log)
        wandb.finish()

    ic("Analysis complete.")
    return

if __name__ == '__main__':
    main()
