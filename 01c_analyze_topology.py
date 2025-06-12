# analyze_topology.py
# Usage: python analyze_topology.py --activations-path outputs/activations/cifar10_mlp_train.npz --output-dir outputs/plots

import click
from pathlib import Path
import numpy as np
from icecream import ic
import wandb

from modules import (
    compute_neuron_distances, 
    run_persistent_homology,
    plot_persistence_diagram,
    plot_neuron_embedding
)

@click.command()
@click.option('--activations-path', type=click.Path(exists=True, path_type=Path), required=True, help='Path to saved activations file (.npz).')
@click.option('--output-dir', type=click.Path(path_type=Path), default='outputs/plots', help='Directory to save analysis plots.')
@click.option('--model-name', type=str, default='MLPnet', help='Name of the model for plot titles.')
@click.option('--wandb-project', default='neural-topology', help='Wandb project name.')
@click.option('--wandb-name', default=None, help='Wandb run name.')
@click.option('--log-to-wandb', is_flag=True, help='Flag to enable logging plots to Wandb.')
def main(activations_path, output_dir, model_name, wandb_project, wandb_name, log_to_wandb):
    """Perform topological analysis on saved activations and optionally log to W&B."""
    
    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    ic(f"Analyzing activations from: {activations_path}")
    ic(f"Saving plots to: {output_dir}")

    if log_to_wandb:
        config = {'activations_path': str(activations_path), 'model_name': model_name}
        wandb.init(project=wandb_project, name=wandb_name, config=config, job_type='analysis')
        ic("Logging to Wandb is enabled.")

    # Load Activations
    activations_data = np.load(activations_path)
    activations_dict = {key: activations_data[key] for key in activations_data.files}
    ic(f"Loaded activation layers: {list(activations_dict.keys())}")

    # Perform Analysis
    distance_matrix, neurons = compute_neuron_distances(activations_dict)
    topology_result = run_persistent_homology(distance_matrix)

    # Visualize and Save Plots
    diag_path = output_dir / 'persistence_diagram.png'
    embed_path = output_dir / 'neuron_embedding.png'
    
    plot_persistence_diagram(
        topology_result,
        title=f"Persistence Diagram for {model_name}",
        save_path=diag_path
    )
    
    plot_neuron_embedding(
        neurons,
        model_name=model_name,
        save_path=embed_path
    )

    # Log plots to Wandb if enabled
    if log_to_wandb:
        ic("Logging plots to Wandb...")
        wandb.log({
            "persistence_diagram": wandb.Image(str(diag_path)),
            "neuron_embedding": wandb.Image(str(embed_path))
        })
        wandb.finish()

    ic("Analysis complete.")
    return

if __name__ == '__main__':
    main()
