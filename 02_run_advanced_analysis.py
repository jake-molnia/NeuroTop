# 02_run_advanced_analysis.py
import os
import yaml
import torch
import argparse
import pandas as pd
from icecream import ic

# --- Local Imports ---
from modules.data import get_dataloaders
from modules.models import get_model
from modules.utils import setup_experiment, get_global_neuron_maps
import modules.analysis as analysis

def main(config_path):
    # --- 1. Setup ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup directories and device
    output_dir = setup_experiment(config['run_name'], "advanced_experiments")
    ic.configureOutput(prefix=f'{config["run_name"]} | ', includeContext=True)
    ic(f"Starting advanced analysis run: {config['run_name']}")
    ic(f"Results will be saved to: {output_dir}")

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    ic(f"Using device: {device}")

    # --- 2. Load Data and Pre-trained Model ---
    train_loader, test_loader = get_dataloaders(config['dataset'], config['batch_size'], config['data_path'])
    
    # Get model architecture to load the state dict correctly
    # We pass num_classes and any other required args for model instantiation
    model = get_model(config['dataset'], num_classes=10) # Assuming MNIST has 10 classes
    
    # Load the pre-trained weights
    checkpoint_path = config['model_checkpoint']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please run 01_run_experiment.py first.")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    ic(f"Successfully loaded pre-trained model from {checkpoint_path}")

    # --- 3. Contextual Activation Extraction ---
    ic("--- Step 1: Stratifying Activations by Context ---")
    layer_to_analyze = config['layer_to_analyze']
    stratified_activations = analysis.stratify_activations_by_context(
        model, 
        test_loader, 
        layers_to_hook=[layer_to_analyze], 
        max_samples=config['max_samples'], 
        device=device,
        normalization_method=config['normalization_method']
    )
    
    # For this pipeline, we will focus on the topology during CORRECT predictions.
    # This is a key analytical choice. One could easily run a parallel analysis on 'incorrect'.
    correct_activations = stratified_activations['correct'][layer_to_analyze]
    if correct_activations.shape[0] == 0:
        raise ValueError("No correct predictions found. Cannot proceed with analysis.")
    ic(f"Proceeding with analysis on {correct_activations.shape[0]} samples from correct predictions.")

    # --- 4. Multi-Scale Topological Analysis ---
    ic("--- Step 2: Multi-Scale Topological Analysis ---")
    dist_matrix = analysis.get_distance_matrix(correct_activations)
    
    filtration_scales = analysis.get_filtration_scales(dist_matrix, num_scales=config['num_filtration_scales'])
    
    neuron_criticality_df = analysis.classify_neuron_criticality(dist_matrix, filtration_scales)
    
    # Save criticality results
    criticality_path = os.path.join(output_dir, "neuron_criticality.csv")
    neuron_criticality_df.to_csv(criticality_path, index=False)
    ic(f"Neuron criticality classification saved to {criticality_path}")

    # Generate and save plots for multi-scale analysis
    analysis.plot_criticality_distribution(
        neuron_criticality_df, 
        os.path.join(output_dir, "criticality_distribution.png")
    )
    analysis.plot_degree_evolution(
        neuron_criticality_df, 
        filtration_scales,
        os.path.join(output_dir, "degree_evolution.png")
    )
    
    # Create topological ranking (most to least critical)
    # Order: Core -> Conditional -> Redundant
    topological_ranking = (
        neuron_criticality_df[neuron_criticality_df['classification'] == 'Core']['neuron_id'].tolist() +
        neuron_criticality_df[neuron_criticality_df['classification'] == 'Conditional']['neuron_id'].tolist() +
        neuron_criticality_df[neuron_criticality_df['classification'] == 'Redundant']['neuron_id'].tolist()
    )
    ic("Generated topological ranking based on criticality classification.")

    # --- 5. Hybrid Analysis (if enabled) ---
    if config['compute_gradient_importance']:
        ic("--- Step 3: Hybrid Analysis with Gradients ---")
        grad_importance_scores = analysis.get_gradient_importance(model, test_loader, device, layer_to_analyze)
        
        # Create gradient ranking
        gradient_ranking = np.argsort(grad_importance_scores)[::-1].tolist()
        ic("Generated gradient-based ranking.")
        
        # Plot comparison of rankings
        analysis.plot_ranking_comparison(
            topological_ranking, 
            gradient_ranking, 
            "Topological Criticality", 
            "Gradient Magnitude", 
            os.path.join(output_dir, "ranking_comparison.png")
        )
        
        # Fuse rankings for a final, robust importance score
        rankings_to_fuse = {
            'topological': topological_ranking,
            'gradient': gradient_ranking
        }
        fused_ranking = analysis.fuse_importance_rankings(rankings_to_fuse)
        
        # Save rankings to a file
        rankings_df = pd.DataFrame({
            'fused_rank': range(len(fused_ranking)),
            'fused_neuron_id': fused_ranking,
            'topological_rank_id': topological_ranking,
            'gradient_rank_id': gradient_ranking
        })
        rankings_path = os.path.join(output_dir, "neuron_rankings.csv")
        rankings_df.to_csv(rankings_path, index=False)
        ic(f"Fused and individual rankings saved to {rankings_path}")

    print("\n" + "="*50)
    print("Advanced analysis complete.")
    print(f"All results saved in: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the advanced topological analysis pipeline.")
    parser.add_argument('--config', type=str, required=True, help="Path to the advanced configuration YAML file.")
    args = parser.parse_args()
    main(args.config)
