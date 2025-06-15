# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click
import yaml
from pathlib import Path

@click.command()
@click.option('--dir', 'experiments_dir', type=click.Path(exists=True), default='./experiments_grid', help='The base directory containing the grid search experiment results.')
def main(experiments_dir):
    """
    Scans a grid search experiment directory, loads all results, and plots
    a comprehensive comparison graph differentiating models by color and
    ablation strategies by line style.
    """
    base_path = Path(experiments_dir)
    results_paths = list(base_path.rglob('ablation_results.csv'))
    
    if not results_paths:
        print(f"No 'ablation_results.csv' files found in {experiments_dir}")
        return

    all_results = []
    for path in results_paths:
        try:
            df = pd.read_csv(path)
            config_path = path.parent / 'config.yaml'
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract structured information for plotting
            # The 'name' from the model spec (e.g., "MLP_BN-T_Drop-0.2")
            df['Model'] = config.get('model', {}).get('name', 'Unknown Model')
            # The strategy from the ablation spec (e.g., "homology_degree")
            df['Strategy'] = config.get('analysis', {}).get('ablation', {}).get('strategy', 'Unknown Strategy')
            
            all_results.append(df)
        except Exception as e:
            print(f"Could not process results in {path.parent}: {e}")
            
    if not all_results:
        print("No valid results could be loaded.")
        return
        
    combined_df = pd.concat(all_results, ignore_index=True)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 9))
    
    # Use seaborn to create a rich plot:
    # - style will use color to distinguish models
    # - hue will use line style (solid, dashed) to distinguish strategies
    sns.lineplot(
        data=combined_df, 
        x='percent_removed', 
        y='accuracy', 
        style='Model',
        hue='Strategy',
        marker='',
        markersize=6,
        linewidth=2
    )
    
    plt.title('Model Performance vs. Neuron Ablation Strategy', fontsize=18, pad=20)
    plt.xlabel('Percentage of Neurons Removed (%)', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.ylim(0, 101)
    plt.xlim(0, 100)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Experiment Setup', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    
    output_path = base_path / 'master_comparison_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMaster comparison plot saved to {output_path}")
    plt.show()


if __name__ == '__main__':
    main()
