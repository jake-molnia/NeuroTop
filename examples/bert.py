import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import click
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader

import ntop
from ntop.monitoring import ActivationMonitor
from ntop import plots

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Preparation ---
def prepare_dataset(dataset_name, subset_size, tokenizer, batch_size, max_length):
    """Prepares GLUE dataset and creates DataLoaders."""
    print(f"Setting up GLUE dataset: {dataset_name}...")
    dataset = load_dataset('glue', dataset_name)
    
    text_cols = [col for col in ['sentence', 'sentence1', 'sentence2', 'question'] if col in dataset['train'].column_names]
    
    def tokenize_function(examples):
        return tokenizer(*[examples[col] for col in text_cols], truncation=True, padding=False, max_length=max_length)

    remove_cols = list(dataset['train'].column_names)
    # Ensure 'label' is not removed if it exists
    if 'label' in remove_cols:
        remove_cols.remove('label')

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=remove_cols)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_data = tokenized_datasets['train'].select(range(min(subset_size, len(tokenized_datasets['train']))))
    val_data = tokenized_datasets['validation'].select(range(min(subset_size // 4, len(tokenized_datasets['validation']))))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collator, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collator)
    
    return train_loader, val_loader

# --- Training & Evaluation ---
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training Epoch"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch['labels']
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

# --- CLI Setup ---
@click.group()
def cli():
    """Neuro-Topological Analysis CLI for BERT models."""
    pass

@cli.command()
@click.option('--model-name', default='bert-base-uncased', help='HuggingFace model name.')
@click.option('--dataset-name', default='cola', type=click.Choice(['cola', 'sst2', 'mrpc']), help='GLUE dataset name.')
@click.option('--subset-size', default=1000, help='Number of samples to use for analysis.')
@click.option('--batch-size', default=16, help='Batch size for data loading.')
@click.option('--max-length', default=128, help='Max sequence length for tokenizer.')
@click.option('--output-dir', default='./outputs/bert_analysis', help='Directory to save analysis results.')
def analyze(model_name, dataset_name, subset_size, batch_size, max_length, output_dir):
    """Analyze a pre-trained BERT model's topology."""
    print(f"Analyzing {model_name} on {dataset_name}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    # Load data
    _, val_loader = prepare_dataset(dataset_name, subset_size, tokenizer, batch_size, max_length)
    
    # Setup and run monitor
    monitor = ActivationMonitor(model, model_type='transformer')
    analysis_params = {
        'distance_metric': 'euclidean',
        'max_dim': 1,
        'persistence_threshold': 0.01,
        'analyze_by_components': True
    }
    state = monitor.analyze(val_loader, **analysis_params)
    
    # Save results
    results_path = os.path.join(output_dir, 'analysis_state.npz')
    np.savez_compressed(results_path, state=state)
    print(f"Analysis state saved to {results_path}")

    # Generate plots for each component
    if 'by_components' in state:
        for comp_name, comp_result in state['by_components'].items():
            # **FIX:** Use a simple histogram for single-state analysis
            plt.figure(figsize=(10, 6))
            all_rf_vals = []
            if 'rf_values' in comp_result:
                for layer_rf in comp_result['rf_values'].values():
                    if 'rf_0' in layer_rf:
                        all_rf_vals.extend(layer_rf['rf_0'])
            
            if all_rf_vals:
                plt.hist(all_rf_vals, bins=50, alpha=0.7)
                plt.xlabel('RF_0 Value')
                plt.ylabel('Neuron Count')
                plt.title(f'RF_0 Distribution - {comp_name.title()}')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'rf_dist_{comp_name}.png'))
                plt.close()
    
    print("Analysis complete.")
    monitor.remove_hooks()

@cli.command()
@click.option('--model-name', default='bert-base-uncased', help='HuggingFace model name.')
@click.option('--dataset-name', default='cola', type=click.Choice(['cola', 'sst2', 'mrpc']), help='GLUE dataset name.')
@click.option('--epochs', default=5, help='Number of training epochs.') # Increased default for meaningful evolution
@click.option('--lr', default=2e-5, help='Learning rate.')
@click.option('--subset-size', default=2000, help='Number of samples for training.')
@click.option('--batch-size', default=16, help='Batch size for training.')
@click.option('--max-length', default=128, help='Max sequence length for tokenizer.')
@click.option('--output-dir', default='./outputs/bert_training_evolution', help='Directory to save model and results.')
def train(model_name, dataset_name, epochs, lr, subset_size, batch_size, max_length, output_dir):
    """Train a BERT model and analyze its topology after every epoch."""
    print(f"Training {model_name} on {dataset_name} with per-epoch analysis...")
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    # Load data
    train_loader, val_loader = prepare_dataset(dataset_name, subset_size, tokenizer, batch_size, max_length)
    
    # Initialize the monitor before the training loop
    monitor = ActivationMonitor(model, model_type='transformer')
    analysis_params = {'distance_metric': 'euclidean', 'max_dim': 1, 'analyze_by_components': True}

    # Analyze initial state (Epoch 0)
    monitor.analyze(val_loader, epoch=0, save=True, **analysis_params)
    
    # Training loop
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer)
        accuracy = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Accuracy={accuracy:.2f}%")
        
        # Analyze topology at the end of the epoch
        monitor.analyze(val_loader, epoch=epoch + 1, save=True, **analysis_params)
        
    # Save the complete history of topology states
    results_path = os.path.join(output_dir, 'topology_evolution.npz')
    monitor.save_states(results_path)
    
    # Save final model
    model_path = os.path.join(output_dir, 'final_model')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Final model saved to {model_path}")

    monitor.remove_hooks()
    
@cli.command()
@click.option('--model-path', required=True, help='Path to the trained model directory.')
@click.option('--analysis-path', required=True, help='Path to the saved *evolution* analysis state (.npz file).')
@click.option('--dataset-name', default='cola', help='GLUE dataset for evaluation.')
@click.option('--component', default=None, help='Component to ablate (e.g., "attention"). If None, ablates globally.')
@click.option('--strategy', type=click.Choice(['percent', 'random']), default='percent', help='Ablation strategy.')
@click.option('--percentage', type=float, default=50.0, help='Percentage of neurons to ablate.')
def ablate(model_path, analysis_path, dataset_name, component, strategy, percentage):
    """Ablate a trained BERT model based on a saved analysis state."""
    print(f"Running ablation on {model_path}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    
    # Load data for evaluation
    _, val_loader = prepare_dataset(dataset_name, 1000, tokenizer, 16, 128)
    
    # --- FIX STARTS HERE ---
    # Load the analysis state from the evolution file.
    # We use the state from the FINAL epoch for ablation.
    print(f"Loading analysis data from {analysis_path}...")
    data = np.load(analysis_path, allow_pickle=True)
    if 'topology_states' not in data:
        raise KeyError("'topology_states' not found in the archive. Please provide a file generated by the `train` command.")
    
    topology_evolution = data['topology_states']
    final_analysis_state = topology_evolution[-1]
    print(f"Using analysis state from final epoch ({final_analysis_state.get('epoch', 'N/A')}) for ablation.")
    # --- FIX ENDS HERE ---
    
    # Evaluate original model
    original_accuracy = evaluate(model, val_loader)
    print(f"Original Model Accuracy: {original_accuracy:.2f}%")
    
    # Perform ablation
    monitor = ActivationMonitor(model, model_type='transformer')
    ablated_model = monitor.ablate(
        model, 
        strategy=strategy, 
        value=percentage, 
        state=final_analysis_state, # Use the final state
        component_name=component
    )
    
    # Evaluate ablated model
    ablated_accuracy = evaluate(ablated_model, val_loader)
    print(f"Ablated Model Accuracy ({percentage}% {strategy} on {component or 'global'}): {ablated_accuracy:.2f}%")
    print(f"Accuracy Drop: {original_accuracy - ablated_accuracy:.2f}%")
    monitor.remove_hooks()

if __name__ == '__main__':
    cli()