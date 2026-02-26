import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import click
import csv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
import contextlib, io
from ntop.monitoring import ActivationMonitor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_CONFIG = {
    'cola': 2, 'sst2': 2, 'mrpc': 2, 'qqp': 2, 'stsb': 1, 'rte': 2
}

def prepare_dataset(dataset_name, subset_size, tokenizer, batch_size=16, max_length=128):
    dataset = load_dataset('glue', dataset_name)
    
    text_cols = (['sentence'] if dataset_name in ['cola', 'sst2'] 
                else ['sentence1', 'sentence2'])
    
    def tokenize_function(examples):
        args = [examples[col] for col in text_cols]
        return tokenizer(*args, truncation=True, padding=False, max_length=max_length)

    tokenized = dataset.map(tokenize_function, batched=True, 
                           remove_columns=[c for c in dataset['train'].column_names if c != 'label'])
    tokenized = tokenized.rename_column("label", "labels").with_format("torch")

    train_data = tokenized['train'].select(range(min(subset_size, len(tokenized['train']))))
    val_split = 'validation' if 'validation' in tokenized else 'test'
    val_data = tokenized[val_split].select(range(min(subset_size // 4, len(tokenized[val_split]))))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return (DataLoader(train_data, batch_size=batch_size, collate_fn=collator, shuffle=True),
            DataLoader(val_data, batch_size=batch_size, collate_fn=collator))

def evaluate(model, dataloader, quiet=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=quiet):
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = torch.argmax(model(**batch).logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    return 100 * correct / total

def get_or_train_model(dataset, model_name, subset_size, models_dir, epochs=50):
    model_path = os.path.join(models_dir, f"{model_name.replace('/', '_')}_{dataset}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if os.path.exists(model_path):
        print(f"Loading existing trained model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        return model, tokenizer
    
    print(f"Training new model for {model_name} on {dataset}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=DATASET_CONFIG[dataset]).to(device)
    
    train_loader, _ = prepare_dataset(dataset, subset_size, tokenizer)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")
    
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Saved trained model to {model_path}")
    
    return model, tokenizer

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero

def prune_by_total_percentage(model, analysis_state, pruning_method, target_pct, component_wise):
    """Prune a fixed percentage of neurons across all prunable components"""
    
    # Get components to prune
    if component_wise:
        components = [c for c in analysis_state['by_components'].keys() 
                     if c not in ['attention.output', 'mlp.output']]
    else:
        components = [None]  # Whole network
    
    # Count total prunable neurons
    total_neurons = 0
    for comp in components:
        if comp:
            rf_vals = analysis_state['by_components'][comp]['rf_values']
        else:
            rf_vals = analysis_state.get('full_network', {}).get('rf_values', {})
        
        for layer_rf in rf_vals.values():
            if 'rf_0' in layer_rf:
                total_neurons += len(layer_rf['rf_0'])
    
    target_pruned = int(total_neurons * target_pct / 100)
    
    print(f"Total prunable neurons: {total_neurons:,}")
    print(f"Target to prune ({target_pct}%): {target_pruned:,}")
    
    # Collect all neurons with RF scores across all components
    all_neurons = []
    for comp in components:
        if comp:
            rf_vals = analysis_state['by_components'][comp]['rf_values']
        else:
            rf_vals = analysis_state.get('full_network', {}).get('rf_values', {})
        
        for layer_name, layer_rf in rf_vals.items():
            if 'rf_0' not in layer_rf:
                continue
            
            for idx, score in enumerate(layer_rf['rf_0']):
                all_neurons.append((comp, layer_name, idx, float(score)))
    
    # Sort by RF score (lowest first for RF-based pruning)
    if pruning_method == 'rf':
        all_neurons.sort(key=lambda x: x[3])
    elif pruning_method == 'random':
        np.random.shuffle(all_neurons)
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")
    
    # Select neurons to prune
    neurons_to_prune = all_neurons[:target_pruned]
    
    # Group by component for ablation
    component_neurons = {}
    for comp, layer_name, idx, score in neurons_to_prune:
        key = comp if comp else 'full_network'
        if key not in component_neurons:
            component_neurons[key] = []
        component_neurons[key].append((layer_name, idx))
    
    # Report distribution
    print("\nPruning distribution:")
    for key, neurons in component_neurons.items():
        print(f"  {key}: {len(neurons):,} neurons")
    
    return component_neurons

@click.command()
@click.option('--dataset', required=True, type=click.Choice(list(DATASET_CONFIG.keys())))
@click.option('--model-name', required=True, type=click.Choice(['bert-base-uncased', 'bert-large-uncased']))
@click.option('--pruning-method', required=True, type=click.Choice(['random', 'rf']))
@click.option('--distance-metric', default='euclidean')
@click.option('--results-csv', required=True)
@click.option('--models-dir', default='./trained_models')
@click.option('--finetune-epochs', default=10)
@click.option('--train-epochs', default=50)
@click.option('--subset-size', default=5000)
@click.option('--component-wise/--no-component-wise', default=True)
def run_experiment(dataset, model_name, pruning_method, distance_metric, results_csv, 
                  models_dir, finetune_epochs, train_epochs, subset_size, component_wise):
    
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} on {dataset}")
    print(f"Mode: {'Component-wise' if component_wise else 'Whole network'} pruning")
    print(f"{'='*60}\n")
    
    # Load/train model
    model, tokenizer = get_or_train_model(dataset, model_name, subset_size, models_dir, train_epochs)
    train_loader, val_loader = prepare_dataset(dataset, subset_size, tokenizer)
    baseline_acc = evaluate(model, val_loader, quiet=True)
    
    # Analyze topology once
    print("Analyzing topology...")
    monitor = ActivationMonitor(model, model_type='transformer')
    analysis_state = monitor.analyze(val_loader, epoch=0, save=False,
                                   distance_metric=distance_metric, max_dim=1, 
                                   analyze_by_components=component_wise)
    monitor.remove_hooks()
    
    original_total, _ = count_parameters(model)
    
    # Prune at fixed percentages: 30%, 50%, 70%
    for prune_pct in [30, 50, 70]:
        print(f"\n{'='*60}")
        print(f"Pruning {prune_pct}% of prunable neurons")
        print(f"{'='*60}")
        
        # Get neurons to prune (distributed by RF across components)
        component_neurons = prune_by_total_percentage(
            model, analysis_state, pruning_method, prune_pct, component_wise
        )
        
        # Apply pruning
        monitor = ActivationMonitor(model, model_type='transformer')
        pruned_model = model
        
        for comp_name, neuron_list in component_neurons.items():
            pruned_model = monitor._mask_neurons(pruned_model, neuron_list)
        
        # Evaluate before fine-tuning
        before_ft = evaluate(pruned_model, val_loader, quiet=True)
        print(f"\nBefore fine-tuning: {before_ft:.2f}% (Δ {baseline_acc - before_ft:.2f}%)")
        
        # Fine-tune
        print(f"Fine-tuning for {finetune_epochs} epochs...")
        optimizer = optim.AdamW(pruned_model.parameters(), lr=2e-5)
        
        for epoch in range(finetune_epochs):
            pruned_model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{finetune_epochs}"):
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = pruned_model(**batch).loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                val_acc = evaluate(pruned_model, val_loader, quiet=True)
                print(f"  Epoch {epoch+1}: Val Acc={val_acc:.2f}%")
        
        final_acc = evaluate(pruned_model, val_loader, quiet=True)
        
        # Count parameters
        _, pruned_nonzero = count_parameters(pruned_model)
        compression_ratio = (pruned_nonzero / original_total) * 100
        params_removed = original_total - pruned_nonzero
        
        print(f"\nResults for {prune_pct}% pruning:")
        print(f"  Baseline:       {baseline_acc:.2f}%")
        print(f"  After pruning:  {before_ft:.2f}% (Δ {baseline_acc - before_ft:.2f}%)")
        print(f"  After finetune: {final_acc:.2f}% (Δ {baseline_acc - final_acc:.2f}%)")
        print(f"  Parameters: {pruned_nonzero:,}/{original_total:,} ({compression_ratio:.1f}% retained)")
        
        # Save results
        os.makedirs(os.path.dirname(results_csv), exist_ok=True)
        file_exists = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0
        
        with open(results_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Model', 'Method', 'Prune %', 'Baseline', 
                                                   'After Pruning', 'After FT', 'Delta', 
                                                   'Params Retained', 'Params Removed'])
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'Model': model_name.split('-')[1].title(),
                'Method': pruning_method,
                'Prune %': prune_pct,
                'Baseline': f"{baseline_acc:.2f}",
                'After Pruning': f"{before_ft:.2f}",
                'After FT': f"{final_acc:.2f}",
                'Delta': f"{baseline_acc - final_acc:.2f}",
                'Params Retained': f"{compression_ratio:.1f}%",
                'Params Removed': f"{params_removed:,}"
            })
        
        monitor.remove_mask_hooks()
    
    print(f"\n{'='*60}")
    print(f"All results saved to {results_csv}")
    print(f"{'='*60}")

if __name__ == '__main__':
    run_experiment()