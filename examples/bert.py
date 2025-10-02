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
    """Load existing trained model or train new one."""
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
    
    # Save trained model
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Saved trained model to {model_path}")
    
    return model, tokenizer

def find_optimal_compression(model, val_loader, analysis_state, pruning_method, 
                           component=None,
                           hard_stop_threshold=5.0, acceptable_threshold=2.0):
    """Find optimal compression for a specific component."""
    
    baseline_acc = evaluate(model, val_loader, quiet=True)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    print(f"Pruning {component or 'whole network'} component")
    print(f"Hard stop at {hard_stop_threshold:.1f}% drop, acceptable threshold: {acceptable_threshold:.1f}%")
    
    # Get RF values for this component
    if component and 'by_components' in analysis_state:
        rf_vals = analysis_state['by_components'][component]['rf_values']
    else:
        rf_vals = analysis_state.get('full_network', {}).get('rf_values', {})
    
    # COMPUTE all_rf FIRST
    all_rf = []
    for layer_rf in rf_vals.values():
        if 'rf_0' in layer_rf:
            all_rf.extend(layer_rf['rf_0'])
    
    if not all_rf:
        print(f"No RF values found for {component}")
        return baseline_acc, baseline_acc, 0.0
    
    # Now use the iterative compression search
    compression = 0.0
    optimal_comp, optimal_acc = 0.0, baseline_acc
    prev_acc = baseline_acc
    step_size = 1.0
    
    with tqdm(desc=f"Testing {compression:.1f}% compression") as pbar:
        while compression < 95.0:
            compression += step_size
            pbar.set_description(f"Testing {compression:.1f}% compression")
            
            with contextlib.redirect_stdout(io.StringIO()):
                monitor = ActivationMonitor(model, model_type='transformer')
                ablated_model = monitor.ablate(
                    model, 
                    strategy='random' if pruning_method == 'random' else 'percent', 
                    value=compression,
                    state=analysis_state,
                    component_name=component
                )
            
            acc = evaluate(ablated_model, val_loader, quiet=True)
            total_drop = baseline_acc - acc
            
            if total_drop >= hard_stop_threshold:
                print(f"\nHard stop triggered at {compression:.1f}% compression with {total_drop:.2f}% drop")
                break
            
            if total_drop <= acceptable_threshold:
                optimal_comp, optimal_acc = compression, acc
            
            prev_acc = acc
            monitor.remove_hooks()
            pbar.update(1)
    
    return baseline_acc, optimal_acc, optimal_comp

def prune_network_compositionally(model, tokenizer, dataset, strategy):
    """
    Prune network with flexible per-component strategies.
    
    strategy: dict mapping component paths to pruning configs
    Example:
    {
        'attention.query': {'method': 'rf', 'percentile': 50},
        'attention.value': {'method': 'rf', 'percentile': 70},  # More aggressive
        'mlp.intermediate': {'method': 'rf', 'percentile': 30},  # Conservative
        'mlp.*': {'method': 'random', 'percentile': 50},  # Wildcard for all MLP
    }
    """
    
    # Analyze once to get RF values for ALL components
    monitor = ActivationMonitor(model, model_type='transformer')
    _, val_loader = prepare_dataset(dataset, 5000, tokenizer)
    
    analysis_state = monitor.analyze(val_loader, epoch=0, save=False,
                                   distance_metric='euclidean', 
                                   max_dim=1,
                                   analyze_by_components=True)
    
    # Apply pruning sequentially to each component
    pruned_model = model
    results = {}
    
    for component_path, config in strategy.items():
        if '*' in component_path:
            # Handle wildcards: prune all matching components
            matching = [c for c in analysis_state['by_components'].keys() 
                       if component_path.replace('*', '') in c]
        else:
            matching = [component_path]
        
        for comp in matching:
            print(f"\nPruning {comp} with {config}")
            
            baseline, optimal, compression = find_optimal_compression(
                pruned_model, val_loader, analysis_state,
                pruning_method=config['method'],
                component=comp,
                percentile=config.get('percentile', 50)
            )
            
            results[comp] = {
                'baseline': baseline,
                'optimal': optimal,
                'compression': compression
            }
            
            # Update model for next iteration
            pruned_model = monitor.ablate(
                pruned_model, 
                strategy=config['method'],
                value=compression,
                state=analysis_state,
                component_name=comp
            )
    
    monitor.remove_hooks()
    return pruned_model, results

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
@click.option('--component-wise/--no-component-wise', default=True, help='Prune each component separately vs whole network')
def run_experiment(dataset, model_name, pruning_method, distance_metric, results_csv, 
                  models_dir, finetune_epochs, train_epochs, subset_size, component_wise):
    """Run compression experiment with optional component-wise analysis."""
    
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} on {dataset}")
    print(f"Mode: {'Component-wise' if component_wise else 'Whole network'} pruning")
    print(f"{'='*60}\n")
    
    # Load/train model
    model, tokenizer = get_or_train_model(dataset, model_name, subset_size, models_dir, train_epochs)
    train_loader, val_loader = prepare_dataset(dataset, subset_size, tokenizer)
    baseline_acc = evaluate(model, val_loader, quiet=True)
    
    # Analyze topology
    print("Analyzing topology...")
    monitor = ActivationMonitor(model, model_type='transformer')
    analysis_state = monitor.analyze(val_loader, epoch=0, save=False,
                                   distance_metric=distance_metric, max_dim=1, 
                                   analyze_by_components=component_wise)
    monitor.remove_hooks()
    
    if component_wise:
        # Find optimal compression for each component independently
        components = [c for c in analysis_state['by_components'].keys() 
                     if c not in ['attention.output', 'mlp.output']]
        
        configs = {}
        for component in components:
            print(f"\nTesting {component}...")
            _, _, compression = find_optimal_compression(
                model, val_loader, analysis_state, pruning_method,
                component=component, hard_stop_threshold=5.0, acceptable_threshold=2.0
            )
            configs[component] = compression
            print(f"  → {compression:.1f}% safe")
        
        # Apply all pruning at once
        print(f"\nApplying all component pruning simultaneously...")
        pruned_model = model
        monitor = ActivationMonitor(model, model_type='transformer')
        for component, compression in configs.items():
            pruned_model = monitor.ablate(
                pruned_model, 
                strategy='random' if pruning_method == 'random' else 'percent',
                value=compression, state=analysis_state, component_name=component
            )
        monitor.remove_hooks()
        
    else:
        # Whole network pruning (original approach)
        print("Finding optimal whole-network compression...")
        _, _, compression = find_optimal_compression(
            model, val_loader, analysis_state, pruning_method,
            component=None, hard_stop_threshold=5.0, acceptable_threshold=2.0
        )
        
        monitor = ActivationMonitor(model, model_type='transformer')
        pruned_model = monitor.ablate(
            model, strategy='random' if pruning_method == 'random' else 'percent',
            value=compression, state=analysis_state, component_name=None
        )
        monitor.remove_hooks()
    
    # Evaluate before fine-tuning
    before_ft = evaluate(pruned_model, val_loader, quiet=True)
    
    # Fine-tune
    print(f"\nFine-tuning for {finetune_epochs} epochs...")
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
    
    # Save results
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    with open(results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Method', 'Baseline', 
                                               'After Pruning', 'After FT', 'Delta'])
        if not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0:
            writer.writeheader()
        
        writer.writerow({
            'Model': model_name.split('-')[1].title(),
            'Method': f"{pruning_method} ({'comp' if component_wise else 'full'})",
            'Baseline': f"{baseline_acc:.2f}",
            'After Pruning': f"{before_ft:.2f}",
            'After FT': f"{final_acc:.2f}",
            'Delta': f"{baseline_acc - final_acc:.2f}"
        })
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Baseline:       {baseline_acc:.2f}%")
    print(f"After pruning:  {before_ft:.2f}% (Δ {baseline_acc - before_ft:.2f}%)")
    print(f"After finetune: {final_acc:.2f}% (Δ {baseline_acc - final_acc:.2f}%)")
    print(f"\nResults saved to {results_csv}")

if __name__ == '__main__':
    run_experiment()