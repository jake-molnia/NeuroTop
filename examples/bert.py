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
                           component=None, percentile=50,
                           hard_stop_threshold=5.0, acceptable_threshold=2.0):
    """Find optimal compression for a specific component using percentile thresholds."""
    
    baseline_acc = evaluate(model, val_loader, quiet=True)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    print(f"Pruning {'whole network' if component is None else f'{component} component'}")
    print(f"Hard stop at {hard_stop_threshold:.1f}% drop, acceptable threshold: {acceptable_threshold:.1f}%")
    
    compression = 0.0
    optimal_comp, optimal_acc = 0.0, baseline_acc
    prev_acc = baseline_acc
    prev_drop = 0.0
    step_size = 1.0
    
    with tqdm(desc="Finding optimal compression") as pbar:
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
            step_drop = prev_acc - acc
            
            # Hard stop at configurable threshold
            if total_drop >= hard_stop_threshold:
                print(f"\nHard stop triggered at {compression:.1f}% compression with {total_drop:.2f}% drop")
                break
            
            # Momentum stop: if degradation accelerates beyond acceptable threshold
            if step_drop - prev_drop > acceptable_threshold:
                print(f"\nMomentum stop triggered at {compression:.1f}% compression with step drop {step_drop:.2f}%")
                break
                
            # Update optimal if within acceptable range
            if total_drop <= acceptable_threshold:
                optimal_comp, optimal_acc = compression, acc
            
            prev_acc = acc
            prev_drop = step_drop
            monitor.remove_hooks()
            pbar.update(1)
    
    threshold = np.percentile(all_rf, percentile)
    print(f"Pruning neurons with RF < {threshold:.4f} (P{percentile})")
    
    # Prune and evaluate
    with contextlib.redirect_stdout(io.StringIO()):
        monitor = ActivationMonitor(model, model_type='transformer')
        ablated_model = monitor.ablate(
            model,
            strategy='percent' if pruning_method == 'rf' else 'random',
            value=threshold,  # Use threshold instead of percentage
            state=analysis_state,
            component_name=component
        )
    
    acc = evaluate(ablated_model, val_loader, quiet=True)
    compression_pct = (1 - len([n for rf in all_rf if rf >= threshold]) / len(all_rf)) * 100
    
    monitor.remove_hooks()
    return baseline_acc, acc, compression_pct

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
@click.option('--epochs', default=50)
@click.option('--subset-size', default=5000)
def find_component_compressions(dataset, model_name, pruning_method, distance_metric, 
                               results_csv, models_dir, epochs, subset_size):
    """Find optimal compression for each component separately."""
    
    print(f"\n{'='*60}")
    print(f"Finding component-wise compression for {model_name} on {dataset}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = get_or_train_model(dataset, model_name, subset_size, models_dir, epochs)
    _, val_loader = prepare_dataset(dataset, subset_size, tokenizer)
    
    # Analyze topology once to get RF values for ALL components
    print("Analyzing topology...")
    monitor = ActivationMonitor(model, model_type='transformer')
    analysis_state = monitor.analyze(val_loader, epoch=0, save=False,
                                   distance_metric=distance_metric, max_dim=1, 
                                   analyze_by_components=True)
    monitor.remove_hooks()
    
    # Components to test
    components = ['query', 'key', 'value', 'intermediate']
    
    # Find optimal compression for each component
    results = []
    for component in components:
        if component not in analysis_state['by_components']:
            print(f"Skipping {component} - not found in model")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing {component.upper()} component")
        print(f"{'='*60}")
        
        baseline, optimal, compression = find_optimal_compression(
            model, val_loader, analysis_state, pruning_method,
            component=component,
            hard_stop_threshold=5.0,
            acceptable_threshold=2.0
        )
        
        results.append({
            'Component': component,
            'Method': pruning_method,
            'Baseline': f"{baseline:.2f}",
            'Optimal': f"{optimal:.2f}",
            'Delta': f"{baseline - optimal:.2f}",
            'Compression': f"{compression:.1f}%"
        })
        
        print(f"\n{component}: {baseline:.2f}% → {optimal:.2f}% at {compression:.1f}% compression")
    
    # Save results
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Component', 'Method', 'Baseline', 
                                              'Optimal', 'Delta', 'Compression'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['Component']:12} | {r['Compression']:>6} compression | Δ {r['Delta']:>5}%")
    print(f"\nResults saved to {results_csv}")

if __name__ == '__main__':
    find_component_compressions()