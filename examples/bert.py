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

def find_optimal_compression(model, val_loader, analysis_state, pruning_method, component=None, 
                           hard_stop_threshold=5.0, acceptable_threshold=2.0):
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
    
    return baseline_acc, optimal_acc, optimal_comp

@click.command()
@click.option('--dataset', required=True, type=click.Choice(list(DATASET_CONFIG.keys())))
@click.option('--model-name', required=True, type=click.Choice(['bert-base-uncased', 'bert-large-uncased']))
@click.option('--pruning-method', required=True, type=click.Choice(['random', 'rf']))
@click.option('--distance-metric', default='euclidean', type=click.Choice(['euclidean', 'manhattan', 'cosine']))
@click.option('--results-csv', required=True)
@click.option('--models-dir', default='./trained_models', help='Directory to cache trained models')
@click.option('--epochs', default=50)
@click.option('--component', default=None, help='Component to prune (attention/feedforward), None for whole network')
@click.option('--subset-size', default=5000)
@click.option('--hard-stop-threshold', default=5.0, help='Hard stop threshold for performance drop (%)')
@click.option('--acceptable-threshold', default=2.0, help='Acceptable performance drop threshold (%)')
def run_experiment(dataset, model_name, pruning_method, distance_metric, results_csv, 
                  models_dir, epochs, component, subset_size, hard_stop_threshold, acceptable_threshold):
    """Run GLUE experiment with RF-based or random pruning."""
    print(f"Experiment: {model_name} on {dataset} with {pruning_method} pruning")
    
    # Get or train model (cached)
    model, tokenizer = get_or_train_model(dataset, model_name, subset_size, models_dir, epochs)
    _, val_loader = prepare_dataset(dataset, subset_size, tokenizer)
    
    # Always analyze topology (needed even for random pruning to identify neurons)
    if pruning_method == 'rf':
        print(f"Analyzing topology with {distance_metric} distance metric...")
    else:
        print("Analyzing topology (identifying neurons for random pruning)...")
    
    monitor = ActivationMonitor(model, model_type='transformer')
    analysis_state = monitor.analyze(val_loader, epoch=0, save=False,
                                   distance_metric=distance_metric, max_dim=1, 
                                   analyze_by_components=True)
    monitor.remove_hooks()
    
    # Find optimal compression
    baseline_acc, optimal_acc, compression_pct = find_optimal_compression(
        model, val_loader, analysis_state, pruning_method, component, 
        hard_stop_threshold, acceptable_threshold)
    
    # Save results
    results_dir = os.path.dirname(results_csv)
    if results_dir:  # Only create directory if there's a path
        os.makedirs(results_dir, exist_ok=True)
    file_exists = os.path.exists(results_csv)
    
    with open(results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Pruning Method', 'Baseline Acc', 
                                              'Optimal Acc', 'Delta Perf', 'Compression %'])
        if not file_exists:
            writer.writeheader()
        
        pruning_label = f"RF ({distance_metric.title()})" if pruning_method == 'rf' else "Random"
        if component:
            pruning_label += f" ({component})"
        
        writer.writerow({
            'Model': 'BERT-large' if 'large' in model_name else 'BERT-base',
            'Pruning Method': pruning_label,
            'Baseline Acc': f"{baseline_acc:.2f}",
            'Optimal Acc': f"{optimal_acc:.2f}",
            'Delta Perf': f"{baseline_acc - optimal_acc:.2f}",
            'Compression %': f"{compression_pct:.1f}"
        })
    
    print(f"\nResults: {baseline_acc:.2f}% → {optimal_acc:.2f}% ({compression_pct:.1f}% compressed)")
    print(f"Saved to {results_csv}")

if __name__ == '__main__':
    run_experiment()