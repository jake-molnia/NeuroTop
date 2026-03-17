import torch
import torch.nn as nn
import torch.optim as optim
import click
import csv
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader

from ntop.monitoring import ActivationMonitor
from ntop.rl_strategy import RLPruningStrategy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_CONFIG = {'cola': 2, 'sst2': 2, 'mrpc': 2}


def count_parameters(model):
    """Count total and non-zero parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


def prepare_dataset(dataset_name, tokenizer, batch_size=16):
    """EXACT copy from your working bert.py example."""
    dataset = load_dataset('glue', dataset_name)
    
    text_cols = (['sentence'] if dataset_name in ['cola', 'sst2'] 
                else ['sentence1', 'sentence2'])
    
    def tokenize_function(examples):
        args = [examples[col] for col in text_cols]
        return tokenizer(*args, truncation=True, padding=False, max_length=128)

    # KEY: Keep 'label' column during tokenization
    tokenized = dataset.map(tokenize_function, batched=True, 
                           remove_columns=[c for c in dataset['train'].column_names if c != 'label'])
    
    # THEN rename it
    tokenized = tokenized.rename_column("label", "labels").with_format("torch")

    # Select subsets
    train_data = tokenized['train'].select(range(min(1000, len(tokenized['train']))))
    val_data = tokenized['validation'].select(range(min(200, len(tokenized['validation']))))

    # Use DataCollator
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return (DataLoader(train_data, batch_size=batch_size, collate_fn=collator, shuffle=True),
            DataLoader(val_data, batch_size=batch_size, collate_fn=collator))


@torch.no_grad()
def evaluate(model, dataloader):
    """Evaluate model accuracy."""
    model.eval()
    correct = total = 0
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        predictions = torch.argmax(model(**batch).logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)
    
    return correct / total


def fine_tune(model, train_loader, epochs=5, lr=2e-5):
    """Fine-tune model after pruning."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"  Fine-tune epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


@click.command()
@click.option('--dataset', default='cola', type=click.Choice(['cola', 'sst2', 'mrpc']))
@click.option('--model-name', default='bert-base-uncased')
@click.option('--target-sparsity', default=0.5, help='Target pruning sparsity (0-1)')
@click.option('--prune-steps', default=10, help='Number of pruning iterations')
@click.option('--sample-num', default=5, help='Samples per RL step')
@click.option('--distance-metric', default='euclidean')
@click.option('--pretrain-epochs', default=3, help='Epochs to pre-train model')
@click.option('--component-wise', is_flag=True, help='Track per-component statistics')
@click.option('--results-csv', default='./results/rl_pruning_results.csv')
def main(dataset, model_name, target_sparsity, prune_steps, sample_num, distance_metric, 
         pretrain_epochs, component_wise, results_csv):
    """Run RL-guided pruning experiment."""
    
    print(f"\n{'='*60}")
    print(f"RL-Pruning: {model_name} on {dataset}")
    print(f"Target sparsity: {target_sparsity}, Steps: {prune_steps}")
    print(f"{'='*60}\n")
    
    # Load model and data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=DATASET_CONFIG[dataset]
    ).to(device)
    
    train_loader, val_loader = prepare_dataset(dataset, tokenizer)
    
    # =================================================================
    # STEP 0: PRE-TRAIN MODEL ON TASK
    # =================================================================    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(pretrain_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}/{pretrain_epochs}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluate progress
        val_acc = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{pretrain_epochs}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
        
    # =================================================================
    # STEP 1: BASELINE EVALUATION & PARAMETER COUNT
    # =================================================================
    
    baseline_acc = evaluate(model, val_loader)
    original_total_params, original_nonzero_params = count_parameters(model)
    
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Original parameters: {original_total_params:,}")
    print(f"Non-zero parameters: {original_nonzero_params:,}\n")
    
    # =================================================================
    # STEP 2: TOPOLOGY ANALYSIS
    # =================================================================
    monitor = ActivationMonitor(model, model_type='transformer')
    analysis_state = monitor.analyze(
        val_loader, epoch=0, save=False,
        distance_metric=distance_metric, max_dim=1,
        analyze_by_components=True, max_samples=100
    )
    monitor.remove_hooks()
    
    # Extract layer information and track component-wise if requested
    rf_values = {}
    total_neurons = 0
    component_stats = {}
    
    for comp_name, comp_data in analysis_state['by_components'].items():
        comp_neurons = 0
        for layer_name, layer_rf in comp_data['rf_values'].items():
            rf_values[layer_name] = layer_rf
            n_neurons = len(layer_rf['rf_0'])
            total_neurons += n_neurons
            comp_neurons += n_neurons
        
        if component_wise:
            component_stats[comp_name] = {
                'neurons': comp_neurons,
                'baseline_acc': baseline_acc
            }
    
    layer_names = list(rf_values.keys())
    print(f"✓ Found {len(layer_names)} layers, {total_neurons} total neurons")
    if component_wise:
        print(f"✓ Tracking {len(component_stats)} components:\n")
        for comp_name, stats in component_stats.items():
            print(f"  - {comp_name}: {stats['neurons']} neurons")
    print()
    
    # =================================================================
    # STEP 3: INITIALIZE RL STRATEGY
    # =================================================================
    rl_strategy = RLPruningStrategy(
        rf_values=rf_values,
        total_neurons=total_neurons,
        layer_names=layer_names,
        target_sparsity=target_sparsity,
        prune_steps=prune_steps,
        sample_num=sample_num,
        greedy_epsilon=0.3,
        explore_strategy='cosine'
    )
    
    print(f"✓ RL Strategy initialized")
    print(f"  - Method: RL + RF (rf_0)")
    print(f"  - Distance metric: {distance_metric}")
    print(f"  - Pruning {rl_strategy.neurons_per_step} neurons per step")
    print(f"  - Initial epsilon: {rl_strategy.greedy_epsilon:.3f}")
    print(f"  - Exploration strategy: {rl_strategy.explore_strategy}\n")
    
    # =================================================================
    # STEP 4: RL PRUNING LOOP
    # =================================================================    
    best_acc = baseline_acc
    best_model = model
    
    with tqdm(total=prune_steps, desc='RL Pruning', position=0) as pbar:
        for step in range(1, prune_steps + 1):
            print(f"\n{'─'*60}")
            print(f"Step {step}/{prune_steps}")
            print(f"{'─'*60}")
            
            # Sample multiple architectures
            rl_strategy.clear_replay_buffer()
            
            print(f"Sampling {sample_num} architectures...")
            for sample_idx in range(sample_num):
                # Create pruned model
                test_model = model
                
                # Sample pruning decision
                monitor_sample = ActivationMonitor(test_model, model_type='transformer')
                pruned_model = monitor_sample.ablate(
                    test_model,
                    strategy='rl',
                    value=0,
                    state=analysis_state,
                    rl_strategy=rl_strategy
                )
                monitor_sample.remove_hooks()
                
                # Evaluate
                acc = evaluate(pruned_model, val_loader)
                
                # Compute Q-value
                neurons_pruned = int(total_neurons * (step / prune_steps) * target_sparsity)
                q_value = rl_strategy.compute_Q_value(acc, neurons_pruned)
                
                # Store in replay buffer
                _, distribution = rl_strategy.sample_pruning_decision()
                rl_strategy.update_replay_buffer(q_value, distribution, sample_idx)
                
                print(f"  Sample {sample_idx+1}/{sample_num}: Acc={acc:.4f}, Q={q_value:.4f}")
            
            # Update distribution based on best samples
            pd_change = rl_strategy.update_distribution()
            best_q = rl_strategy.replay_buffer[:, 0].max().item()
            print(f"\n  ✓ Distribution updated:")
            print(f"    - Best Q-value: {best_q:.4f}")
            print(f"    - Max change: {pd_change.abs().max():.4f}")
            
            # Apply best pruning decision
            monitor_best = ActivationMonitor(model, model_type='transformer')
            model = monitor_best.ablate(
                model, strategy='rl', value=0,
                state=analysis_state, rl_strategy=rl_strategy
            )
            monitor_best.remove_hooks()
            
            # Evaluate before fine-tuning
            acc_before_ft = evaluate(model, val_loader)
            print(f"\n  Accuracy before fine-tune: {acc_before_ft:.4f}")
            
            # Fine-tune periodically
            if step % 2 == 0:
                print(f"  Fine-tuning for 3 epochs...")
                fine_tune(model, train_loader, epochs=3)
                
                # Re-analyze topology after fine-tuning
                print(f"  Re-analyzing topology...")
                monitor_reanalyze = ActivationMonitor(model, model_type='transformer')
                analysis_state = monitor_reanalyze.analyze(
                    val_loader, epoch=step, save=False,
                    distance_metric=distance_metric, max_dim=1,
                    analyze_by_components=True, max_samples=100
                )
                monitor_reanalyze.remove_hooks()
                
                # Update RF values in RL strategy
                new_rf_values = {}
                for comp_name, comp_data in analysis_state['by_components'].items():
                    for layer_name, layer_rf in comp_data['rf_values'].items():
                        new_rf_values[layer_name] = layer_rf
                rl_strategy.rf_values = new_rf_values
            
            # Final evaluation for this step
            acc_after_ft = evaluate(model, val_loader)
            print(f"  Accuracy after fine-tune: {acc_after_ft:.4f}")
            _, step_nonzero = count_parameters(model)
            print(f"  Params remaining: {step_nonzero:,} ({step_nonzero/original_nonzero_params*100:.1f}%)")
            
            best_acc = acc_after_ft
            best_model = model
            print(f"  Current accuracy: {acc_after_ft:.4f}")
            
            # Step RL strategy (updates epsilon)
            rl_strategy.step()
            
            # Update progress bar
            pbar.set_postfix({
                'Best': f'{best_acc:.4f}',
                'Cur': f'{acc_after_ft:.4f}',
                'ε': f'{rl_strategy.greedy_epsilon:.3f}'
            })
            pbar.update(1)
    
    # =================================================================
    # FINAL RESULTS & PARAMETER COUNT
    # =================================================================
    final_acc = evaluate(best_model, val_loader)
    final_total_params, final_nonzero_params = count_parameters(best_model)
    
    # Calculate compression
    params_removed = original_nonzero_params - final_nonzero_params
    compression_ratio = (1 - final_nonzero_params / original_nonzero_params) * 100
    params_retained = (final_nonzero_params / original_nonzero_params) * 100
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\nAccuracy:")
    print(f"  Baseline:     {baseline_acc:.4f}")
    print(f"  Final:        {final_acc:.4f}")
    print(f"  Delta:        {baseline_acc - final_acc:.4f} ({(baseline_acc - final_acc)/baseline_acc*100:.2f}%)")
    print(f"\nParameters:")
    print(f"  Original:     {original_nonzero_params:,}")
    print(f"  Final:        {final_nonzero_params:,}")
    print(f"  Removed:      {params_removed:,}")
    print(f"  Retained:     {params_retained:.1f}%")
    print(f"  Compression:  {compression_ratio:.1f}%")
    print(f"\nTarget vs Actual:")
    print(f"  Target sparsity:  {target_sparsity:.2%}")
    print(f"  Actual sparsity:  {compression_ratio/100:.2%}")
    print(f"{'='*60}\n")
    
    # =================================================================
    # SAVE RESULTS TO CSV
    # =================================================================
    os.makedirs(os.path.dirname(results_csv) if os.path.dirname(results_csv) else '.', exist_ok=True)
    
    file_exists = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0
    
    # Main result
    result_data = {
        'Component': 'Full Network',
        'Model': model_name,
        'Dataset': dataset,
        'Method': f'RL+RF ({distance_metric})',
        'Baseline': f"{baseline_acc:.4f}",
        'Final': f"{final_acc:.4f}",
        'Delta': f"{baseline_acc - final_acc:.4f}",
        'Params_Original': f"{original_nonzero_params:,}",
        'Params_Final': f"{final_nonzero_params:,}",
        'Params_Removed': f"{params_removed:,}",
        'Compression': f"{compression_ratio:.1f}%",
        'Target_Sparsity': f"{target_sparsity:.2%}"
    }
    
    with open(results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Component', 'Model', 'Dataset', 'Method', 
            'Baseline', 'Final', 'Delta',
            'Params_Original', 'Params_Final', 'Params_Removed', 
            'Compression', 'Target_Sparsity'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_data)
    
    print(f"✓ Results saved to {results_csv}")
    
    # Save component-wise results if tracked
    if component_wise and component_stats:
        component_csv = results_csv.replace('.csv', '_components.csv')
        comp_file_exists = os.path.exists(component_csv) and os.path.getsize(component_csv) > 0
        
        with open(component_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Component', 'Dataset', 'Method', 'Neurons', 
                'Baseline', 'Final', 'Delta'
            ])
            if not comp_file_exists:
                writer.writeheader()
            
            for comp_name, stats in component_stats.items():
                writer.writerow({
                    'Component': comp_name,
                    'Dataset': dataset,
                    'Method': f'RL+RF ({distance_metric})',
                    'Neurons': stats['neurons'],
                    'Baseline': f"{stats['baseline_acc']:.4f}",
                    'Final': f"{final_acc:.4f}",  # Approximate - full model accuracy
                    'Delta': f"{stats['baseline_acc'] - final_acc:.4f}"
                })
        
        print(f"✓ Component-wise results saved to {component_csv}")


if __name__ == '__main__':
    main()