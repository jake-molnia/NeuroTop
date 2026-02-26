import torch
import torch.nn as nn
import torch.optim as optim
import os
import click
import csv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader

from ntop.monitoring import ActivationMonitor
from ntop.gating import GatedPruning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_CONFIG = {
    'cola': 2, 'sst2': 2, 'mrpc': 2, 'qqp': 2, 'stsb': 1, 'rte': 2
}

def prepare_dataset(dataset_name, subset_size, tokenizer, batch_size=16, max_length=128):
    dataset = load_dataset('glue', dataset_name)
    text_cols = (['sentence'] if dataset_name in ['cola', 'sst2'] else ['sentence1', 'sentence2'])

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

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = torch.argmax(model(**batch).logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    return 100 * correct / total

def get_or_train_model(dataset, model_name, subset_size, models_dir, epochs=50):
    model_path = os.path.join(models_dir, f"{model_name.replace('/', '_')}_{dataset}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        return model, tokenizer

    print(f"Training {model_name} on {dataset}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=DATASET_CONFIG[dataset]).to(device)

    train_loader, _ = prepare_dataset(dataset, subset_size, tokenizer)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Saved model to {model_path}")
    return model, tokenizer

def log_rf_stats(rf_state):
    import numpy as np
    rf_values = rf_state.get('rf_values', {})
    if not rf_values:
        for comp_data in rf_state.get('by_components', {}).values():
            rf_values.update(comp_data.get('rf_values', {}))
    all_vals = []
    for layer_rf in rf_values.values():
        if isinstance(layer_rf, dict) and 'rf_0' in layer_rf:
            all_vals.extend(layer_rf['rf_0'])
    if all_vals:
        import numpy as np
        v = np.array(all_vals)
        print(f"  RF — min:{v.min():.4f} max:{v.max():.4f} "
              f"mean:{v.mean():.4f} std:{v.std():.4f} "
              f"nonzero:{(v > 0).mean()*100:.1f}%")
    else:
        print("  RF — no scores found, check layer name matching!")

def log_gate_stats(gated: GatedPruning):
    """Log per-layer τ, temp, and gate distribution."""
    if gated.thresholds is None or not gated.gates:
        return
    gate_vals = torch.cat([g.detach() for g in gated.gates.values()])
    print(f"  Gates — mean:{gate_vals.mean():.3f} std:{gate_vals.std():.3f} "
          f"<0.1:{( gate_vals < 0.1).float().mean()*100:.1f}% "
          f">0.9:{( gate_vals > 0.9).float().mean()*100:.1f}%")

    stats = gated.thresholds.layer_stats()
    # Summarize τ and temp across layers rather than printing every layer
    taus = [s['tau'] for s in stats.values()]
    temps = [s['temp'] for s in stats.values()]
    import numpy as np
    print(f"  τ — min:{min(taus):.4f} max:{max(taus):.4f} mean:{np.mean(taus):.4f} | "
          f"temp — min:{min(temps):.4f} max:{max(temps):.4f} mean:{np.mean(temps):.4f}")

@click.command()
@click.option('--dataset', required=True, type=click.Choice(list(DATASET_CONFIG.keys())))
@click.option('--model-name', required=True, type=click.Choice(['bert-base-uncased', 'bert-large-uncased']))
@click.option('--results-csv', required=True)
@click.option('--models-dir', default='./trained_models')
@click.option('--train-epochs', default=50)
@click.option('--gating-epochs', default=20)
@click.option('--subset-size', default=5000)
@click.option('--lambda-sparse', default=0.01)
@click.option('--lambda-topo', default=0.1)
@click.option('--lambda-polar', default=0.01)
@click.option('--prune-threshold', default=0.5)
@click.option('--prune-interval', default=5)
@click.option('--rf-max-samples', default=512)
@click.option('--gate-lr', default=1e-2, help='LR for τ and temp — higher than model LR is fine')
def run_gating_experiment(dataset, model_name, results_csv, models_dir, train_epochs,
                         gating_epochs, subset_size, lambda_sparse, lambda_topo, lambda_polar,
                         prune_threshold, prune_interval, rf_max_samples, gate_lr):

    print(f"\n{'='*60}")
    print(f"{model_name} | {dataset}")
    print(f"λ_sparse={lambda_sparse} λ_topo={lambda_topo} λ_polar={lambda_polar}")
    print(f"prune_threshold={prune_threshold} prune_interval={prune_interval}")
    print(f"{'='*60}\n")

    model, tokenizer = get_or_train_model(dataset, model_name, subset_size, models_dir, train_epochs)
    train_loader, val_loader = prepare_dataset(dataset, subset_size, tokenizer)
    baseline_acc = evaluate(model, val_loader)
    print(f"Baseline: {baseline_acc:.2f}%\n")

    print("Computing initial RF analysis...")
    monitor = ActivationMonitor(model, model_type='transformer')
    rf_state = monitor.analyze(val_loader, epoch=0, save=False,
                              distance_metric='euclidean', max_dim=1,
                              analyze_by_components=True,
                              max_samples=rf_max_samples)
    monitor.remove_hooks()
    log_rf_stats(rf_state)

    gated = GatedPruning(model, device, lambda_sparse=lambda_sparse,
                        lambda_topo=lambda_topo, lambda_polar=lambda_polar)

    # First compute_gates call triggers lazy init of per-layer thresholds
    gated.compute_gates(rf_state)
    gated.apply_gates()

    model_optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    # Separate optimizer for the gating parameters (τ and log_temp per layer)
    gate_optimizer = optim.Adam(gated.thresholds.parameters(), lr=gate_lr)

    for epoch in range(gating_epochs):
        model.train()
        gated.thresholds.train()
        total_loss = 0.0
        total_task = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{gating_epochs}", leave=False):
            model_optimizer.zero_grad()
            gate_optimizer.zero_grad()

            # Recompute gates from current τ/temp before every forward pass
            gated.compute_gates(rf_state)
            gated.apply_gates()

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            task_loss = outputs.loss

            loss_val = gated.compute_loss(task_loss, rf_state)
            loss_val.backward()

            model_optimizer.step()
            gate_optimizer.step()

            total_loss += loss_val.item()
            total_task += task_loss.item()

        n = len(train_loader)
        print(f"Epoch {epoch+1}: total={total_loss/n:.4f} task={total_task/n:.4f}")
        log_gate_stats(gated)

        if (epoch + 1) % prune_interval == 0:
            newly_pruned = gated.hard_prune(prune_threshold)
            stats = gated.get_sparsity_stats()
            acc = evaluate(model, val_loader)
            print(f"  [Hard prune @ {epoch+1}] pruned={newly_pruned} | "
                  f"sparsity={stats['sparsity']*100:.1f}% | acc={acc:.2f}%")

            print("  Recomputing RF...")
            monitor = ActivationMonitor(model, model_type='transformer')
            rf_state = monitor.analyze(val_loader, epoch=epoch+1, save=False,
                                      distance_metric='euclidean', max_dim=1,
                                      analyze_by_components=True,
                                      max_samples=rf_max_samples)
            monitor.remove_hooks()
            log_rf_stats(rf_state)

            # Recompute gates with fresh RF — τ/temp are retained and continue learning
            gated.compute_gates(rf_state)
            gated.apply_gates()

    final_acc = evaluate(model, val_loader)
    final_stats = gated.get_sparsity_stats()

    print(f"\n{'='*60}")
    print(f"Baseline:  {baseline_acc:.2f}%")
    print(f"Final:     {final_acc:.2f}%")
    print(f"Delta:     {baseline_acc - final_acc:+.2f}%")
    print(f"Sparsity:  {final_stats['sparsity']*100:.1f}%")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(results_csv) or '.', exist_ok=True)
    file_exists = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0
    with open(results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Dataset', 'Method',
                                               'Baseline', 'Final', 'Delta', 'Sparsity'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'Model': model_name.split('-')[1].title(),
            'Dataset': dataset.upper(),
            'Method': 'LayerwiseGating',
            'Baseline': f"{baseline_acc:.2f}",
            'Final': f"{final_acc:.2f}",
            'Delta': f"{baseline_acc - final_acc:.2f}",
            'Sparsity': f"{final_stats['sparsity']*100:.1f}%",
        })

    gated.remove_hooks()
    print(f"Results saved to {results_csv}")

if __name__ == '__main__':
    run_gating_experiment()