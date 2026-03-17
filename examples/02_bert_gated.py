import torch
import torch.optim as optim
import os
import csv
import click
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader

from ntop.monitoring import collect_over_loader, analyze
from ntop.gating import GatedPruning

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_CONFIG = {'cola': 2, 'sst2': 2, 'mrpc': 2, 'qqp': 2, 'rte': 2}


def get_loaders(dataset_name, subset_size, tokenizer, batch_size=16):
    dataset = load_dataset('glue', dataset_name)
    text_cols = ['sentence'] if dataset_name in ['cola', 'sst2'] else ['sentence1', 'sentence2']

    def tokenize(examples):
        return tokenizer(*[examples[c] for c in text_cols], truncation=True, padding=False, max_length=128)

    tok = dataset.map(tokenize, batched=True,
                      remove_columns=[c for c in dataset['train'].column_names if c != 'label'])
    tok = tok.rename_column('label', 'labels').with_format('torch')

    collator = DataCollatorWithPadding(tokenizer)
    train = DataLoader(tok['train'].select(range(min(subset_size, len(tok['train'])))),
                       batch_size=batch_size, collate_fn=collator, shuffle=True)
    val_split = 'validation' if 'validation' in tok else 'test'
    val = DataLoader(tok[val_split].select(range(min(subset_size // 4, len(tok[val_split])))),
                     batch_size=batch_size, collate_fn=collator)
    return train, val


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            correct += (model(**batch).logits.argmax(-1) == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    return 100 * correct / total


def get_or_train(dataset, model_name, subset_size, models_dir, epochs):
    path = os.path.join(models_dir, f"{model_name.replace('/', '_')}_{dataset}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if os.path.exists(path):
        return AutoModelForSequenceClassification.from_pretrained(path).to(device), tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=DATASET_CONFIG[dataset]).to(device)
    train_loader, _ = get_loaders(dataset, subset_size, tokenizer)
    opt = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}"):
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch).loss.backward()
            opt.step()

    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return model, tokenizer


@click.command()
@click.option('--dataset', required=True, type=click.Choice(list(DATASET_CONFIG)))
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
@click.option('--gate-lr', default=1e-3)
@click.option('--max-samples', default=512)
def run(dataset, model_name, results_csv, models_dir, train_epochs, gating_epochs,
        subset_size, lambda_sparse, lambda_topo, lambda_polar,
        prune_threshold, prune_interval, max_samples, gate_lr):

    model, tokenizer = get_or_train(dataset, model_name, subset_size, models_dir, train_epochs)
    train_loader, val_loader = get_loaders(dataset, subset_size, tokenizer)
    baseline = evaluate(model, val_loader)
    print(f"Baseline: {baseline:.2f}%")

    # Collect activations and compute RF scores — no model inspection needed
    print("Collecting activations...")
    acts = collect_over_loader(model, val_loader, max_samples=max_samples)
    rf_scores = analyze(acts)

    gated = GatedPruning(model, device, rf_scores,
                         lambda_sparse=lambda_sparse,
                         lambda_topo=lambda_topo,
                         lambda_polar=lambda_polar)

    model_opt = optim.AdamW(model.parameters(), lr=2e-5)
    gate_opt = optim.Adam(gated.thresholds.parameters(), lr=gate_lr)

    for epoch in range(gating_epochs):
        model.train()
        gated.thresholds.train()

        total_loss = 0.0
        total_task = 0.0
        for batch in tqdm(train_loader, desc=f"Gate epoch {epoch+1}/{gating_epochs}"):
            model_opt.zero_grad()
            gate_opt.zero_grad()
            # Recompute fresh gates each batch — they're part of the graph via thresholds
            gated.compute_gates(rf_scores)
            gated.apply_gates()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            task_loss = outputs.loss
            loss = gated.compute_loss(task_loss, rf_scores)
            loss.backward()
            model_opt.step()
            gate_opt.step()
            total_loss += loss.item()
            total_task += task_loss.item()

        acc = evaluate(model, val_loader)
        stats = gated.sparsity()
        n = len(train_loader)
        print(f"Epoch {epoch+1}: total={total_loss/n:.4f} task={total_task/n:.4f} | acc={acc:.2f}% | sparsity={stats['sparsity']*100:.1f}%")

        if (epoch + 1) % prune_interval == 0:
            pruned = gated.hard_prune(prune_threshold)
            acc = evaluate(model, val_loader)
            print(f"Hard prune: {pruned} neurons | acc={acc:.2f}%")

            # Refresh RF scores after pruning
            acts = collect_over_loader(model, val_loader, max_samples=max_samples)
            rf_scores = analyze(acts)
            gated.compute_gates(rf_scores)
            gated.apply_gates()

    final = evaluate(model, val_loader)
    stats = gated.sparsity()
    print(f"\nBaseline: {baseline:.2f}% | Final: {final:.2f}% | Sparsity: {stats['sparsity']*100:.1f}%")

    os.makedirs(os.path.dirname(results_csv) or '.', exist_ok=True)
    exists = os.path.exists(results_csv) and os.path.getsize(results_csv) > 0
    with open(results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Dataset', 'Baseline', 'Final', 'Delta', 'Sparsity'])
        if not exists:
            writer.writeheader()
        writer.writerow({
            'Model': model_name,
            'Dataset': dataset,
            'Baseline': f"{baseline:.2f}",
            'Final': f"{final:.2f}",
            'Delta': f"{baseline - final:.2f}",
            'Sparsity': f"{stats['sparsity']*100:.1f}%",
        })

    gated.remove_hooks()


if __name__ == '__main__':
    run()