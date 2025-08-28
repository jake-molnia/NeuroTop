#%% Setup
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
import os
from tqdm import tqdm
import time

import ntop
from ntop.monitoring import ActivationMonitor
from ntop import analysis, plots

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"BERT Training Analysis with Topology Monitoring (PyTorch + HuggingFace)")
print(f"Device: {device}")

#%% Data Preparation Functions
def prepare_dataset(task='cola', subset_size=2000, max_length=128):
    """Prepare GLUE dataset with HuggingFace"""
    print(f"Setting up BERT experiment for {task}...")
    
    # Load dataset with debug info
    try:
        print(f"Attempting to download/load GLUE dataset for task: {task}")
        if task == 'cola':
            dataset = load_dataset('glue', 'cola')
            text_column = 'sentence'
            label_column = 'label'
            num_labels = 2
        elif task == 'sst2':
            dataset = load_dataset('glue', 'sst2')
            text_column = 'sentence'
            label_column = 'label'
            num_labels = 2
        elif task == 'mrpc':
            dataset = load_dataset('glue', 'mrpc')
            text_column = ['sentence1', 'sentence2']
            label_column = 'label'
            num_labels = 2
        elif task == 'qnli':
            dataset = load_dataset('glue', 'qnli')
            text_column = ['question', 'sentence']
            label_column = 'label'
            num_labels = 2
        else:
            raise ValueError(f"Unsupported task: {task}")
        print(f"Dataset loaded: {dataset}")
    except Exception as e:
        print(f"Failed to load GLUE dataset for task {task}: {e}")
        raise
    
    # Apply subset for faster experimentation
    if subset_size:
        train_dataset = dataset['train'].select(range(min(subset_size, len(dataset['train']))))
        dev_dataset = dataset['validation'].select(range(min(subset_size//4, len(dataset['validation']))))
    else:
        train_dataset = dataset['train']
        dev_dataset = dataset['validation']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(dev_dataset)}")
    
    return train_dataset, dev_dataset, text_column, label_column, num_labels

def tokenize_function(examples, tokenizer, text_column, max_length=128):
    """Tokenize texts using HuggingFace tokenizer"""
    if isinstance(text_column, list):
        # For sentence pair tasks
        return tokenizer(
            examples[text_column[0]], 
            examples[text_column[1]],
            truncation=True,
            padding=False,  # Dynamic padding
            max_length=max_length
        )
    else:
        # For single sentence tasks
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,  # Dynamic padding
            max_length=max_length
        )

def create_data_loaders(train_dataset, dev_dataset, tokenizer, text_column, batch_size=16, max_length=128):
    """Create PyTorch DataLoaders with proper tokenization"""
    
    # Tokenize datasets
    # Determine which columns to remove (the original text columns)
    if isinstance(text_column, list):
        remove_columns = text_column.copy()
    else:
        remove_columns = [text_column]

    # Also remove 'idx' if present (common in GLUE)
    for col in ['idx']:
        if col in train_dataset.column_names:
            remove_columns.append(col)


    # Tokenize and remove only text columns
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, text_column, max_length),
        batched=True,
        remove_columns=remove_columns
    )
    tokenized_dev = dev_dataset.map(
        lambda x: tokenize_function(x, tokenizer, text_column, max_length),
        batched=True,
        remove_columns=remove_columns
    )

    # Ensure label column is named 'labels' for HuggingFace models
    def rename_label_column(dataset):
        if 'label' in dataset.column_names and 'labels' not in dataset.column_names:
            return dataset.rename_column('label', 'labels')
        return dataset

    tokenized_train = rename_label_column(tokenized_train)
    tokenized_dev = rename_label_column(tokenized_dev)
    
    # Set format for PyTorch, keep only required columns
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_dev.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create DataLoaders
    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    dev_loader = DataLoader(
        tokenized_dev,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    return train_loader, dev_loader

#%% Training Functions
def train_epoch(model, train_loader, optimizer, epoch_desc="Training"):
    """Train one epoch using PyTorch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=epoch_desc)
    
    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        batch_correct = (predictions == batch['labels']).sum().item()
        correct += batch_correct
        total += batch['labels'].size(0)
        
        # Update progress bar
        accuracy = 100 * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    return total_loss / len(train_loader), 100 * correct / total

def evaluate(model, dev_loader, desc="Validation"):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc=desc):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    return total_loss / len(dev_loader), 100 * correct / total

#%% Initialize Experiment
print("Initializing BERT experiment...")

# Setup experiment parameters
task = 'cola'
subset_size = 1000
max_length = 64
batch_size = 16

# Load model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset
train_dataset, dev_dataset, text_column, label_column, num_labels = prepare_dataset(
    task=task,
    subset_size=subset_size,
    max_length=max_length
)

# Create model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False
).to(device)

# Create data loaders
train_loader, dev_loader = create_data_loaders(
    train_dataset, dev_dataset, tokenizer, text_column,
    batch_size=batch_size, max_length=max_length
)

print(f"Train batches: {len(train_loader)}")
print(f"Dev batches: {len(dev_loader)}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-5)

#%% Configure ntop Monitoring
output_folder = './outputs/bert_training_analysis'
os.makedirs(output_folder, exist_ok=True)

monitor = ActivationMonitor(model, model_type='transformer')
monitor.set_config(
    output_file=f'{output_folder}/topology_evolution.npz',
    max_samples=1000,
    distance_metric='euclidean',
    normalize_activations='none',
    max_dim=2,
    random_seed=42,
    filter_inactive_neurons=False,
    persistence_threshold=0.01,
    use_quantization=True,
    quantization_resolution=0.01,
    sequence_strategy='cls',
    analyze_full_network=False,
    analyze_by_layers=False,        
    analyze_by_components=True,
)

#%% Initial Analysis
print("\nAnalyzing initial BERT topology...")
initial_state = monitor.analyze(dev_loader, epoch=0, description="INITIAL_STATE", save=True)

#%% Training Loop
print(f"\nStarting BERT training on {task.upper()} task...")

epochs = 30
best_accuracy = 0
topology_evolution = []

# Track training metrics
train_losses = []
train_accuracies = []
dev_accuracies = []

print(f"Training for {epochs} epochs...")
overall_pbar = tqdm(range(epochs), desc="Overall Progress", unit="epoch")

for epoch in overall_pbar:
    start_time = time.time()
    
    # Training
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, 
        epoch_desc=f"Epoch {epoch+1}/{epochs}"
    )
    
    # Validation
    dev_loss, dev_acc = evaluate(model, dev_loader, desc="Validation")
    
    # Store metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    dev_accuracies.append(dev_acc)
    
    # Update best accuracy
    if dev_acc > best_accuracy:
        best_accuracy = dev_acc
    
    # Topology analysis at key epochs
    if epoch % 5 == 0 or epoch == epochs - 1 or epoch < 3:
        print(f"\nAnalyzing topology at epoch {epoch}...")
        
        state = monitor.analyze(
            dev_loader,
            epoch=epoch,
            description=f"EPOCH_{epoch}",
            save=True
        )
        topology_evolution.append((epoch, state, dev_acc))
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Dev Acc={dev_acc:.2f}%")
    
    # Update progress bar
    epoch_time = time.time() - start_time
    overall_pbar.set_postfix({
        'Train Acc': f'{train_acc:.1f}%',
        'Dev Acc': f'{dev_acc:.1f}%',
        'Best': f'{best_accuracy:.1f}%',
        'Time': f'{epoch_time:.1f}s'
    })

print(f"\nTraining complete! Best dev accuracy: {best_accuracy:.2f}%")

#%% Component Analysis
print(f"\nBERT Component Analysis")
print(f"="*50)

# Get final topology state
final_state = topology_evolution[-1][1]

# Analyze by components (attention, feedforward, etc.)
if 'by_components' in final_state:
    component_results = final_state['by_components']
    
    print(f"Component Analysis Results:")
    for comp_name, comp_result in component_results.items():
        rf_data = comp_result['rf_values']
        all_rf_values = []
        for layer_rf in rf_data.values():
            if isinstance(layer_rf, dict) and 'rf_0' in layer_rf:
                rf_0_values = np.array(layer_rf['rf_0'])
                all_rf_values.extend(rf_0_values[rf_0_values > 0])
        
        total_neurons = comp_result['total_neurons']
        betti = comp_result['betti_numbers']
        median_rf = np.median(all_rf_values) if all_rf_values else 0.0
        
        print(f"{comp_name.title()}: {total_neurons} neurons, "
              f"Betti: {betti}, RF median: {median_rf:.4f}")

#%% Generate Comprehensive Visualizations
print("Generating BERT training analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Training curves
ax = axes[0, 0]
epochs_range = range(len(train_losses))
ax.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Accuracy curves
ax = axes[0, 1]
ax.plot(epochs_range, train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
ax.plot(epochs_range, dev_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Component importance comparison
ax = axes[1, 0]
if 'by_components' in final_state:
    component_results = final_state['by_components']
    comp_names = list(component_results.keys())
    medians = []
    
    for comp_name in comp_names:
        rf_data = component_results[comp_name]['rf_values']
        all_rf_values = []
        for layer_rf in rf_data.values():
            if isinstance(layer_rf, dict) and 'rf_0' in layer_rf:
                rf_0_values = np.array(layer_rf['rf_0'])
                all_rf_values.extend(rf_0_values[rf_0_values > 0])
        medians.append(np.median(all_rf_values) if all_rf_values else 0.0)
    
    bars = ax.bar(comp_names, medians, alpha=0.8)
    ax.set_ylabel('Median RF Value')
    ax.set_title('Component Importance (Higher = More Important)')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.3f}', ha='center', va='bottom')

# 4. Topology evolution during training
ax = axes[1, 1]
if topology_evolution:
    epochs_tracked = [ep for ep, _, _ in topology_evolution]
    accuracies = [acc for _, _, acc in topology_evolution]
    
    # Get Betti numbers evolution
    betti_0_vals = []
    betti_1_vals = []
    
    for _, state, _ in topology_evolution:
        if 'by_components' in state:
            # Use attention component for demo (most stable)
            attention_comp = None
            for comp_name, comp_data in state['by_components'].items():
                if 'attention' in comp_name.lower():
                    attention_comp = comp_data
                    break
            
            if attention_comp is None:
                # Fallback to first component
                attention_comp = list(state['by_components'].values())[0]
            
            betti_0_vals.append(attention_comp['betti_numbers'].get(0, 0))
            betti_1_vals.append(attention_comp['betti_numbers'].get(1, 0))
        else:
            betti_0_vals.append(0)
            betti_1_vals.append(0)
    
    ax2 = ax.twinx()
    ax.plot(epochs_tracked, betti_0_vals, 'b-o', label='β₀ (Components)', markersize=4)
    ax.plot(epochs_tracked, betti_1_vals, 'r-s', label='β₁ (Loops)', markersize=4)
    ax2.plot(epochs_tracked, accuracies, 'g--^', label='Dev Accuracy', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Betti Numbers', color='black')
    ax2.set_ylabel('Dev Accuracy (%)', color='green')
    ax.set_title('Topology Evolution During Training')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.tight_layout()
plt.savefig(f'{output_folder}/bert_training_summary.png', dpi=150, bbox_inches='tight')
plt.show()

#%% Individual Component Analysis Plots
if 'by_components' in final_state:
    component_results = final_state['by_components']
    
    print("Generating individual component plots...")
    
    for comp_name, comp_result in component_results.items():
        # Persistence diagrams
        plots.plot_persistence_diagram(comp_result)
        plt.title(f'Persistence Diagram - {comp_name.title()}')
        plt.savefig(f'{output_folder}/persistence_{comp_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # t-SNE plots
        plots.plot_tsne_2d(comp_result)
        plt.title(f't-SNE - {comp_name.title()}')
        plt.savefig(f'{output_folder}/tsne_{comp_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Distance matrices
        plots.plot_distance_matrix(comp_result)
        plt.title(f'Distance Matrix - {comp_name.title()}')
        plt.savefig(f'{output_folder}/distance_{comp_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

#%% RF Evolution Analysis
print("Generating RF evolution plots...")
topology_states = monitor.topology_states

if topology_states:
    # RF evolution by component
    plots.plot_rf_distribution_evolution(topology_states, 'rf_0')
    plt.suptitle('BERT RF Distribution Evolution by Component')
    plt.savefig(f'{output_folder}/rf_distribution_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # RF statistics evolution
    plots.plot_rf_box_evolution(topology_states, 'rf_0')
    plt.suptitle('BERT RF Statistics Evolution by Component')
    plt.savefig(f'{output_folder}/rf_box_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

#%% Compression Analysis
print(f"\nBERT Compression Analysis")
print(f"="*50)

if 'by_components' in final_state:
    component_results = final_state['by_components']
    total_neurons = sum(comp_result['total_neurons'] for comp_result in component_results.values())
    print(f"Total neurons analyzed: {total_neurons:,}")
    
    for comp_name, comp_result in component_results.items():
        rf_data = comp_result['rf_values']
        all_rf_values = []
        for layer_rf in rf_data.values():
            if isinstance(layer_rf, dict) and 'rf_0' in layer_rf:
                rf_0_values = np.array(layer_rf['rf_0'])
                all_rf_values.extend(rf_0_values)
        
        if all_rf_values:
            all_rf_values = np.array(all_rf_values)
            active_values = all_rf_values[all_rf_values > 0]
            if len(active_values) > 0:
                p30 = np.percentile(active_values, 30)
                p50 = np.percentile(active_values, 50)
                prunable_30 = sum(1 for val in all_rf_values if val <= p30)
                prunable_50 = sum(1 for val in all_rf_values if val <= p50)
                
                print(f"{comp_name}: 30th percentile prunable: {prunable_30/len(all_rf_values)*100:.1f}%, "
                      f"50th: {prunable_50/len(all_rf_values)*100:.1f}%")

# Save training results
training_results = {
    'task': task,
    'model_name': model_name,
    'best_accuracy': best_accuracy,
    'epochs': epochs,
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'dev_accuracies': dev_accuracies,
    'topology_evolution': topology_evolution
}

results_file = f'{output_folder}/training_results.npz'
np.savez_compressed(results_file, **training_results)
print(f"Training results saved to: {results_file}")

monitor.remove_hooks()
print(f"\nBERT training analysis complete!")
print(f"Results saved to: {output_folder}")
print(f"Final performance: {best_accuracy:.2f}% accuracy on {task.upper()} validation set")

#%%