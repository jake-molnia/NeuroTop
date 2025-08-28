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
print(f"BERT Ablation Testing with RF-Guided Pruning (PyTorch + HuggingFace)")
print(f"Device: {device}")

#%% Data Setup Functions
def prepare_dataset(dataset_name='cola', subset_size=1000, max_length=64):
    """Prepare GLUE dataset with HuggingFace"""
    print(f"Loading {dataset_name} dataset...")
    
    # Load dataset
    if dataset_name == 'cola':
        dataset = load_dataset('glue', 'cola')
        text_column = 'sentence'
        label_column = 'label'
    elif dataset_name == 'sst2':
        dataset = load_dataset('glue', 'sst2')
        text_column = 'sentence'
        label_column = 'label'
    elif dataset_name == 'mrpc':
        dataset = load_dataset('glue', 'mrpc')
        text_column = ['sentence1', 'sentence2']
        label_column = 'label'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Apply subset if specified
    if subset_size:
        train_dataset = dataset['train'].select(range(min(subset_size, len(dataset['train']))))
        validation_dataset = dataset['validation'].select(range(min(subset_size//4, len(dataset['validation']))))
    else:
        train_dataset = dataset['train']
        validation_dataset = dataset['validation']
    
    return train_dataset, validation_dataset, text_column, label_column

def tokenize_function(examples, tokenizer, text_column, max_length=64):
    """Tokenize texts using HuggingFace tokenizer"""
    if isinstance(text_column, list):
        # For sentence pair tasks like MRPC
        return tokenizer(
            examples[text_column[0]], 
            examples[text_column[1]],
            truncation=True,
            padding=False,  # We'll pad dynamically in DataLoader
            max_length=max_length
        )
    else:
        # For single sentence tasks like CoLA
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,  # We'll pad dynamically in DataLoader
            max_length=max_length
        )

def create_data_loaders(train_dataset, val_dataset, tokenizer, text_column, batch_size=16, max_length=64):
    """Create PyTorch DataLoaders"""
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, text_column, max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_val = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, text_column, max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Set format for PyTorch
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create DataLoaders
    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    val_loader = DataLoader(
        tokenized_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    return train_loader, val_loader

def evaluate_model(model, test_loader, description="Evaluation"):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=description, unit="batch", leave=False):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    model.train()  # Reset to training mode
    return 100 * correct / total

#%% Cell 1: Model Setup and Initial Analysis
print("Setting up BERT model for ablation testing...")

# Load model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
).to(device)

# Prepare dataset
train_dataset, val_dataset, text_column, label_column = prepare_dataset(
    dataset_name='cola',
    subset_size=800,  # Reasonable subset for testing
    max_length=64
)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    train_dataset, val_dataset, tokenizer, text_column,
    batch_size=16, max_length=64
)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Quick training simulation to get a reasonable model
print("Simulating trained model (running brief training simulation)...")
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

model.train()
for i, batch in enumerate(train_loader):
    if i >= 5:  # Just a few steps
        break
    
    batch = {k: v.to(device) for k, v in batch.items()}
    
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()

# Evaluate original model performance
print("Evaluating original model performance...")
original_accuracy = evaluate_model(model, val_loader, "Original Model")
print(f"Original Model Accuracy: {original_accuracy:.2f}%")

#%% Cell 2: Topology Analysis and RF Computation
print("\nSetting up topology monitoring and analyzing BERT structure...")

# Setup monitoring for HuggingFace BERT
monitor = ActivationMonitor(model, model_type='transformer')
monitor.set_config(
    output_file='./outputs/bert_ablation_test.npz',
    max_samples=500,
    distance_metric='euclidean',
    normalize_activations='none',
    max_dim=1,
    random_seed=42,
    filter_inactive_neurons=False,
    persistence_threshold=0.01,
    use_quantization=True,
    quantization_resolution=0.1,
    sequence_strategy='cls',
    analyze_full_network=False,
    analyze_by_components=True,
)

# Analyze the model to get RF values
print("Computing RF values for all components...")
state = monitor.analyze(
    val_loader, 
    epoch=0, 
    description="ABLATION_ANALYSIS"
)

# Print component analysis summary
if 'by_components' in state:
    component_results = state['by_components']
    print(f"\nComponent Analysis Summary:")
    for comp_name, comp_result in component_results.items():
        total_neurons = comp_result['total_neurons']
        betti = comp_result['betti_numbers']
        
        # Get RF statistics
        rf_data = comp_result['rf_values']
        all_rf_values = []
        for layer_rf in rf_data.values():
            if isinstance(layer_rf, dict) and 'rf_0' in layer_rf:
                rf_0_values = np.array(layer_rf['rf_0'])
                all_rf_values.extend(rf_0_values[rf_0_values > 0])
        
        median_rf = np.median(all_rf_values) if all_rf_values else 0.0
        print(f"{comp_name.title()}: {total_neurons} neurons, Betti: {betti}, RF median: {median_rf:.4f}")

#%% Cell 3: Comprehensive Ablation Testing
print(f"\nRunning comprehensive ablation tests...")

# Test different ablation strategies and percentages
strategies = ['percent', 'random']
percentages = list(range(5, 101, 5))  # Test every 5% from 5% to 100%

# Store results
results = {}
for strategy in strategies:
    results[strategy] = {}

print(f"Testing {len(percentages)} percentage points for {len(strategies)} strategies...")
print("This comprehensive test will take significant time!")

# Create overall progress bar
total_tests = len(strategies) * len(percentages)
pbar = tqdm(total=total_tests, desc="Ablation Testing", unit="test")

for strategy in strategies:
    print(f"\nTesting {strategy} ablation strategy...")
    
    for percentage in percentages:
        try:
            # Create ablated model using ntop
            ablated_model = monitor.ablate(model, strategy, percentage, rf_dim='rf_0')
            
            # Evaluate ablated model
            ablated_accuracy = evaluate_model(
                ablated_model, val_loader, 
                f"{strategy.title()} {percentage}%"
            )
            
            accuracy_drop = original_accuracy - ablated_accuracy
            
            results[strategy][percentage] = {
                'accuracy': ablated_accuracy,
                'drop': accuracy_drop,
                'relative_drop': accuracy_drop / original_accuracy * 100 if original_accuracy > 0 else 0
            }
            
        except Exception as e:
            print(f"Error at {strategy} {percentage}%: {e}")
            results[strategy][percentage] = {
                'accuracy': 0.0,
                'drop': original_accuracy,
                'relative_drop': 100.0
            }
        
        pbar.update(1)

pbar.close()

#%% Cell 4: Results Analysis and Summary
print(f"\nAblation Testing Results Summary")
print(f"="*80)
print(f"Original BERT Model Accuracy: {original_accuracy:.2f}%")
print(f"Total test configurations: {total_tests}")

# Display comprehensive results table
print(f"\nDetailed Ablation Strategy Comparison:")
print(f"{'Percentage':<12} {'RF-Guided (percent)':<30} {'Random Ablation':<30} {'Difference':<12}")
print(f"{'Removed':<12} {'Accuracy (Drop)':<30} {'Accuracy (Drop)':<30} {'(RF - Random)':<12}")
print("-" * 84)

# Show every 10% for readability
display_percentages = list(range(10, 101, 10))
for percentage in display_percentages:
    if percentage in results['percent'] and percentage in results['random']:
        rf_acc = results['percent'][percentage]['accuracy']
        rf_drop = results['percent'][percentage]['drop']
        random_acc = results['random'][percentage]['accuracy']
        random_drop = results['random'][percentage]['drop']
        
        difference = rf_acc - random_acc
        
        print(f"{percentage}%{'':<9} "
              f"{rf_acc:.2f}% ({rf_drop:+.2f}%){'':<14} "
              f"{random_acc:.2f}% ({random_drop:+.2f}%){'':<14} "
              f"{difference:+.2f}%")

# Find key insights
print(f"\nKey Insights:")

# Find breakeven points
breakeven_points = []
for percentage in percentages:
    if percentage in results['percent'] and percentage in results['random']:
        rf_acc = results['percent'][percentage]['accuracy']
        random_acc = results['random'][percentage]['accuracy']
        if abs(rf_acc - random_acc) < 1.0:  # Within 1% difference
            breakeven_points.append(percentage)

if breakeven_points:
    print(f"RF-guided and random pruning perform similarly at: {breakeven_points}% neuron removal")

# Find maximum prunable percentage while maintaining reasonable performance
performance_threshold = original_accuracy * 0.8  # 80% of original performance
max_prunable_rf = max_prunable_random = 0

for percentage in percentages:
    if percentage in results['percent']:
        if results['percent'][percentage]['accuracy'] >= performance_threshold:
            max_prunable_rf = percentage
    if percentage in results['random']:
        if results['random'][percentage]['accuracy'] >= performance_threshold:
            max_prunable_random = percentage

print(f"Maximum prunable (80% performance): RF-guided={max_prunable_rf}%, Random={max_prunable_random}%")

# Calculate average advantage
avg_advantage = 0
valid_comparisons = 0
for percentage in percentages:
    if percentage in results['percent'] and percentage in results['random']:
        rf_acc = results['percent'][percentage]['accuracy']
        random_acc = results['random'][percentage]['accuracy']
        avg_advantage += (rf_acc - random_acc)
        valid_comparisons += 1

if valid_comparisons > 0:
    avg_advantage /= valid_comparisons
    print(f"Average RF-guided advantage: {avg_advantage:+.2f}% accuracy")

#%% Cell 5: Component-Specific Ablation Analysis
print(f"\nComponent-Specific Ablation Analysis")
print(f"="*50)

if 'by_components' in state:
    component_results = state['by_components']
    
    # Test ablation by component
    component_ablation_results = {}
    
    for comp_name, comp_result in component_results.items():
        print(f"\nTesting {comp_name} component ablation (50% removal)...")
        
        try:
            # Test 50% ablation for each component
            component_state = {'by_components': {comp_name: comp_result}}
            ablated_model = monitor.ablate(model, 'percent', 50, rf_dim='rf_0', state=component_state)
            component_accuracy = evaluate_model(ablated_model, val_loader, f"{comp_name} 50%")
            
            component_ablation_results[comp_name] = {
                'accuracy': component_accuracy,
                'drop': original_accuracy - component_accuracy,
                'neurons': comp_result['total_neurons']
            }
            
            print(f"{comp_name.title()}: {component_accuracy:.2f}% ({original_accuracy - component_accuracy:+.2f}% drop)")
            
        except Exception as e:
            print(f"Error ablating {comp_name}: {e}")
            component_ablation_results[comp_name] = {
                'accuracy': 0.0,
                'drop': original_accuracy,
                'neurons': comp_result['total_neurons']
            }
    
    # Find most/least critical components
    if component_ablation_results:
        most_critical = max(component_ablation_results.items(), key=lambda x: x[1]['drop'])
        least_critical = min(component_ablation_results.items(), key=lambda x: x[1]['drop'])
        
        print(f"\nMost critical component: {most_critical[0]} ({most_critical[1]['drop']:.2f}% drop)")
        print(f"Least critical component: {least_critical[0]} ({least_critical[1]['drop']:.2f}% drop)")

#%% Cell 6: Comprehensive Visualization
print(f"\nGenerating comprehensive ablation analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy vs Pruning Percentage
ax = axes[0, 0]
rf_accuracies = [results['percent'][p]['accuracy'] for p in percentages if p in results['percent']]
random_accuracies = [results['random'][p]['accuracy'] for p in percentages if p in results['random']]

ax.plot(percentages[:len(rf_accuracies)], rf_accuracies, 'b-o', 
        label='RF-Guided Ablation', linewidth=2, markersize=4)
ax.plot(percentages[:len(random_accuracies)], random_accuracies, 'r-s', 
        label='Random Ablation', linewidth=2, markersize=4)
ax.axhline(y=original_accuracy, color='green', linestyle='--', alpha=0.7, 
           label=f'Original ({original_accuracy:.1f}%)', linewidth=2)

ax.set_xlabel('Percentage of Neurons Removed (%)')
ax.set_ylabel('Model Accuracy (%)')
ax.set_title('BERT Performance vs. Neuron Ablation Strategy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)

# 2. Performance Drop Comparison
ax = axes[0, 1]
rf_drops = [results['percent'][p]['drop'] for p in percentages if p in results['percent']]
random_drops = [results['random'][p]['drop'] for p in percentages if p in results['random']]

ax.plot(percentages[:len(rf_drops)], rf_drops, 'b-o', 
        label='RF-Guided Drop', linewidth=2, markersize=4)
ax.plot(percentages[:len(random_drops)], random_drops, 'r-s', 
        label='Random Drop', linewidth=2, markersize=4)

ax.set_xlabel('Percentage of Neurons Removed (%)')
ax.set_ylabel('Accuracy Drop (%)')
ax.set_title('Performance Degradation Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)

# 3. Advantage of RF-Guided Pruning
ax = axes[1, 0]
advantages = []
valid_percentages = []
for p in percentages:
    if p in results['percent'] and p in results['random']:
        advantage = results['percent'][p]['accuracy'] - results['random'][p]['accuracy']
        advantages.append(advantage)
        valid_percentages.append(p)

ax.plot(valid_percentages, advantages, 'g-o', linewidth=2, markersize=4)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax.fill_between(valid_percentages, 0, advantages, where=[a > 0 for a in advantages], 
                alpha=0.3, color='green', label='RF Advantage')
ax.fill_between(valid_percentages, 0, advantages, where=[a < 0 for a in advantages], 
                alpha=0.3, color='red', label='Random Advantage')

ax.set_xlabel('Percentage of Neurons Removed (%)')
ax.set_ylabel('RF-Guided Advantage (%)')
ax.set_title('RF-Guided vs Random Pruning Advantage')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Component Ablation Results
ax = axes[1, 1]
if 'component_ablation_results' in locals() and component_ablation_results:
    comp_names = list(component_ablation_results.keys())
    comp_drops = [component_ablation_results[comp]['drop'] for comp in comp_names]
    
    bars = ax.bar(comp_names, comp_drops, alpha=0.8, 
                  color=['blue', 'orange', 'green', 'red', 'purple'][:len(comp_names)])
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('Component Importance (50% Ablation)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add values on bars
    for bar, drop in zip(bars, comp_drops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{drop:.1f}%', ha='center', va='bottom')
else:
    ax.text(0.5, 0.5, 'Component analysis\nnot available', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Component Ablation Analysis')

plt.tight_layout()

# Save plot
output_dir = './outputs/bert_ablation_analysis'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/bert_ablation_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Cell 7: Advanced Analysis and Export
print(f"\nGenerating advanced ablation analysis...")

# Create detailed breakdown by performance ranges
performance_ranges = {
    'minimal_loss': (original_accuracy * 0.95, 100),    # 95-100% of original
    'acceptable_loss': (original_accuracy * 0.85, original_accuracy * 0.95),  # 85-95%
    'significant_loss': (original_accuracy * 0.70, original_accuracy * 0.85), # 70-85%
    'severe_loss': (0, original_accuracy * 0.70)       # <70%
}

print("\nPerformance Impact Analysis:")
for range_name, (min_acc, max_acc) in performance_ranges.items():
    rf_count = random_count = 0
    rf_percentages = []
    random_percentages = []
    
    for percentage in percentages:
        if percentage in results['percent']:
            rf_acc = results['percent'][percentage]['accuracy']
            if min_acc <= rf_acc < max_acc:
                rf_count += 1
                rf_percentages.append(percentage)
        
        if percentage in results['random']:
            random_acc = results['random'][percentage]['accuracy']
            if min_acc <= random_acc < max_acc:
                random_count += 1
                random_percentages.append(percentage)
    
    print(f"{range_name.replace('_', ' ').title()}: "
          f"RF-guided: {rf_count} configs ({rf_percentages[:3] if rf_percentages else []}...), "
          f"Random: {random_count} configs ({random_percentages[:3] if random_percentages else []}...)")

# Save detailed results
results_file = f'{output_dir}/bert_ablation_detailed_results.txt'
with open(results_file, 'w') as f:
    f.write("BERT Ablation Test Results - Comprehensive Analysis\n")
    f.write("="*60 + "\n")
    f.write(f"Original Model Accuracy: {original_accuracy:.2f}%\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Dataset: CoLA\n")
    f.write(f"Total Tests Performed: {total_tests}\n")
    f.write(f"Average RF-guided Advantage: {avg_advantage:+.2f}%\n\n")
    
    f.write("Complete Results by Percentage:\n")
    f.write("Percentage\tRF_Accuracy\tRF_Drop\tRandom_Accuracy\tRandom_Drop\tAdvantage\n")
    for percentage in percentages:
        if percentage in results['percent'] and percentage in results['random']:
            rf_acc = results['percent'][percentage]['accuracy']
            rf_drop = results['percent'][percentage]['drop']
            random_acc = results['random'][percentage]['accuracy']
            random_drop = results['random'][percentage]['drop']
            advantage = rf_acc - random_acc
            
            f.write(f"{percentage}\t{rf_acc:.2f}\t{rf_drop:.2f}\t{random_acc:.2f}\t{random_drop:.2f}\t{advantage:+.2f}\n")
    
    if 'component_ablation_results' in locals() and component_ablation_results:
        f.write("\nComponent Ablation Results (50%):\n")
        f.write("Component\tAccuracy\tDrop\tNeurons\n")
        for comp_name, comp_data in component_ablation_results.items():
            f.write(f"{comp_name}\t{comp_data['accuracy']:.2f}\t{comp_data['drop']:.2f}\t{comp_data['neurons']}\n")

print(f"Detailed results saved to: {results_file}")

# Cleanup
monitor.remove_hooks()
print(f"\nBERT ablation testing complete!")

#%%