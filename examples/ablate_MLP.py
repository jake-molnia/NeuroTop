# examples/test_mlp_ablation.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import ntop
from ntop.monitoring import ActivationMonitor

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x): 
        return self.layers(x.view(x.size(0), -1))

def setup_data(batch_size=128, subset_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    if subset_size:
        train_indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        test_indices = torch.randperm(len(test_dataset))[:subset_size//4].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0), 
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0))

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total

def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# %% Cell 1: Model Setup and Training
print("Setting up model and data...")

# Initialize model and data
model = FashionMLP().to(device)
train_loader, test_loader = setup_data()

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
epochs = 50

print("Training model...")

# Add progress bar for epochs
epoch_pbar = tqdm(range(epochs), desc="Epochs", unit="epoch")

for epoch in epoch_pbar:
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # Update epoch progress bar description
    epoch_pbar.set_postfix({'Loss': f'{train_loss:.4f}', 'Acc': f'{train_acc:.2f}%'})
    
    if epoch % 5 == 0 or epoch == epochs - 1:
        test_acc = evaluate(model, test_loader, device)
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}', 
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%'
        })

# Evaluate original model
original_accuracy = evaluate(model, test_loader, device)
print(f"\nOriginal Model Accuracy: {original_accuracy:.2f}%")

# %% Cell 2: Ablation Analysis
print("\nSetting up monitoring and analyzing model topology...")

# Setup monitoring and analysis
monitor = ActivationMonitor(model)
monitor.set_config(
    output_file='./outputs/ablation_test.npz',
    max_samples=500,
    distance_metric='euclidean',
    normalize_activations='none',
    max_dim=1,
    random_seed=42,
    filter_inactive_neurons=False,
    persistence_threshold=0.01,
    use_quantization=False,
    analyze_full_network=True,
)

# Analyze the trained model
model.eval()
with torch.no_grad():
    state = monitor.analyze(test_loader, epoch=0, description="POST_TRAINING")

# Test different ablation strategies and percentages - EVERY SINGLE PERCENTAGE POINT!
strategies = ['percent', 'random']
percentages = list(range(1, 101))  # 1% to 100%

# Store results for comparison table
results = {}
for strategy in strategies:
    results[strategy] = {}

print(f"\nRunning ablation tests for {len(percentages)} percentage points...")
print("This will take a while - testing every percentage from 1% to 100%!")

# Create progress bar for the overall ablation testing
total_tests = len(strategies) * len(percentages)
pbar = tqdm(total=total_tests, desc="Ablation Testing", unit="test")

for strategy in strategies:
    for percentage in percentages:
        ablated_model = monitor.ablate(model, strategy, percentage, rf_dim='rf_0')
        ablated_accuracy = evaluate(ablated_model, test_loader, device)
        accuracy_drop = original_accuracy - ablated_accuracy        
        results[strategy][percentage] = {
            'accuracy': ablated_accuracy,
            'drop': accuracy_drop
        }
        pbar.update(1)

pbar.close()

# Display summary for key percentages (since showing all 100 would be overwhelming)
summary_percentages = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
print(f"\nAblation Strategy Comparison (Summary of Key Percentages):")
print(f"Original Model Accuracy: {original_accuracy:.2f}%")
print(f"Total data points collected: {len(percentages)} percentage points\n")

# Header
print(f"{'Percentage':<12} {'rf Score (percent)':<25} {'Random Ablation':<25} {'Difference':<12}")
print(f"{'Removed':<12} {'Acc (Drop)':<25} {'Acc (Drop)':<25} {'(rf - Random)':<12}")
print("-" * 80)

# Data rows for summary
for percentage in summary_percentages:
    if percentage in results['percent']:
        percent_acc = results['percent'][percentage]['accuracy']
        percent_drop = results['percent'][percentage]['drop']
        random_acc = results['random'][percentage]['accuracy']
        random_drop = results['random'][percentage]['drop']
        
        difference = percent_acc - random_acc
        
        # Show change from baseline (negative = drop, positive = improvement)
        percent_change = percent_acc - original_accuracy
        random_change = random_acc - original_accuracy
        
        print(f"{percentage}%{'':<9} "
              f"{percent_acc:.2f}% ({percent_change:+.2f}%){'':<8} "
              f"{random_acc:.2f}% ({random_change:+.2f}%){'':<8} "
              f"{difference:+.2f}%")

# Cleanup
monitor.remove_hooks()
print("\nAblation test complete!")

# %% Cell 3: Visualization
print("\nGenerating ablation performance plots for all 100 percentage points...")

# Prepare data for plotting
rf_accuracies = [results['percent'][p]['accuracy'] for p in percentages]
random_accuracies = [results['random'][p]['accuracy'] for p in percentages]

# Create the plot
plt.figure(figsize=(14, 8))

# Plot both strategies - using lines without markers for clean visualization
plt.plot(percentages, rf_accuracies, '-', label='RF Score Ablation', linewidth=2, color='blue', alpha=0.8)
plt.plot(percentages, random_accuracies, '-', label='Random Ablation', linewidth=2, color='red', alpha=0.8)

# Add horizontal line for original accuracy
plt.axhline(y=original_accuracy, color='green', linestyle='--', alpha=0.7, 
           label=f'Original Model ({original_accuracy:.2f}%)', linewidth=2)

# Customize the plot
plt.xlabel('Percentage of Neurons Removed (%)', fontsize=12)
plt.ylabel('Model Accuracy (%)', fontsize=12)
plt.title('Model Performance vs. Neuron Ablation Strategy (All Percentage Points)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Set axis limits for better visualization
plt.xlim(0, 100)
plt.ylim(0, max(original_accuracy, max(rf_accuracies), max(random_accuracies)) + 5)

# Add annotations for key differences at specific points
key_points = [10, 25, 50, 75, 90]
for point in key_points:
    if point < len(percentages):
        diff = rf_accuracies[point-1] - random_accuracies[point-1]  # -1 because list is 0-indexed
        if abs(diff) > 1:  # Only annotate significant differences
            plt.annotate(f'{diff:+.1f}%', 
                        xy=(point, max(rf_accuracies[point-1], random_accuracies[point-1])), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=9, alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()

# Save the plot
os.makedirs('./outputs/ablation_analysis', exist_ok=True)
plt.savefig('./outputs/ablation_analysis/ablation_performance_comparison.png', 
           dpi=300, bbox_inches='tight')
plt.show()
#%% 