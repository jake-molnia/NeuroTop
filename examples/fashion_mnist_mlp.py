#%% Setup
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

import ntop
from ntop.monitoring import ActivationMonitor
from ntop import analysis, plots

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

#%% Model Definition
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

#%% Load Data and Model
def setup_data(batch_size=128, subset_size=8000):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    if subset_size:
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        test_indices = torch.randperm(len(test_dataset))[:subset_size//4]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0), 
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0))

# Initialize model and data
model = FashionMLP().to(device).eval()
train_loader, test_loader = setup_data()
total_params = sum(p.numel() for p in model.parameters())

print(f"Fashion-MNIST MLP Analysis")
print(f"Device: {device}")
print(f"Model parameters: {total_params:,}")
print(f"Data: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")

#%% Configure Analysis
output_folder = './outputs/fashion_mnist_analysis'
os.makedirs(output_folder, exist_ok=True)

monitor = ActivationMonitor(model)
monitor.set_config(
    output_file=f'{output_folder}/topology_analysis.npz',
    max_samples=1500,
    distance_metric='manhattan',
    normalize_activations='none',
    max_dim=1,
    random_seed=42,
    filter_inactive_neurons=True,
    persistence_threshold=0.01,
    use_quantization=True, 
    quantization_resolution=0.1,
)

#%% Run Initial Analysis
print("\nAnalyzing initial network topology...")
with torch.no_grad():
    initial_state = monitor.analyze(test_loader, 0, "INITIAL_STATE")

#%% Training Loop (Following Paper's Approach)
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

# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
epochs = 30

# Track topology evolution during training
print(f"\nTraining for {epochs} epochs with topology monitoring...")
best_accuracy = 0
topology_evolution = []

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    test_acc = evaluate(model, test_loader, device)
    scheduler.step()
    
    # Analyze topology at key epochs
    if epoch % 5 == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            state = monitor.analyze(test_loader, epoch, save=True)
            topology_evolution.append((epoch, state, test_acc))
        print(f"Epoch {epoch:2d}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
    
    if test_acc > best_accuracy: 
        best_accuracy = test_acc

print(f"Training complete! Best test accuracy: {best_accuracy:.2f}%")

#%% Extract RF Values by Layer (Key Finding)
final_state = topology_evolution[-1][1]
rf_data = final_state['rf_values']
layer_stats = {}

for layer_name, layer_rf in rf_data.items():
    # Categorize layers
    if 'layers.0' in layer_name:
        layer_type = 'Input_Layer'
    elif 'layers.3' in layer_name:
        layer_type = 'Hidden_Layer_1'
    elif 'layers.6' in layer_name:
        layer_type = 'Hidden_Layer_2'
    elif 'layers.9' in layer_name:
        layer_type = 'Hidden_Layer_3'
    elif 'layers.11' in layer_name:
        layer_type = 'Output_Layer'
    else:
        layer_type = 'Other'
    
    if layer_type not in layer_stats:
        layer_stats[layer_type] = []
    
    rf_0_values = np.array(layer_rf['rf_0'])
    layer_stats[layer_type].extend(rf_0_values[rf_0_values > 0])

#%% Results Summary (Following Paper's Analysis)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Layer RF distributions
ax = axes[0,0]
for layer_type, values in layer_stats.items():
    if len(values) > 0:
        ax.hist(values, bins=30, alpha=0.6, label=f'{layer_type} (med: {np.median(values):.3f})')
ax.set_xlabel('RF Value')
ax.set_ylabel('Count')
ax.set_title('RF Distribution by Layer Type')
ax.legend()

# Layer importance (Key insight)
ax = axes[0,1]
layer_names = list(layer_stats.keys())
medians = [np.median(layer_stats[layer]) if len(layer_stats[layer]) > 0 else 0 
           for layer in layer_names]
bars = ax.bar(layer_names, medians)
ax.set_ylabel('Median RF Value')
ax.set_title('Layer Importance (Higher = More Important)')
ax.tick_params(axis='x', rotation=45)

# Add values on bars
for bar, val in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
            f'{val:.3f}', ha='center', va='bottom')

# Persistence diagram
ax = axes[1,0]
diagrams = final_state['persistence']['dgms']
colors = ['red', 'blue', 'green']
for dim, diagram in enumerate(diagrams):
    if len(diagram) > 0:
        finite_mask = ~np.isinf(diagram[:, 1])
        finite_points = diagram[finite_mask]
        if len(finite_points) > 0:
            ax.scatter(finite_points[:, 0], finite_points[:, 1], 
                      c=colors[dim], alpha=0.7, s=20, label=f'H{dim}')
ax.set_xlabel('Birth')
ax.set_ylabel('Death')
ax.set_title('Persistence Diagram')
ax.legend()

# Topology evolution during training
ax = axes[1,1]
epochs_tracked = [ep for ep, _, _ in topology_evolution]
accuracies = [acc for _, _, acc in topology_evolution]
betti_0 = [state['betti_numbers'][0] for _, state, _ in topology_evolution]
betti_1 = [state['betti_numbers'][1] for _, state, _ in topology_evolution]

ax2 = ax.twinx()
ax.plot(epochs_tracked, betti_0, 'b-o', label='β₀ (Components)', markersize=4)
ax.plot(epochs_tracked, betti_1, 'r-s', label='β₁ (Loops)', markersize=4)
ax2.plot(epochs_tracked, accuracies, 'g--^', label='Test Accuracy', markersize=4)

ax.set_xlabel('Epoch')
ax.set_ylabel('Betti Numbers', color='black')
ax2.set_ylabel('Test Accuracy (%)', color='green')
ax.set_title('Topology Evolution During Training')

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.tight_layout()
plt.savefig(f'{output_folder}/fashion_analysis_summary.png', dpi=150, bbox_inches='tight')
plt.show()

#%% Neuron Importance Analysis
all_rf_values = []
neuron_info = []

for layer_name, layer_rf in rf_data.items():
    rf_vals = np.array(layer_rf['rf_0'])
    for i, val in enumerate(rf_vals):
        all_rf_values.append(val)
        neuron_info.append((layer_name, i, val))

all_rf_values = np.array(all_rf_values)
p30 = np.percentile(all_rf_values[all_rf_values > 0], 30)
p50 = np.percentile(all_rf_values[all_rf_values > 0], 50)
p70 = np.percentile(all_rf_values[all_rf_values > 0], 70)

prunable_30 = sum(1 for val in all_rf_values if val <= p30)
prunable_50 = sum(1 for val in all_rf_values if val <= p50)
prunable_70 = sum(1 for val in all_rf_values if val <= p70)

total_neurons = len(all_rf_values)

print(f"Fashion-MNIST MLP Compression Analysis")
print(f"="*50)
print(f"Total neurons: {total_neurons}")
print(f"Prunable at 30th percentile: {prunable_30} ({prunable_30/total_neurons*100:.1f}%)")
print(f"Prunable at 50th percentile: {prunable_50} ({prunable_50/total_neurons*100:.1f}%)")
print(f"Prunable at 70th percentile: {prunable_70} ({prunable_70/total_neurons*100:.1f}%)")
print(f"RF thresholds - 30th: {p30:.4f}, 50th: {p50:.4f}, 70th: {p70:.4f}")

# Layer-wise importance ranking
print(f"\nLayer Importance Ranking (Median RF):")
layer_medians = [(layer, np.median(vals)) for layer, vals in layer_stats.items() if len(vals) > 0]
layer_medians.sort(key=lambda x: x[1], reverse=True)
for layer, median in layer_medians:
    print(f"{layer}: {median:.4f}")

#%% Network Topology Summary
initial_betti = initial_state['betti_numbers']
final_betti = final_state['betti_numbers']

print(f"\nTopological Evolution:")
print(f"Initial topology - Connected components (β₀): {initial_betti[0]}, Loops (β₁): {initial_betti[1]}")
print(f"Final topology - Connected components (β₀): {final_betti[0]}, Loops (β₁): {final_betti[1]}")
print(f"Total neurons analyzed: {final_state['total_neurons']}")
print(f"Final test accuracy: {best_accuracy:.2f}%")

#%% Display Topology Visualizations
print(f"\nDisplaying topology visualizations...")
topology_states = monitor.topology_states

# Core topology plots
plots.plot_distance_matrix(final_state)
plt.show()

plots.plot_persistence_diagram(final_state)
plt.show()

plots.plot_tsne_2d(final_state)
plt.show()

plots.plot_betti_numbers(final_state)
plt.show()

plots.plot_layer_composition(final_state)
plt.show()

# RF evolution analysis plots
plots.plot_rf_heatmap_by_layer(topology_states, 'rf_0')
plt.show()

plots.plot_rf_heatmap_network(topology_states, 'rf_0')
plt.show()

plots.plot_rf_distribution_evolution(topology_states)
plt.show()

plots.plot_rf_box_evolution(topology_states)
plt.show()

plots.plot_rf_evolution_comparison(topology_states, 'rf_0')
plt.show()

monitor.remove_hooks()
print(f"\nAnalysis complete!")
# %%
