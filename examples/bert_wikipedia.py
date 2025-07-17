#%% Setup
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import os

import ntop
from ntop.monitoring import ActivationMonitor
from ntop import analysis, plots

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

#%% Load Data and Model
def setup_data(subset_size=150):
    """Load and prepare CoLA dataset from GLUE (following paper approach)"""
    dataset = load_dataset('glue', 'cola', split='validation')
    # Convert to list to handle indexing properly
    texts = []
    for i, item in enumerate(dataset):
        if i >= subset_size:
            break
        texts.append(item['sentence'])
    return texts

def setup_model():
    """Initialize BERT model and tokenizer"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()
    return model, tokenizer

# Initialize model and data
model, tokenizer = setup_model()
texts = setup_data()
total_params = sum(p.numel() for p in model.parameters())

print(f"BERT Topological Analysis")
print(f"Device: {device}")
print(f"Model parameters: {total_params:,}")
print(f"Data: {len(texts)} text samples")

# Tokenize data
encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

#%% Configure Analysis
output_folder = './outputs/bert_topology_analysis'
os.makedirs(output_folder, exist_ok=True)

monitor = ActivationMonitor(model, model_type='transformer')
monitor.set_config(
    output_file=f'{output_folder}/topology_analysis.npz',
    max_samples=150,
    distance_metric='euclidean',
    normalize_activations='none',
    max_dim=2,
    random_seed=42,
    filter_inactive_neurons=True,
    persistence_threshold=0.01,
    use_quantization=True,
    quantization_resolution=1000000000,
    sequence_strategy='cls',
    rf_distribution_strategy='uniform'
)

#%% Run BERT Analysis
print("\nAnalyzing BERT network topology...")
with torch.no_grad():
    # Move inputs to device
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Run model and capture activations
    outputs = model(input_ids, attention_mask=attention_mask)
    
    # Get activations and analyze
    activations = monitor.get_activations()
    result = analysis.analyze(
        activations, 
        max_dim=2, 
        distance_metric='euclidean',
        use_quantization=True, 
        quantization_resolution=0.1
    )
#%% Extract RF Values by Component (Paper's Key Finding)
rf_data = result['rf_values']
component_stats = {}

for layer_name, layer_rf in rf_data.items():
    if 'intermediate' in layer_name.lower():
        component_type = 'Intermediate'
    elif 'attention.output' in layer_name.lower():
        component_type = 'Attention_Output'
    elif 'output' in layer_name.lower():
        component_type = 'Output'
    elif 'query' in layer_name.lower() or '.q.' in layer_name:
        component_type = 'Query'
    elif 'key' in layer_name.lower() or '.k.' in layer_name:
        component_type = 'Key'
    elif 'value' in layer_name.lower() or '.v.' in layer_name:
        component_type = 'Value'
    else:
        component_type = 'Other'
    
    if component_type not in component_stats:
        component_stats[component_type] = []
    
    rf_0_values = np.array(layer_rf['rf_0'])
    component_stats[component_type].extend(rf_0_values[rf_0_values > 0])

#%% Results Summary (Following Paper's Analysis)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Component RF distributions
ax = axes[0,0]
for comp_type, values in component_stats.items():
    if len(values) > 0:
        ax.hist(values, bins=30, alpha=0.6, label=f'{comp_type} (med: {np.median(values):.3f})')
ax.set_xlabel('RF Value')
ax.set_ylabel('Count')
ax.set_title('RF Distribution by Component Type')
ax.legend()

# Component importance (Key insight)
ax = axes[0,1]
comp_names = list(component_stats.keys())
medians = [np.median(component_stats[comp]) if len(component_stats[comp]) > 0 else 0 
           for comp in comp_names]
bars = ax.bar(comp_names, medians)
ax.set_ylabel('Median RF Value')
ax.set_title('Component Importance (Higher = More Important)')
ax.tick_params(axis='x', rotation=45)

# Add values on bars
for bar, val in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
            f'{val:.3f}', ha='center', va='bottom')

# Persistence diagram
ax = axes[1,0]
diagrams = result['persistence']['dgms']
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

# RF by layer depth (BERT-specific)
ax = axes[1,1]
layer_depths = []
layer_medians = []
for layer_name, layer_rf in rf_data.items():
    # Extract layer number for BERT
    if 'encoder.layer.' in layer_name:
        try:
            layer_num = int(layer_name.split('encoder.layer.')[1].split('.')[0])
            rf_vals = np.array(layer_rf['rf_0'])
            if len(rf_vals) > 0:
                layer_depths.append(layer_num)
                layer_medians.append(np.median(rf_vals[rf_vals > 0]))
        except:
            pass

if layer_depths:
    ax.scatter(layer_depths, layer_medians)
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Median RF')
    ax.set_title('RF vs Layer Depth')

plt.tight_layout()
plt.savefig(f'{output_folder}/bert_analysis_summary.png', dpi=150, bbox_inches='tight')
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

print(f"BERT Compression Analysis")
print(f"="*50)
print(f"Total neurons: {total_neurons}")
print(f"Prunable at 30th percentile: {prunable_30} ({prunable_30/total_neurons*100:.1f}%)")
print(f"Prunable at 50th percentile: {prunable_50} ({prunable_50/total_neurons*100:.1f}%)")
print(f"Prunable at 70th percentile: {prunable_70} ({prunable_70/total_neurons*100:.1f}%)")
print(f"RF thresholds - 30th: {p30:.4f}, 50th: {p50:.4f}, 70th: {p70:.4f}")

# Component-wise importance ranking
print(f"\nComponent Importance Ranking (Median RF):")
comp_medians = [(comp, np.median(vals)) for comp, vals in component_stats.items() if len(vals) > 0]
comp_medians.sort(key=lambda x: x[1], reverse=True)
for comp, median in comp_medians:
    print(f"{comp}: {median:.4f}")

#%% Network Topology Summary
betti = result['betti_numbers']
print(f"\nBERT Topological Features:")
print(f"Connected components (β₀): {betti[0]}")
print(f"Loops (β₁): {betti[1]}")
print(f"Voids (β₂): {betti[2]}")
print(f"Total neurons analyzed: {result['total_neurons']}")

#%% Display Topology Visualizations
print(f"\nDisplaying topology visualizations...")

# Core topology plots
plots.plot_distance_matrix(result)
plt.show()

plots.plot_persistence_diagram(result)
plt.show()

plots.plot_tsne_2d(result)
plt.show()

plots.plot_betti_numbers(result)
plt.show()

plots.plot_layer_composition(result)
plt.show()

monitor.remove_hooks()
print(f"\nBERT analysis complete!")