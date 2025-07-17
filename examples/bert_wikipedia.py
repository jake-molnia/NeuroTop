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

print(f"BERT Hierarchical Analysis")
print(f"Device: {device}")
print(f"Model parameters: {total_params:,}")
print(f"Data: {len(texts)} text samples")

# Tokenize data
encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

#%% Configure Analysis
output_folder = './outputs/bert_hierarchical_analysis'
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
    quantization_resolution=0.1,
    sequence_strategy='cls',
    analyze_full_network=False,
    analyze_by_layers=False,        
    analyze_by_components=True,    
)

#%% Run BERT Analysis
print("\nAnalyzing BERT network topology...")
with torch.no_grad():
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask)
    
    activations = monitor.get_activations()
    result = analysis.analyze(
        activations, 
        max_dim=2, 
        distance_metric='euclidean',
        use_quantization=True, 
        quantization_resolution=5,
        analyze_by_layers=False,
        analyze_by_components=True, 
        analyze_full_network=False
    )

#%% Extract Component Results
component_results = result['by_components']

print(f"\nComponent Analysis Results:")
for comp_name, comp_result in component_results.items():
    rf_data = comp_result['rf_values']
    all_rf_values = []
    for layer_rf in rf_data.values():
        rf_0_values = np.array(layer_rf['rf_0'])
        all_rf_values.extend(rf_0_values[rf_0_values > 0])
    
    print(f"{comp_name}: {comp_result['total_neurons']} neurons, Betti: {comp_result['betti_numbers']}, RF median: {np.median(all_rf_values):.4f}")

#%% Individual Component Plots
print("Generating individual component plots...")

for comp_name, comp_result in component_results.items():
    plots.plot_persistence_diagram(comp_result)
    plt.title(f'Persistence Diagram - {comp_name.title()}')
    plt.savefig(f'{output_folder}/persistence_{comp_name}.png', dpi=150, bbox_inches='tight')
    # plt.show()

for comp_name, comp_result in component_results.items():
    plots.plot_tsne_2d(comp_result)
    plt.title(f't-SNE - {comp_name.title()}')
    plt.savefig(f'{output_folder}/tsne_{comp_name}.png', dpi=150, bbox_inches='tight')
    # plt.show()

for comp_name, comp_result in component_results.items():
    plots.plot_distance_matrix(comp_result)
    plt.title(f'Distance Matrix - {comp_name.title()}')
    plt.savefig(f'{output_folder}/distance_{comp_name}.png', dpi=150, bbox_inches='tight')
    # plt.show()

for comp_name, comp_result in component_results.items():
    plots.plot_betti_numbers(comp_result)
    plt.title(f'Betti Numbers - {comp_name.title()}')
    plt.savefig(f'{output_folder}/betti_{comp_name}.png', dpi=150, bbox_inches='tight')
    # plt.show()

#%% Cross-Component Comparisons
print("Generating cross-component comparisons...")

# RF Distribution Comparison
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple']

for i, (comp_name, comp_result) in enumerate(component_results.items()):
    rf_data = comp_result['rf_values']
    all_rf_values = []
    for layer_rf in rf_data.values():
        rf_0_values = np.array(layer_rf['rf_0'])
        all_rf_values.extend(rf_0_values[rf_0_values > 0])
    
    if all_rf_values:
        ax.hist(all_rf_values, bins=30, alpha=0.6, 
               label=f'{comp_name.title()} (med: {np.median(all_rf_values):.3f})',
               color=colors[i % len(colors)])

ax.set_xlabel('RF Value')
ax.set_ylabel('Count')
ax.set_title('RF Distribution Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(f'{output_folder}/rf_comparison.png', dpi=150, bbox_inches='tight')
# plt.show()

# Betti Number Comparison
fig, ax = plt.subplots(figsize=(12, 6))
comp_names = list(component_results.keys())
betti_0 = [component_results[comp]['betti_numbers'][0] for comp in comp_names]
betti_1 = [component_results[comp]['betti_numbers'][1] for comp in comp_names]
betti_2 = [component_results[comp]['betti_numbers'][2] for comp in comp_names]

x = np.arange(len(comp_names))
width = 0.25

ax.bar(x - width, betti_0, width, label='β₀', alpha=0.8)
ax.bar(x, betti_1, width, label='β₁', alpha=0.8)
ax.bar(x + width, betti_2, width, label='β₂', alpha=0.8)

ax.set_xlabel('Component')
ax.set_ylabel('Betti Number')
ax.set_title('Betti Numbers by Component')
ax.set_xticks(x)
ax.set_xticklabels([comp.title() for comp in comp_names])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.savefig(f'{output_folder}/betti_comparison.png', dpi=150, bbox_inches='tight')
# plt.show()

# Component Size Comparison
fig, ax = plt.subplots(figsize=(10, 6))
neuron_counts = [component_results[comp]['total_neurons'] for comp in comp_names]
bars = ax.bar(comp_names, neuron_counts, alpha=0.8)

ax.set_xlabel('Component')
ax.set_ylabel('Number of Neurons')
ax.set_title('Neuron Count by Component')
ax.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, neuron_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(neuron_counts)*0.01,
            f'{count:,}', ha='center', va='bottom')

plt.xticks(rotation=45)
plt.savefig(f'{output_folder}/size_comparison.png', dpi=150, bbox_inches='tight')
# plt.show()

# RF Median Comparison
fig, ax = plt.subplots(figsize=(10, 6))
medians = []
for comp_name in comp_names:
    rf_data = component_results[comp_name]['rf_values']
    all_rf_values = []
    for layer_rf in rf_data.values():
        rf_0_values = np.array(layer_rf['rf_0'])
        all_rf_values.extend(rf_0_values[rf_0_values > 0])
    medians.append(np.median(all_rf_values) if all_rf_values else 0.0)

bars = ax.bar(comp_names, medians, alpha=0.8)
ax.set_xlabel('Component')
ax.set_ylabel('Median RF Value')
ax.set_title('Component Importance (Median RF)')
ax.grid(True, alpha=0.3, axis='y')

for bar, median in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(medians)*0.01,
            f'{median:.3f}', ha='center', va='bottom')

plt.xticks(rotation=45)
plt.savefig(f'{output_folder}/rf_medians.png', dpi=150, bbox_inches='tight')
# plt.show()

#%% Attention Sub-Components
if 'attention' in component_results:
    print("Analyzing attention sub-components...")
    attention_rf = component_results['attention']['rf_values']
    
    sub_components = {'query': [], 'key': [], 'value': [], 'attention_output': []}
    
    for layer_name, layer_rf in attention_rf.items():
        rf_0_values = np.array(layer_rf['rf_0'])
        valid_rf = rf_0_values[rf_0_values > 0]
        
        if 'query' in layer_name.lower():
            sub_components['query'].extend(valid_rf)
        elif 'key' in layer_name.lower():
            sub_components['key'].extend(valid_rf)
        elif 'value' in layer_name.lower():
            sub_components['value'].extend(valid_rf)
        elif 'attention.output' in layer_name.lower():
            sub_components['attention_output'].extend(valid_rf)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (sub_comp, rf_values) in enumerate(sub_components.items()):
        if rf_values:
            ax.hist(rf_values, bins=30, alpha=0.6,
                   label=f'{sub_comp.title()} (med: {np.median(rf_values):.3f})',
                   color=colors[i])
    
    ax.set_xlabel('RF Value')
    ax.set_ylabel('Count')
    ax.set_title('Attention Sub-Component RF Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{output_folder}/attention_subcomponents.png', dpi=150, bbox_inches='tight')
    # plt.show()

#%% Feedforward Sub-Components
if 'feedforward' in component_results:
    print("Analyzing feedforward sub-components...")
    ff_rf = component_results['feedforward']['rf_values']
    
    sub_components = {'intermediate': [], 'output': []}
    
    for layer_name, layer_rf in ff_rf.items():
        rf_0_values = np.array(layer_rf['rf_0'])
        valid_rf = rf_0_values[rf_0_values > 0]
        
        if 'intermediate' in layer_name.lower():
            sub_components['intermediate'].extend(valid_rf)
        elif 'output' in layer_name.lower() and 'attention' not in layer_name.lower():
            sub_components['output'].extend(valid_rf)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green', 'orange']
    
    for i, (sub_comp, rf_values) in enumerate(sub_components.items()):
        if rf_values:
            ax.hist(rf_values, bins=30, alpha=0.6,
                   label=f'{sub_comp.title()} (med: {np.median(rf_values):.3f})',
                   color=colors[i])
    
    ax.set_xlabel('RF Value')
    ax.set_ylabel('Count')
    ax.set_title('Feedforward Sub-Component RF Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{output_folder}/feedforward_subcomponents.png', dpi=150, bbox_inches='tight')
    # plt.show()

#%% RF Heatmap Analysis
print("Generating RF heatmap visualizations...")

# Create fake topology_states for each component to use existing plot functions
for comp_name, comp_result in component_results.items():
    print(f"RF heatmap for {comp_name}...")
    
    # Create single-state topology data for heatmap functions
    fake_topology_states = [{
        'epoch': 0,
        'rf_values': comp_result['rf_values']
    }]
    
    # RF distribution evolution (static for single snapshot)
    plots.plot_rf_distribution_evolution(fake_topology_states, 'rf_0')
    plt.suptitle(f'RF Distribution - {comp_name.title()}')
    plt.savefig(f'{output_folder}/rf_distribution_{comp_name}.png', dpi=150, bbox_inches='tight')
    # plt.show()
    
    # RF box evolution (static for single snapshot)  
    plots.plot_rf_box_evolution(fake_topology_states, 'rf_0')
    plt.suptitle(f'RF Statistics - {comp_name.title()}')
    plt.savefig(f'{output_folder}/rf_box_{comp_name}.png', dpi=150, bbox_inches='tight')
    # plt.show()

# Network-wide RF comparison across components
print("Generating network-wide RF analysis...")

# Combine all component RF data for network view
all_components_rf = {}
for comp_name, comp_result in component_results.items():
    for layer_name, layer_rf in comp_result['rf_values'].items():
        # Add component prefix to layer names to distinguish
        prefixed_name = f"{comp_name}_{layer_name}"
        all_components_rf[prefixed_name] = layer_rf

# Create network-wide fake topology state
network_topology_state = [{
    'epoch': 0, 
    'rf_values': all_components_rf
}]

plots.plot_rf_distribution_evolution_network(network_topology_state, 'rf_0')
plt.title('Network-wide RF Distribution (All Components)')
plt.savefig(f'{output_folder}/rf_network_distribution.png', dpi=150, bbox_inches='tight')
# plt.show()

plots.plot_rf_box_evolution_network(network_topology_state, 'rf_0')
plt.title('Network-wide RF Statistics (All Components)')
plt.savefig(f'{output_folder}/rf_network_box.png', dpi=150, bbox_inches='tight')
# plt.show()

#%% Compression Analysis
print(f"\nBERT Compression Analysis")
print(f"="*50)

total_neurons = sum(comp_result['total_neurons'] for comp_result in component_results.values())
print(f"Total neurons analyzed: {total_neurons:,}")

for comp_name, comp_result in component_results.items():
    rf_data = comp_result['rf_values']
    all_rf_values = []
    for layer_rf in rf_data.values():
        rf_0_values = np.array(layer_rf['rf_0'])
        all_rf_values.extend(rf_0_values)
    
    if all_rf_values:
        all_rf_values = np.array(all_rf_values)
        p30 = np.percentile(all_rf_values[all_rf_values > 0], 30)
        p50 = np.percentile(all_rf_values[all_rf_values > 0], 50)
        prunable_30 = sum(1 for val in all_rf_values if val <= p30)
        prunable_50 = sum(1 for val in all_rf_values if val <= p50)
        
        print(f"{comp_name}: 30th percentile prunable: {prunable_30/len(all_rf_values)*100:.1f}%, 50th: {prunable_50/len(all_rf_values)*100:.1f}%")

monitor.remove_hooks()
print(f"\nBERT analysis complete!")