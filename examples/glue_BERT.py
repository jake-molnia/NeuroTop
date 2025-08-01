#%% Setup
import numpy as np
import matplotlib.pyplot as plt
from tinygrad.tensor import Tensor
from tinygrad import nn
import os
from tqdm import tqdm
import time

# Import tinyzoo and ntop (assuming ntop will be modified to work with tinygrad)
from tinyzoo.models.bert import BERTForSequenceClassification
from tinyzoo.data.glue import GLUE
import ntop
from ntop.monitoring import ActivationMonitor
from ntop import analysis, plots

#%% Data Preparation Functions
def create_simple_vocab(texts, max_vocab=5000):
    """Create simple vocabulary from texts"""
    word_freq = {}
    for text in texts:
        for word in text.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}
    
    for word, _ in sorted_words[:max_vocab-4]:
        vocab[word] = len(vocab)
    
    return vocab

def tokenize_texts(texts, vocab, max_length=128):
    """Convert texts to token IDs"""
    tokenized = []
    for text in texts:
        tokens = [vocab['[CLS]']]
        words = text.lower().split()[:max_length-2]
        
        for word in words:
            tokens.append(vocab.get(word, vocab['[UNK]']))
        
        tokens.append(vocab['[SEP]'])
        
        while len(tokens) < max_length:
            tokens.append(vocab['[PAD]'])
        
        tokens = tokens[:max_length]
        tokenized.append(tokens)
    
    return tokenized

class TinygradDataLoader:
    """Simple DataLoader for tinygrad"""
    def __init__(self, tokens, labels, batch_size=16, shuffle=True):
        self.tokens = tokens
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = list(range(len(self.tokens)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_tokens = [self.tokens[idx] for idx in batch_indices]
            batch_labels = [self.labels[idx] for idx in batch_indices]
            
            input_ids = Tensor(batch_tokens)
            attention_mask = (input_ids != 0).float()
            labels_tensor = Tensor([int(label) for label in batch_labels])
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels_tensor
            }
    
    def __len__(self):
        return (len(self.tokens) + self.batch_size - 1) // self.batch_size

#%% Model and Data Setup
def setup_bert_experiment(task='cola', subset_size=2000, max_length=128, batch_size=16):
    """Setup BERT model and data for tinygrad"""
    print(f"Setting up BERT experiment for {task}...")
    
    # Load datasets
    train_dataset = GLUE(root='./data', task=task, split='train')
    dev_dataset = GLUE(root='./data', task=task, split='dev')
    
    # Apply subset for faster experimentation
    if subset_size:
        train_texts = train_dataset.texts[:subset_size]
        train_labels = train_dataset.labels[:subset_size]
        dev_texts = dev_dataset.texts[:subset_size//4]
        dev_labels = dev_dataset.labels[:subset_size//4]
    else:
        train_texts = train_dataset.texts
        train_labels = train_dataset.labels
        dev_texts = dev_dataset.texts
        dev_labels = dev_dataset.labels
    
    # Create vocabulary
    all_texts = train_texts + dev_texts
    vocab = create_simple_vocab(all_texts)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Tokenize
    train_tokens = tokenize_texts(train_texts, vocab, max_length)
    dev_tokens = tokenize_texts(dev_texts, vocab, max_length)
    
    # Create data loaders
    train_loader = TinygradDataLoader(train_tokens, train_labels, batch_size, shuffle=True)
    dev_loader = TinygradDataLoader(dev_tokens, dev_labels, batch_size, shuffle=False)
    
    # Initialize BERT model
    model = BERTForSequenceClassification('base', num_labels=2)
    
    print(f"Model created: BERT-Base")
    print(f"Training batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    
    return model, train_loader, dev_loader, vocab

def get_model_parameters(model):
    """Extract all trainable parameters from tinygrad model"""
    params = []
    
    def find_tensors(obj, prefix=""):
        found = []
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(obj, attr_name)
                if isinstance(attr, Tensor):
                    found.append(attr)
                elif hasattr(attr, '__dict__') and not isinstance(attr, type):
                    found.extend(find_tensors(attr, f"{prefix}.{attr_name}" if prefix else attr_name))
            except:
                continue
        return found
    
    return find_tensors(model)

#%% Training Functions
def train_epoch(model, train_loader, optimizer, epoch_desc="Training"):
    """Train one epoch using tinygrad"""
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=epoch_desc)
    
    with Tensor.train():  # Set training mode
        for batch in pbar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += float(loss.numpy())
            predictions = logits.argmax(axis=-1)
            batch_correct = float((predictions == labels).sum().numpy())
            correct += batch_correct
            total += len(labels)
            
            # Update progress bar
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'Loss': f'{float(loss.numpy()):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    return total_loss / len(train_loader), 100 * correct / total

def evaluate(model, dev_loader, desc="Validation"):
    """Evaluate model"""
    total_loss = 0.0
    correct = 0
    total = 0
    
    # No training mode in tinygrad eval
    for batch in tqdm(dev_loader, desc=desc):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass (no gradients)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Statistics
        total_loss += float(loss.numpy())
        predictions = logits.argmax(axis=-1)
        correct += float((predictions == labels).sum().numpy())
        total += len(labels)
    
    return total_loss / len(dev_loader), 100 * correct / total

#%% Initialize Experiment
print("Initializing BERT experiment...")
model, train_loader, dev_loader, vocab = setup_bert_experiment(
    task='cola', 
    subset_size=1000,
    max_length=64,
    batch_size=16
)

# Extract parameters for optimizer
model_params = get_model_parameters(model)
print(f"Found {len(model_params)} model parameters")

# Initialize optimizer
optimizer = nn.optim.Adam(model_params, lr=2e-5)

# Total parameters count
total_params = sum(param.numel() for param in model_params)
print(f"Total parameters: {total_params:,}")

#%% Configure ntop Monitoring (assumes ntop works with tinygrad)
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
    filter_inactive_neurons=True,
    persistence_threshold=0.01,
    use_quantization=True,
    quantization_resolution=0.1,
    sequence_strategy='cls',
    analyze_full_network=False,
    analyze_by_layers=False,        
    analyze_by_components=True,
)

#%% Initial Analysis
print("\nAnalyzing initial BERT topology...")
initial_state = monitor.analyze(dev_loader, epoch=0, description="INITIAL_STATE", save=True)

#%% Training Loop
print(f"\nStarting BERT training on CoLA task...")

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

# Training curves
ax = axes[0, 0]
epochs_range = range(len(train_losses))
ax.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# Accuracy curves
ax = axes[0, 1]
ax.plot(epochs_range, train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
ax.plot(epochs_range, dev_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# Component importance comparison
ax = axes[1, 0]
if 'by_components' in final_state:
    component_results = final_state['by_components']
    comp_names = list(component_results.keys())
    medians = []
    
    for comp_name in comp_names:
        rf_data = component_results[comp_name]['rf_values']
        all_rf_values = []
        for layer_rf in rf_data.values():
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

# Topology evolution during training
ax = axes[1, 1]
if topology_evolution:
    epochs_tracked = [ep for ep, _, _ in topology_evolution]
    accuracies = [acc for _, _, acc in topology_evolution]
    
    # Get Betti numbers evolution
    betti_0_vals = []
    betti_1_vals = []
    
    for _, state, _ in topology_evolution:
        if 'by_components' in state:
            # Use first component for demo
            first_comp = list(state['by_components'].values())[0]
            betti_0_vals.append(first_comp['betti_numbers'].get(0, 0))
            betti_1_vals.append(first_comp['betti_numbers'].get(1, 0))
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
            rf_0_values = np.array(layer_rf['rf_0'])
            all_rf_values.extend(rf_0_values)
        
        if all_rf_values:
            all_rf_values = np.array(all_rf_values)
            p30 = np.percentile(all_rf_values[all_rf_values > 0], 30)
            p50 = np.percentile(all_rf_values[all_rf_values > 0], 50)
            prunable_30 = sum(1 for val in all_rf_values if val <= p30)
            prunable_50 = sum(1 for val in all_rf_values if val <= p50)
            
            print(f"{comp_name}: 30th percentile prunable: {prunable_30/len(all_rf_values)*100:.1f}%, "
                  f"50th: {prunable_50/len(all_rf_values)*100:.1f}%")

monitor.remove_hooks()
print(f"\nBERT training analysis complete!")
print(f"Results saved to: {output_folder}")
print(f"Final performance: {best_accuracy:.2f}% accuracy on CoLA validation set")

#%%