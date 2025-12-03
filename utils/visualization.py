import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import os

def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        history: dict with keys 'train_loss', 'train_ap', 'val_loss', 'val_ap'
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # AP curves
    axes[1].plot(history['train_ap'], label='Train AP', linewidth=2)
    axes[1].plot(history['val_ap'], label='Val AP', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Average Precision', fontsize=12)
    axes[1].set_title('Training and Validation AP', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def visualize_attention_matrix(attention_weights, edge_index, num_nodes, node_mask=None, 
                                save_path=None, title="Attention Matrix", head_idx=0):
    """
    Visualize attention weights as a matrix
    
    Args:
        attention_weights: [num_edges, num_heads] attention weights
        edge_index: [2, num_edges] edge indices
        num_nodes: number of nodes in graph
        node_mask: optional mask to show subset of nodes
        save_path: path to save figure
        title: plot title
        head_idx: which attention head to visualize
    """
    # Create adjacency matrix with attention weights
    att_matrix = torch.zeros((num_nodes, num_nodes))
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        att_matrix[dst, src] = attention_weights[i, head_idx].item()
    
    att_matrix = att_matrix.cpu().numpy()
    
    # Apply node mask if provided
    if node_mask is not None:
        att_matrix = att_matrix[node_mask][:, node_mask]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(att_matrix, cmap='YlOrRd', square=True, 
                cbar_kws={'label': 'Attention Weight'})
    plt.title(f'{title} - Head {head_idx}', fontsize=14, fontweight='bold')
    plt.xlabel('Source Node', fontsize=12)
    plt.ylabel('Target Node', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention matrix saved to {save_path}")
    
    plt.show()


def plot_attention_distance_distribution(attention_weights, edge_index, num_heads, 
                                         save_path=None):
    """
    Plot distribution of attention weights by edge distance
    
    Args:
        attention_weights: [num_edges, num_heads]
        edge_index: [2, num_edges]
        num_heads: number of attention heads
    """
    # Compute edge distances (graph distance would require BFS, using index distance as proxy)
    distances = torch.abs(edge_index[0] - edge_index[1]).cpu().numpy()
    attention_weights = attention_weights.cpu().numpy()
    
    fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for head in range(num_heads):
        ax = axes[head]
        
        # Create dataframe for easier plotting
        df = pd.DataFrame({
            'distance': distances,
            'attention': attention_weights[:, head]
        })
        
        # Bin by distance
        df['distance_bin'] = pd.cut(df['distance'], bins=10)
        grouped = df.groupby('distance_bin')['attention'].mean()
        
        grouped.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'Head {head}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Edge Distance', fontsize=9)
        ax.set_ylabel('Mean Attention', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention distance distribution saved to {save_path}")
    
    plt.show()


def plot_attention_entropy(attention_weights, num_heads, save_path=None):
    """
    Plot entropy of attention distribution for each head
    
    Lower entropy = more focused attention
    Higher entropy = more distributed attention
    """
    # Compute entropy for each edge and head
    att_probs = attention_weights.cpu().numpy()
    entropy = -np.sum(att_probs * np.log(att_probs + 1e-9), axis=0)  # [num_heads]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_heads), entropy, color='steelblue', alpha=0.7)
    plt.xlabel('Attention Head', fontsize=12)
    plt.ylabel('Average Entropy', fontsize=12)
    plt.title('Attention Entropy by Head', fontsize=14, fontweight='bold')
    plt.xticks(range(num_heads))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention entropy plot saved to {save_path}")
    
    plt.show()


def visualize_embeddings(embeddings, labels, method='tsne', save_path=None, title="Embedding Visualization"):
    """
    Visualize node/graph embeddings in 2D
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] labels for coloring
        method: 'tsne' or 'pca'
    """
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:  # pca
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.title(f'{title} ({method.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding visualization saved to {save_path}")
    
    plt.show()


def plot_layer_wise_performance(layer_performances, save_path=None):
    """
    Plot how performance changes across layers
    
    Args:
        layer_performances: dict {layer_idx: ap_score}
    """
    layers = sorted(layer_performances.keys())
    aps = [layer_performances[l] for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, aps, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Average Precision', fontsize=12)
    plt.title('Layer-wise Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layer-wise performance plot saved to {save_path}")
    
    plt.show()


def plot_per_class_performance(class_names, ap_scores, save_path=None):
    """
    Plot per-class average precision scores
    
    Args:
        class_names: list of class names
        ap_scores: list of AP scores per class
    """
    # Sort by AP score
    sorted_indices = np.argsort(ap_scores)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_aps = [ap_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(sorted_classes)), sorted_aps, color='steelblue', alpha=0.7)
    
    # Color code by performance
    for i, bar in enumerate(bars):
        if sorted_aps[i] > 0.7:
            bar.set_color('green')
        elif sorted_aps[i] < 0.5:
            bar.set_color('red')
    
    plt.yticks(range(len(sorted_classes)), sorted_classes)
    plt.xlabel('Average Precision', fontsize=12)
    plt.title('Per-Class Performance', fontsize=14, fontweight='bold')
    plt.axvline(x=np.mean(ap_scores), color='black', linestyle='--', 
                linewidth=2, label=f'Mean AP: {np.mean(ap_scores):.3f}')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class performance plot saved to {save_path}")
    
    plt.show()


def plot_confusion_style_heatmap(y_true, y_pred, class_names, save_path=None):
    """
    Plot co-occurrence matrix for multi-label predictions
    Shows which classes are predicted together
    """
    # Binarize predictions
    y_pred_binary = (y_pred > 0.5).float()
    
    # Compute co-occurrence matrix
    co_occurrence = torch.matmul(y_pred_binary.T, y_pred_binary)
    co_occurrence = co_occurrence.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_occurrence, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={'label': 'Co-occurrence Count'})
    plt.title('Class Co-occurrence in Predictions', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Co-occurrence heatmap saved to {save_path}")
    
    plt.show()