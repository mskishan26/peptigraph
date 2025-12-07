"""
Enhanced Attention Analysis for Graph Transformers on Peptides-func
====================================================================

This script provides comprehensive analysis to answer:
"Why should we move from pure graph transformers to hybrid architectures like GraphViT?"

Key analyses:
1. Attention by graph distance (not index distance!) - THE KEY METRIC
2. Layer-wise attention pattern evolution
3. Head specialization analysis
4. Fixed t-SNE visualization for multi-label data
5. Correct entropy computation
6. Connection between attention patterns and prediction quality

Author: Enhanced for Kishan's research project
"""

import torch
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Import your existing modules
from utils.data_loader import get_peptides_loaders
from utils.metrics import evaluate_model
from models.transformer import GraphTransformer


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_shortest_path_distances(edge_index, num_nodes):
    """
    Compute actual shortest path distances between all node pairs.
    This is graph distance, not index distance!
    
    Returns:
        dist_matrix: [num_nodes, num_nodes] shortest path distances
                    -1 indicates unreachable pairs
    """
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(len(row))
    
    # Build sparse adjacency matrix
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # Compute shortest paths (BFS for unweighted)
    dist_matrix = shortest_path(adj, directed=False, unweighted=True)
    dist_matrix[dist_matrix == np.inf] = -1
    
    return dist_matrix


def get_graph_from_batch(data, graph_idx):
    """Extract a single graph from a batched PyG data object."""
    node_mask = data.batch == graph_idx
    num_nodes = node_mask.sum().item()
    node_indices = torch.where(node_mask)[0]
    node_start = node_indices[0].item()
    
    # Get edges for this graph
    edge_mask = (data.edge_index[0] >= node_start) & \
                (data.edge_index[0] < node_start + num_nodes) & \
                (data.edge_index[1] >= node_start) & \
                (data.edge_index[1] < node_start + num_nodes)
    
    graph_edges = data.edge_index[:, edge_mask] - node_start
    
    return num_nodes, node_start, graph_edges, edge_mask


# =============================================================================
# CORE ANALYSIS: ATTENTION BY GRAPH DISTANCE
# =============================================================================

def analyze_attention_by_graph_distance(model, loader, device, output_dir, num_samples=100):
    """
    THE KEY ANALYSIS: What fraction of attention goes to each graph distance?
    
    This answers: "Is the transformer actually using long-range attention,
    or is it essentially behaving like a local message-passing GNN?"
    """
    print("\n" + "="*70)
    print("CRITICAL ANALYSIS: Attention Distribution by Graph Distance")
    print("="*70)
    
    model.eval()
    
    # Store attention weights by (layer, distance)
    # distance_attention[layer][distance] = list of attention weights
    distance_attention = {layer: defaultdict(list) 
                         for layer in range(model.num_layers)}
    
    # Also track per-head patterns
    head_distance_attention = {layer: {head: defaultdict(list) 
                                       for head in range(model.num_heads)}
                              for layer in range(model.num_layers)}
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if sample_count >= num_samples:
                break
            
            data = data.to(device)
            
            # Forward pass with attention
            _ = model(data, return_attention=True)
            attention_weights = model.get_attention_weights()
            
            # Process each graph in batch
            for graph_idx in range(data.num_graphs):
                if sample_count >= num_samples:
                    break
                
                num_nodes, node_start, graph_edges, _ = get_graph_from_batch(data, graph_idx)
                
                # Compute shortest path distances for this graph
                if num_nodes > 1:
                    dist_matrix = compute_shortest_path_distances(graph_edges, num_nodes)
                else:
                    continue
                
                # Analyze attention at each layer
                for layer_idx, (edge_index_att, att_weights) in enumerate(attention_weights):
                    # Get attention edges for this graph
                    att_mask = (edge_index_att[0] >= node_start) & \
                              (edge_index_att[0] < node_start + num_nodes) & \
                              (edge_index_att[1] >= node_start) & \
                              (edge_index_att[1] < node_start + num_nodes)
                    
                    layer_edges = edge_index_att[:, att_mask] - node_start
                    layer_att = att_weights[att_mask]  # [num_edges, num_heads]
                    
                    # For each edge with attention, get its graph distance
                    for e in range(layer_edges.shape[1]):
                        src = layer_edges[0, e].item()
                        tgt = layer_edges[1, e].item()
                        
                        dist = int(dist_matrix[src, tgt])
                        
                        if dist >= 0:  # Valid distance
                            # Average across heads for overall analysis
                            avg_att = layer_att[e].mean().item()
                            distance_attention[layer_idx][dist].append(avg_att)
                            
                            # Per-head analysis
                            for h in range(model.num_heads):
                                head_distance_attention[layer_idx][h][dist].append(
                                    layer_att[e, h].item()
                                )
                
                sample_count += 1
                
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {sample_count}/{num_samples} samples...")
    
    print(f"\nAnalyzed {sample_count} graphs")
    
    # ==========================================================================
    # Compute and visualize results
    # ==========================================================================
    
    summary = {}
    
    # Create figure for layer comparison
    fig, axes = plt.subplots(1, model.num_layers, figsize=(5*model.num_layers, 4))
    if model.num_layers == 1:
        axes = [axes]
    
    print("\n" + "-"*70)
    print("ATTENTION DISTRIBUTION BY GRAPH DISTANCE (per layer)")
    print("-"*70)
    
    for layer_idx in range(model.num_layers):
        ax = axes[layer_idx]
        
        # Get distances and compute statistics
        distances = sorted(distance_attention[layer_idx].keys())
        
        if not distances:
            print(f"  Layer {layer_idx}: No attention data")
            continue
        
        # Compute total attention mass at each distance
        total_attention = []
        mean_attention = []
        
        for d in distances:
            weights = distance_attention[layer_idx][d]
            total_attention.append(np.sum(weights))
            mean_attention.append(np.mean(weights))
        
        # Normalize to get percentage distribution
        total_attention = np.array(total_attention)
        if total_attention.sum() > 0:
            attention_pct = total_attention / total_attention.sum() * 100
        else:
            attention_pct = np.zeros_like(total_attention)
        
        # Plot
        colors = ['#d62728' if d <= 1 else '#2ca02c' if d <= 2 else '#1f77b4' 
                  for d in distances]
        ax.bar(distances[:8], attention_pct[:8], color=colors[:8], alpha=0.7, 
               edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Graph Distance (hops)', fontsize=11)
        ax.set_ylabel('% of Total Attention', fontsize=11)
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.set_xticks(distances[:8])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Compute summary statistics
        one_hop = attention_pct[distances.index(1)] if 1 in distances else 0
        two_hop = attention_pct[distances.index(2)] if 2 in distances else 0
        local = sum(attention_pct[i] for i, d in enumerate(distances) if d <= 2)
        long_range = 100 - local
        
        summary[layer_idx] = {
            '1-hop (%)': round(one_hop, 2),
            '2-hop (%)': round(two_hop, 2),
            'local_<=2 (%)': round(local, 2),
            'long_range_>2 (%)': round(long_range, 2)
        }
        
        print(f"\n  Layer {layer_idx}:")
        print(f"    1-hop attention:     {one_hop:6.2f}%")
        print(f"    2-hop attention:     {two_hop:6.2f}%")
        print(f"    Local (≤2 hops):     {local:6.2f}%")
        print(f"    Long-range (>2):     {long_range:6.2f}%")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.7, label='1-hop (immediate neighbor)'),
        Patch(facecolor='#2ca02c', alpha=0.7, label='2-hop'),
        Patch(facecolor='#1f77b4', alpha=0.7, label='3+ hops (long-range)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
               bbox_to_anchor=(0.5, 1.08), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attention_by_graph_distance.png', dpi=150, 
                bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved: {output_dir}/attention_by_graph_distance.png")
    
    # Save summary to JSON
    with open(f'{output_dir}/attention_distance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary, head_distance_attention


# =============================================================================
# HEAD SPECIALIZATION ANALYSIS
# =============================================================================

def analyze_head_specialization(head_distance_attention, num_layers, num_heads, output_dir):
    """
    Do different attention heads specialize for different distance ranges?
    
    This helps understand if some heads focus locally while others attend globally.
    """
    print("\n" + "="*70)
    print("HEAD SPECIALIZATION ANALYSIS")
    print("="*70)
    
    # Analyze last layer (most informative for final predictions)
    last_layer = num_layers - 1
    
    fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(4 * ((num_heads + 1) // 2), 8))
    axes = axes.flatten()
    
    head_profiles = {}
    
    for head in range(num_heads):
        ax = axes[head]
        
        head_data = head_distance_attention[last_layer][head]
        distances = sorted(head_data.keys())[:6]  # Focus on distances 0-5
        
        if not distances:
            continue
        
        # Compute attention distribution for this head
        total_att = [np.sum(head_data[d]) for d in distances]
        total_sum = sum(total_att)
        
        if total_sum > 0:
            pct_att = np.array(total_att) / total_sum * 100
        else:
            pct_att = np.zeros(len(distances))
        
        # Color by distance
        colors = ['#d62728' if d <= 1 else '#2ca02c' if d == 2 else '#1f77b4' 
                  for d in distances]
        
        ax.bar(distances, pct_att, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Distance', fontsize=10)
        ax.set_ylabel('% Attention', fontsize=10)
        ax.set_title(f'Head {head}', fontsize=11, fontweight='bold')
        ax.set_xticks(distances)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Classify head type
        local_focus = sum(pct_att[i] for i, d in enumerate(distances) if d <= 1)
        mid_focus = pct_att[distances.index(2)] if 2 in distances else 0
        
        if local_focus > 80:
            head_type = "LOCAL"
            color = '#d62728'
        elif local_focus > 60:
            head_type = "MOSTLY LOCAL"
            color = '#ff7f0e'
        elif local_focus < 40:
            head_type = "DISTRIBUTED"
            color = '#1f77b4'
        else:
            head_type = "BALANCED"
            color = '#2ca02c'
        
        head_profiles[head] = {
            'type': head_type,
            'local_pct': round(local_focus, 1),
            'peak_distance': distances[np.argmax(pct_att)]
        }
        
        # Add type annotation
        ax.text(0.95, 0.95, head_type, transform=ax.transAxes, fontsize=9,
                fontweight='bold', ha='right', va='top', color=color)
    
    # Hide unused subplots
    for i in range(num_heads, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Attention Head Specialization (Layer {last_layer})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/head_specialization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\nLayer {last_layer} Head Analysis:")
    local_heads = sum(1 for h in head_profiles.values() if 'LOCAL' in h['type'])
    print(f"  Local heads (>60% to 1-hop):     {local_heads}/{num_heads}")
    print(f"  Distributed heads (<40% to 1-hop): {num_heads - local_heads}/{num_heads}")
    
    print("\n  Per-head breakdown:")
    for head, profile in head_profiles.items():
        print(f"    Head {head}: {profile['type']:12s} | {profile['local_pct']:5.1f}% local | peak at {profile['peak_distance']}-hop")
    
    print(f"\n  Saved: {output_dir}/head_specialization.png")
    
    return head_profiles


# =============================================================================
# FIXED ENTROPY COMPUTATION
# =============================================================================

def analyze_attention_entropy_fixed(model, loader, device, output_dir, num_samples=50):
    """
    Correct entropy analysis: compute entropy of each node's attention distribution,
    then average across nodes.
    
    Low entropy = attention focused on few neighbors (sharp)
    High entropy = attention spread across many neighbors (diffuse)
    """
    print("\n" + "="*70)
    print("ATTENTION ENTROPY ANALYSIS (Corrected)")
    print("="*70)
    
    model.eval()
    
    # Store per-node entropy for each layer and head
    layer_head_entropy = {layer: {head: [] for head in range(model.num_heads)}
                         for layer in range(model.num_layers)}
    
    sample_count = 0
    
    with torch.no_grad():
        for data in loader:
            if sample_count >= num_samples:
                break
            
            data = data.to(device)
            _ = model(data, return_attention=True)
            attention_weights = model.get_attention_weights()
            
            for graph_idx in range(min(data.num_graphs, num_samples - sample_count)):
                num_nodes, node_start, _, _ = get_graph_from_batch(data, graph_idx)
                
                for layer_idx, (edge_index_att, att_weights) in enumerate(attention_weights):
                    # Get attention for this graph
                    att_mask = (edge_index_att[1] >= node_start) & \
                              (edge_index_att[1] < node_start + num_nodes)
                    
                    layer_edges = edge_index_att[:, att_mask]
                    layer_att = att_weights[att_mask]  # [num_edges, num_heads]
                    
                    # Compute entropy per target node per head
                    for head in range(model.num_heads):
                        for node in range(node_start, node_start + num_nodes):
                            # Get attention weights for edges pointing to this node
                            node_mask = layer_edges[1] == node
                            if node_mask.sum() > 1:  # Need >1 neighbor for meaningful entropy
                                node_att = layer_att[node_mask, head]
                                # Normalize to get distribution
                                node_att = node_att / (node_att.sum() + 1e-9)
                                # Compute entropy
                                entropy = -torch.sum(node_att * torch.log(node_att + 1e-9)).item()
                                layer_head_entropy[layer_idx][head].append(entropy)
                
                sample_count += 1
    
    # Visualize
    fig, axes = plt.subplots(1, model.num_layers, figsize=(5*model.num_layers, 4))
    if model.num_layers == 1:
        axes = [axes]
    
    print("\nMean Attention Entropy by Layer and Head:")
    print("(Higher = more distributed attention, Lower = more focused)")
    
    for layer_idx in range(model.num_layers):
        ax = axes[layer_idx]
        
        means = []
        stds = []
        
        for head in range(model.num_heads):
            entropies = layer_head_entropy[layer_idx][head]
            if entropies:
                means.append(np.mean(entropies))
                stds.append(np.std(entropies))
            else:
                means.append(0)
                stds.append(0)
        
        x = np.arange(model.num_heads)
        bars = ax.bar(x, means, yerr=stds, capsize=3, color='steelblue', 
                      alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Attention Head', fontsize=11)
        ax.set_ylabel('Mean Entropy', fontsize=11)
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, axis='y')
        
        print(f"\n  Layer {layer_idx}:")
        for head in range(model.num_heads):
            print(f"    Head {head}: {means[head]:.3f} ± {stds[head]:.3f}")
    
    plt.suptitle('Attention Entropy by Head (per-node, then averaged)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attention_entropy_corrected.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved: {output_dir}/attention_entropy_corrected.png")


# =============================================================================
# FIXED t-SNE VISUALIZATION FOR MULTI-LABEL DATA
# =============================================================================

def visualize_embeddings_fixed(model, loader, device, output_dir, num_samples=500):
    """
    Fixed embedding visualization for multi-label classification.
    
    Options:
    1. Use dominant class (argmax of labels)
    2. Show separate plots per class
    3. Use prediction confidence
    """
    print("\n" + "="*70)
    print("EMBEDDING VISUALIZATION (Fixed for Multi-label)")
    print("="*70)
    
    model.eval()
    
    # Collect graph-level embeddings
    all_embeddings = []
    all_labels = []
    all_preds = []
    
    sample_count = 0
    
    with torch.no_grad():
        for data in loader:
            if sample_count >= num_samples:
                break
            
            data = data.to(device)
            
            # Get embeddings before final prediction layer
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            
            if model.pe is not None:
                pe = model.pe(edge_index, batch, x.shape[0])
                x = torch.cat([x, pe], dim=-1)
            
            x = model.node_encoder(x)
            x = torch.relu(x)
            
            edge_attr_encoded = None
            if model.edge_encoder is not None and edge_attr is not None:
                edge_attr_encoded = model.edge_encoder(edge_attr)
            
            for transformer, norm in zip(model.transformer_layers, model.layer_norms):
                x_residual = x
                x = transformer(x, edge_index, edge_attr_encoded)
                x = norm(x + x_residual)
                x = model.dropout(x)
            
            # Global pooling to get graph embeddings
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(x, batch)
            
            all_embeddings.append(graph_emb.cpu())
            all_labels.append(data.y.cpu())
            
            # Get predictions
            out = model.graph_pred_linear(graph_emb)
            all_preds.append(torch.sigmoid(out).cpu())
            
            sample_count += data.num_graphs
    
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    
    print(f"  Collected {len(all_embeddings)} graph embeddings")
    
    # Run t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # CLASS NAMES
    class_names = [
        'Antifungal', 'Cell-cell comm.', 'Anticancer', 'Drug delivery',
        'Antimicrobial', 'Antiviral', 'Antihypertensive', 'Antibacterial',
        'Antiparasitic', 'Toxic'
    ]
    
    # =========================================================================
    # Plot 1: Color by dominant class (most confident prediction)
    # =========================================================================
    dominant_class = np.argmax(all_preds, axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=dominant_class, cmap='tab10', alpha=0.6, s=30)
    
    # Add colorbar with class names
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.ax.set_yticklabels(class_names)
    cbar.set_label('Dominant Predicted Class', fontsize=11)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('Graph Embeddings Colored by Dominant Class', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embeddings_tsne_dominant_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/embeddings_tsne_dominant_class.png")
    
    # =========================================================================
    # Plot 2: Separate subplot for each class (presence/absence)
    # =========================================================================
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for class_idx in range(10):
        ax = axes[class_idx]
        
        # Binary labels for this class (threshold at 0.5)
        class_labels = (all_labels[:, class_idx] > 0.5).astype(int)
        
        colors = ['#1f77b4' if l == 0 else '#d62728' for l in class_labels]
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   c=colors, alpha=0.4, s=20)
        
        ax.set_title(f'{class_names[class_idx]}', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Count positives
        n_pos = class_labels.sum()
        ax.text(0.02, 0.98, f'n={n_pos}', transform=ax.transAxes,
                fontsize=9, va='top', color='#d62728')
    
    plt.suptitle('Graph Embeddings by Class (Red = Positive)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embeddings_tsne_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/embeddings_tsne_per_class.png")


# =============================================================================
# PREDICTION QUALITY VS ATTENTION PATTERNS
# =============================================================================

def analyze_attention_vs_performance(model, loader, device, output_dir, num_samples=200):
    """
    Compare attention patterns between correctly and incorrectly classified samples.
    
    Hypothesis: If local attention is sufficient, correct/incorrect samples
    might not differ. If long-range is needed but missing, incorrect samples
    might be those requiring distant dependencies.
    """
    print("\n" + "="*70)
    print("ATTENTION PATTERNS: Correct vs Incorrect Predictions")
    print("="*70)
    
    model.eval()
    
    correct_local_pct = []
    incorrect_local_pct = []
    correct_entropy = []
    incorrect_entropy = []
    
    sample_count = 0
    
    with torch.no_grad():
        for data in loader:
            if sample_count >= num_samples:
                break
            
            data = data.to(device)
            
            # Get predictions
            output = model(data, return_attention=True)
            preds = torch.sigmoid(output)
            attention_weights = model.get_attention_weights()
            
            # Use mean AP across all classes for correctness
            # Simplified: use first class for analysis
            labels = data.y[:, 0]
            pred_labels = (preds[:, 0] > 0.5).float()
            correct_mask = (pred_labels == (labels > 0.5).float())
            
            # Analyze last layer
            edge_index_att, att_weights = attention_weights[-1]
            
            for graph_idx in range(data.num_graphs):
                if sample_count >= num_samples:
                    break
                
                num_nodes, node_start, graph_edges, _ = get_graph_from_batch(data, graph_idx)
                
                if num_nodes <= 1:
                    continue
                
                # Compute graph distances
                dist_matrix = compute_shortest_path_distances(graph_edges, num_nodes)
                
                # Get attention for this graph
                att_mask = (edge_index_att[0] >= node_start) & \
                          (edge_index_att[0] < node_start + num_nodes)
                layer_edges = edge_index_att[:, att_mask] - node_start
                layer_att = att_weights[att_mask].mean(dim=1)  # Average over heads
                
                # Compute local attention percentage
                local_att = 0
                total_att = 0
                entropies = []
                
                for e in range(layer_edges.shape[1]):
                    src, tgt = layer_edges[0, e].item(), layer_edges[1, e].item()
                    dist = int(dist_matrix[src, tgt])
                    att = layer_att[e].item()
                    
                    if dist >= 0:
                        total_att += att
                        if dist <= 1:
                            local_att += att
                
                local_pct = (local_att / (total_att + 1e-9)) * 100
                
                # Simple entropy proxy
                att_normalized = layer_att / (layer_att.sum() + 1e-9)
                entropy = -torch.sum(att_normalized * torch.log(att_normalized + 1e-9)).item()
                
                if correct_mask[graph_idx]:
                    correct_local_pct.append(local_pct)
                    correct_entropy.append(entropy)
                else:
                    incorrect_local_pct.append(local_pct)
                    incorrect_entropy.append(entropy)
                
                sample_count += 1
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Local attention percentage
    ax = axes[0]
    data_to_plot = [correct_local_pct, incorrect_local_pct]
    bp = ax.boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ca02c')
    bp['boxes'][1].set_facecolor('#d62728')
    ax.set_ylabel('Local Attention (%)', fontsize=12)
    ax.set_title('Local (≤1-hop) Attention by Prediction Quality', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add means
    ax.scatter([1, 2], [np.mean(correct_local_pct), np.mean(incorrect_local_pct)],
               marker='D', color='black', s=50, zorder=5, label='Mean')
    ax.legend()
    
    # Entropy
    ax = axes[1]
    data_to_plot = [correct_entropy, incorrect_entropy]
    bp = ax.boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ca02c')
    bp['boxes'][1].set_facecolor('#d62728')
    ax.set_ylabel('Attention Entropy', fontsize=12)
    ax.set_title('Attention Entropy by Prediction Quality', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.scatter([1, 2], [np.mean(correct_entropy), np.mean(incorrect_entropy)],
               marker='D', color='black', s=50, zorder=5, label='Mean')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attention_correct_vs_incorrect.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nCorrect predictions (n={len(correct_local_pct)}):")
    print(f"  Mean local attention: {np.mean(correct_local_pct):.1f}%")
    print(f"  Mean entropy: {np.mean(correct_entropy):.3f}")
    
    print(f"\nIncorrect predictions (n={len(incorrect_local_pct)}):")
    print(f"  Mean local attention: {np.mean(incorrect_local_pct):.1f}%")
    print(f"  Mean entropy: {np.mean(incorrect_entropy):.3f}")
    
    print(f"\n  Saved: {output_dir}/attention_correct_vs_incorrect.png")


# =============================================================================
# GENERATE SUMMARY REPORT
# =============================================================================

def generate_summary_report(summary, head_profiles, output_dir):
    """Generate a text summary of findings."""
    
    report = []
    report.append("="*70)
    report.append("GRAPH TRANSFORMER ATTENTION ANALYSIS - SUMMARY REPORT")
    report.append("="*70)
    report.append("")
    
    # Key finding: local attention
    last_layer = max(summary.keys())
    local_pct = summary[last_layer]['local_<=2 (%)']
    one_hop_pct = summary[last_layer]['1-hop (%)']
    
    report.append("KEY FINDING: Attention Locality")
    report.append("-"*40)
    
    if one_hop_pct > 70:
        report.append(f"⚠️  CRITICAL: {one_hop_pct:.1f}% of attention goes to immediate (1-hop) neighbors!")
        report.append(f"   The transformer is essentially behaving as a local message-passing GNN.")
        report.append("")
        report.append("   IMPLICATION: The model is NOT leveraging long-range dependencies")
        report.append("   despite having the architectural capacity to do so.")
        report.append("")
        report.append("   RECOMMENDATION: Consider hybrid architectures (e.g., GraphViT) that")
        report.append("   explicitly separate local message passing from global attention.")
    elif local_pct > 60:
        report.append(f"⚡ Attention is mostly local ({local_pct:.1f}% within 2 hops)")
        report.append(f"   Some long-range attention exists but is limited.")
    else:
        report.append(f"✓ Good balance: {local_pct:.1f}% local, {100-local_pct:.1f}% long-range")
    
    report.append("")
    report.append("")
    report.append("LAYER-BY-LAYER BREAKDOWN")
    report.append("-"*40)
    
    for layer_idx in sorted(summary.keys()):
        stats = summary[layer_idx]
        report.append(f"Layer {layer_idx}:")
        report.append(f"  1-hop:     {stats['1-hop (%)']:6.2f}%")
        report.append(f"  2-hop:     {stats['2-hop (%)']:6.2f}%")
        report.append(f"  Local:     {stats['local_<=2 (%)']:6.2f}%")
        report.append(f"  Long-range:{stats['long_range_>2 (%)']:6.2f}%")
        report.append("")
    
    report.append("")
    report.append("HEAD SPECIALIZATION (Last Layer)")
    report.append("-"*40)
    
    local_heads = sum(1 for p in head_profiles.values() if 'LOCAL' in p['type'])
    report.append(f"Local-focused heads: {local_heads}/{len(head_profiles)}")
    report.append("")
    
    for head, profile in head_profiles.items():
        report.append(f"  Head {head}: {profile['type']:12s} ({profile['local_pct']:.1f}% to 1-hop)")
    
    report.append("")
    report.append("")
    report.append("SUGGESTED RESEARCH DIRECTIONS")
    report.append("-"*40)
    report.append("1. Implement GraphViT to separate local (GNN) and global (Transformer) processing")
    report.append("2. Investigate why positional encodings aren't enabling long-range attention")
    report.append("3. Compare failure cases: do they require longer-range dependencies?")
    report.append("4. Try attention biasing to encourage long-range connections")
    
    report_text = "\n".join(report)
    
    with open(f'{output_dir}/analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nReport saved to {output_dir}/analysis_report.txt")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced Graph Transformer Attention Analysis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples for attention analysis')
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_args = checkpoint['args']
    
    # Set output directory
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_dir = f'{checkpoint_dir}/enhanced_analysis'
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Using device: {args.device}")
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader, num_node_features, num_edge_features, num_tasks = get_peptides_loaders(
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = GraphTransformer(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_tasks=num_tasks,
        hidden_dim=model_args['hidden_dim'],
        num_layers=model_args['num_layers'],
        num_heads=model_args['num_heads'],
        dropout=model_args['dropout'],
        pe_type=model_args['pe_type'],
        pe_dim=model_args['pe_dim'],
        use_edge_features=model_args['use_edge_features']
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel: {model_args['num_layers']} layers, {model_args['num_heads']} heads, "
          f"{model_args['hidden_dim']} hidden dim")
    
    # ==========================================================================
    # Run analyses
    # ==========================================================================
    
    # 1. Key analysis: attention by graph distance
    summary, head_distance_attention = analyze_attention_by_graph_distance(
        model, test_loader, args.device, args.output_dir, 
        num_samples=args.num_samples
    )
    
    # 2. Head specialization
    head_profiles = analyze_head_specialization(
        head_distance_attention, model.num_layers, model.num_heads, args.output_dir
    )
    
    # 3. Fixed entropy analysis
    analyze_attention_entropy_fixed(
        model, test_loader, args.device, args.output_dir,
        num_samples=min(50, args.num_samples)
    )
    
    # 4. Fixed t-SNE visualization
    visualize_embeddings_fixed(
        model, test_loader, args.device, args.output_dir,
        num_samples=500
    )
    
    # 5. Attention vs prediction quality
    analyze_attention_vs_performance(
        model, test_loader, args.device, args.output_dir,
        num_samples=args.num_samples
    )
    
    # 6. Generate summary report
    generate_summary_report(summary, head_profiles, args.output_dir)
    
    print("\n" + "="*70)
    print(f"All analyses complete! Results saved to: {args.output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()