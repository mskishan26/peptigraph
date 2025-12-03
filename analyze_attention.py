import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import get_peptides_loaders
from utils.visualization import (
    visualize_attention_matrix, 
    plot_attention_distance_distribution,
    plot_attention_entropy,
    visualize_embeddings
)
from models.transformer import GraphTransformer

def extract_layer_embeddings(model, data, device):
    """
    Extract embeddings from each transformer layer
    """
    model.eval()
    data = data.to(device)
    
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    
    # Compute positional encoding
    if model.pe is not None:
        pe = model.pe(edge_index, batch, x.shape[0])
        x = torch.cat([x, pe], dim=-1)
    
    # Encode node features
    x = model.node_encoder(x)
    x = torch.relu(x)
    
    # Encode edge features
    edge_attr_encoded = None
    if model.edge_encoder is not None and edge_attr is not None:
        edge_attr_encoded = model.edge_encoder(edge_attr)
    
    # Store embeddings from each layer
    layer_embeddings = [x.clone()]
    
    with torch.no_grad():
        for i, (transformer, norm) in enumerate(zip(model.transformer_layers, model.layer_norms)):
            x_residual = x
            x = transformer(x, edge_index, edge_attr_encoded)
            x = norm(x + x_residual)
            x = model.dropout(x)
            
            layer_embeddings.append(x.clone())
    
    return layer_embeddings


def analyze_attention_patterns(model, loader, device, output_dir, num_samples=10):
    """
    Analyze attention patterns across the dataset
    """
    print("\n" + "="*50)
    print("Analyzing Attention Patterns")
    print("="*50)
    
    model.eval()
    
    all_attention_weights = [[] for _ in range(model.num_layers)]
    all_attention_entropy = [[] for _ in range(model.num_layers)]
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if sample_count >= num_samples:
                break
            
            data = data.to(device)
            
            # Forward pass with attention
            _ = model(data, return_attention=True)
            attention_weights = model.get_attention_weights()
            
            # Store attention weights for each layer
            for layer_idx, (edge_index_att, att_weights) in enumerate(attention_weights):
                all_attention_weights[layer_idx].append(att_weights.cpu())
                
                # Compute entropy
                att_probs = att_weights.cpu().numpy()
                entropy = -np.sum(att_probs * np.log(att_probs + 1e-9), axis=0)
                all_attention_entropy[layer_idx].append(entropy)
            
            # Visualize attention for first few samples
            if sample_count < 5:
                print(f"\nVisualizing sample {sample_count + 1}...")
                
                # Get number of nodes in this graph
                num_nodes = (data.batch == 0).sum().item()
                
                # Visualize each layer's attention
                for layer_idx, (edge_index_att, att_weights) in enumerate(attention_weights):
                    # Get edges for first graph only
                    first_graph_mask = (edge_index_att[0] < num_nodes) & (edge_index_att[1] < num_nodes)
                    first_graph_edges = edge_index_att[:, first_graph_mask]
                    first_graph_att = att_weights[first_graph_mask]
                    
                    # Visualize first attention head
                    visualize_attention_matrix(
                        first_graph_att,
                        first_graph_edges,
                        num_nodes,
                        save_path=f'{output_dir}/attention_sample{sample_count}_layer{layer_idx}.png',
                        title=f'Sample {sample_count} - Layer {layer_idx}',
                        head_idx=0
                    )
            
            sample_count += 1
    
    # Aggregate statistics
    print("\n" + "="*50)
    print("Attention Statistics per Layer")
    print("="*50)
    
    for layer_idx in range(model.num_layers):
        print(f"\nLayer {layer_idx}:")
        
        # Concatenate all attention weights for this layer
        layer_attention = torch.cat(all_attention_weights[layer_idx], dim=0)
        layer_entropy = np.concatenate(all_attention_entropy[layer_idx], axis=0)
        
        print(f"  Mean attention: {layer_attention.mean():.4f}")
        print(f"  Std attention: {layer_attention.std():.4f}")
        print(f"  Mean entropy per head: {layer_entropy.mean(axis=0)}")
        
        # Plot entropy for this layer
        plot_attention_entropy(
            layer_attention,
            model.num_heads,
            save_path=f'{output_dir}/attention_entropy_layer{layer_idx}.png'
        )
        
        # Plot attention distance distribution for this layer
        # Use first batch as example
        first_batch = next(iter(loader)).to(device)
        _ = model(first_batch, return_attention=True)
        attention_weights = model.get_attention_weights()
        edge_index_att, att_weights = attention_weights[layer_idx]
        
        plot_attention_distance_distribution(
            att_weights,
            edge_index_att,
            model.num_heads,
            save_path=f'{output_dir}/attention_distance_layer{layer_idx}.png'
        )


def analyze_layer_representations(model, loader, device, output_dir, num_samples=500):
    """
    Analyze how representations evolve across layers
    """
    print("\n" + "="*50)
    print("Analyzing Layer Representations")
    print("="*50)
    
    model.eval()
    
    # Collect embeddings and labels
    all_layer_embeddings = [[] for _ in range(model.num_layers + 1)]  # +1 for input
    all_labels = []
    
    sample_count = 0
    
    with torch.no_grad():
        for data in loader:
            if sample_count >= num_samples:
                break
            
            data = data.to(device)
            
            # Extract layer embeddings
            layer_embeddings = extract_layer_embeddings(model, data, device)
            
            # Pool to graph level
            for layer_idx, emb in enumerate(layer_embeddings):
                graph_emb = torch.stack([
                    emb[data.batch == i].mean(dim=0)
                    for i in range(data.num_graphs)
                ])
                all_layer_embeddings[layer_idx].append(graph_emb.cpu())
            
            # Store first label for coloring (just use first task)
            all_labels.append(data.y[:, 0].cpu())
            
            sample_count += data.num_graphs
    
    # Concatenate all embeddings
    all_labels = torch.cat(all_labels, dim=0)
    
    # Visualize each layer's representations
    print("\nVisualizing layer representations...")
    for layer_idx in range(len(all_layer_embeddings)):
        layer_emb = torch.cat(all_layer_embeddings[layer_idx], dim=0)
        
        print(f"  Layer {layer_idx}: shape {layer_emb.shape}")
        
        visualize_embeddings(
            layer_emb,
            all_labels,
            method='tsne',
            save_path=f'{output_dir}/embeddings_layer{layer_idx}_tsne.png',
            title=f'Layer {layer_idx} Representations'
        )


def main():
    parser = argparse.ArgumentParser(description='Analyze attention patterns in trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_attention_samples', type=int, default=10)
    parser.add_argument('--num_embedding_samples', type=int, default=500)
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_args = checkpoint['args']
    
    # Set output directory
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_dir = f'{checkpoint_dir}/attention_analysis'
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Analyze attention patterns
    analyze_attention_patterns(
        model, test_loader, args.device, args.output_dir,
        num_samples=args.num_attention_samples
    )
    
    # Analyze layer representations
    analyze_layer_representations(
        model, test_loader, args.device, args.output_dir,
        num_samples=args.num_embedding_samples
    )
    
    print(f"\nAll analysis saved to {args.output_dir}/")


if __name__ == '__main__':
    main()