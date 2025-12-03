import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
import os

def transform_edge_attr(data):
    """Convert edge attributes to float"""
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.float()
    return data

def get_peptides_loaders(batch_size=32, num_workers=4):
    """
    Load Peptides-func dataset with proper splits
    """
    # Download/load dataset
    path = './data/LRGB'
    
    train_dataset = LRGBDataset(root=path, name='Peptides-func', split='train', transform=transform_edge_attr)
    val_dataset = LRGBDataset(root=path, name='Peptides-func', split='val', transform=transform_edge_attr)
    test_dataset = LRGBDataset(root=path, name='Peptides-func', split='test', transform=transform_edge_attr)
    
    print(f"Dataset Statistics:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Num node features: {train_dataset.num_node_features}")
    print(f"Num edge features: {train_dataset.num_edge_features}")
    print(f"Num tasks: {train_dataset.num_classes}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_node_features, train_dataset.num_edge_features, train_dataset.num_classes