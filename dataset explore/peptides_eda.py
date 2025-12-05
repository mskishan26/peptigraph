

import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_peptides_datasets(root: str = "data/LRGB"):
    """
    Load LRGB Peptides-func train/val/test datasets.
    """
    train_dataset = LRGBDataset(root=root, name="peptides-func", split="train")
    val_dataset = LRGBDataset(root=root, name="peptides-func", split="val")
    test_dataset = LRGBDataset(root=root, name="peptides-func", split="test")

    print("Loaded Peptides-func dataset:")
    print("  train:", len(train_dataset))
    print("  val  :", len(val_dataset))
    print("  test :", len(test_dataset))

    return train_dataset, val_dataset, test_dataset


def create_peptides_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Create PyG DataLoader for train/val/test.
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader



def summarize_dataset(dataset, name: str = "dataset", show_example: bool = True):
    """
    Print basic information of a dataset, optionally show the first graph.
    """
    print(f"\nSummary of {name}:")
    print(f"  #graphs: {len(dataset)}")

    if not show_example or len(dataset) == 0:
        return

    g = dataset[0]
    print("\nExample graph 0:")
    print(g)
    print("  x shape        :", g.x.shape)
    print("  edge_index shape:", g.edge_index.shape)
    print("  edge_attr shape:", g.edge_attr.shape if g.edge_attr is not None else None)
    print("  y shape        :", g.y.shape)



def plot_graph_size_stats(dataset, name: str = "train"):
    """
    Plot node-count and edge-count distributions for a dataset.
    """
    num_nodes_list = [g.num_nodes for g in dataset]
    num_edges_list = [g.edge_index.size(1) for g in dataset]

    print(f"\nGraph size stats ({name}):")
    print("  Avg #nodes :", np.mean(num_nodes_list))
    print("  Median #nodes :", np.median(num_nodes_list))
    print("  Avg #edges :", np.mean(num_edges_list))
    print("  Median #edges :", np.median(num_edges_list))

    plt.figure(figsize=(6, 4))
    plt.hist(num_nodes_list, bins=50)
    plt.xlabel("Num nodes per graph")
    plt.ylabel("Count")
    plt.title(f"Node count distribution ({name})")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(num_edges_list, bins=50)
    plt.xlabel("Num edges per graph")
    plt.ylabel("Count")
    plt.title(f"Edge count distribution ({name})")
    plt.tight_layout()
    plt.show()



def get_label_matrix(dataset):
    """
    Stack all y into a matrix of shape [N_graphs, num_labels].
    """
    ys = torch.cat([g.y for g in dataset], dim=0)  # [N, num_labels]
    return ys.numpy()


def plot_label_stats(dataset, name: str = "train"):
    """
    Plot label frequency and labels-per-graph distribution.
    """
    ys_np = get_label_matrix(dataset)
    num_graphs, num_labels = ys_np.shape

    print(f"\nLabel stats ({name}):")
    print("  Label matrix shape:", ys_np.shape)
    print("  Unique values:", np.unique(ys_np))


    pos_counts = ys_np.sum(axis=0)

    plt.figure(figsize=(6, 4))
    x = np.arange(num_labels)
    plt.bar(x, pos_counts)
    plt.xlabel("Label index")
    plt.ylabel("# positive graphs")
    plt.title(f"Positive count per label ({name})")
    plt.tight_layout()
    plt.show()

    labels_per_graph = ys_np.sum(axis=1)
    print("  Avg labels per graph:", labels_per_graph.mean())
    print("  Median labels per graph:", np.median(labels_per_graph))

    plt.figure(figsize=(6, 4))
    plt.hist(labels_per_graph, bins=range(0, num_labels + 2))
    plt.xlabel("#labels per graph")
    plt.ylabel("Count")
    plt.title(f"Labels per graph distribution ({name})")
    plt.tight_layout()
    plt.show()



