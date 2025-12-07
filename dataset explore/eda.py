from peptides_eda import (
    load_peptides_datasets,
    create_peptides_loaders,
    summarize_dataset,
    plot_graph_size_stats,
    plot_label_stats,
)

train_ds, val_ds, test_ds = load_peptides_datasets("data/LRGB")
train_loader, val_loader, test_loader = create_peptides_loaders(
    train_ds, val_ds, test_ds, batch_size=32
)




summarize_dataset(train_ds, name="train")
plot_graph_size_stats(train_ds, name="train")
plot_label_stats(train_ds, name="train")