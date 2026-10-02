from torch_geometric.loader import DataLoader

from data_factory.data_loader import Dataset_MTT_Detection


def data_provider(args, flag):
    """flag: 'train' | 'val' | 'test' | 'pred' (all rows, e.g. to score a new CSV)."""
    data_set = Dataset_MTT_Detection(args, flag)
    shuffle = flag == "train"
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_set, data_loader