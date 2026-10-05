from torch_geometric.loader import DataLoader

from data_factory.data_loader import Dataset_MTT_Detection


def data_provider(args, flag):
    """flag: 'train' | 'val' | 'test' | 'pred' (all rows, e.g. to score a new CSV)."""
    data_set = Dataset_MTT_Detection(args, flag)
    shuffle = flag == "train"
    print(flag, len(data_set))
    # BatchNorm can't normalise a batch of ONE graph, so if the last training batch would
    # contain exactly 1 graph we skip it (it's a different random graph every epoch).
    drop_last = flag == "train" and len(data_set) % args.batch_size == 1
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader