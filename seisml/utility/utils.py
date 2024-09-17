import gin
import numpy as np
from torch.utils.data import SubsetRandomSampler, SequentialSampler, DataLoader

@gin.configurable(blacklist=['ds'])
def split_dataset(ds, training_split=0.7, batch_size=128, shuffle=True, num_workers=10):
    """
    Helper to get training and testing datasets
    Returns:
        training dataset and testing dataloaders
    """
    ds_size = len(ds)
    indices = list(range(ds_size))
    offset = int(np.floor(training_split * ds_size))

    if shuffle:
        np.random.shuffle(indices)

        train_indices, test_indices = indices[:offset], indices[offset+1:]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
    else:
        split_idx = int(len(ds) * training_split)
        train_ds = ds[:split_idx]
        test_ds = ds[split_idx:]

        train_sampler = SequentialSampler(train_ds)
        test_sampler = SequentialSampler(test_ds)

    train_dl = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )
    test_dl = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers
    )

    return train_dl, test_dl