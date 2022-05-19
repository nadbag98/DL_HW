import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing


class HousingDataset(torch.utils.data.Dataset):
    """
  Prepare the California Housing dataset for regression
  Code was taken from https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
  """

    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def get_california_dataset():
    X, y = fetch_california_housing(return_X_y=True)
    dataset = HousingDataset(X, y)
    num_samples = X.shape[0]
    train_size = int(num_samples * 0.01)
    test_size = num_samples - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    # setting batch sizes equal to set size in order to run full batch GD
    train_loader = DataLoader(train_set, batch_size=train_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=train_size, shuffle=False)
    in_dim = X.shape[1]
    out_dim = 1 if len(y.shape) == 1 else y.shape[1]
    return train_loader, test_loader, in_dim, out_dim
