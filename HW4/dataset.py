import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset


def get_cifar_dls(batch_size=128):
    # taken from https://pytorch.org/hub/pytorch_vision_inception_v3/
    # preprocess = transforms.Compose([
    #     transforms.Resize(299),
    #     transforms.CenterCrop(299),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # train_set = datasets.CIFAR10("./data", train=True, download=True, transform=preprocess)
    # test_set = datasets.CIFAR10("./data", train=False, download=True, transform=preprocess)

    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
    train_set = Subset(train_set, range(5000))
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())
    test_set = Subset(test_set, range(1000))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class RandomDataset(Dataset):
    def __init__(self, X, y):
        super(RandomDataset, self).__init__()
        self.values = X
        self.labels = y

    def __len__(self):
        return len(self.values)  # number of samples in the dataset

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


def get_random_dl(batch_size=128):
    X = torch.rand((5000, 3, 32, 32))
    y = torch.randint(high=10, size=(5000, 1)).view(-1)
    train_set = RandomDataset(X, y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    return train_loader


def get_half_random_dls(batch_size=128):
    cifar_set = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())

    cifar_train_data = torch.tensor(cifar_set.data[:2500])
    cifar_train_labels = torch.tensor(cifar_set.targets[:2500])
    random_train_data = torch.rand((2500, 32, 32, 3))
    random_train_labels = torch.randint(high=10, size=(2500, 1))

    train_data = torch.cat(cifar_train_data, random_train_data)
    train_labels = torch.cat(cifar_train_labels, random_train_labels)

    train_set = RandomDataset(train_data, train_labels)

    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())
    test_set = Subset(test_set, range(1000))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
