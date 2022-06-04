import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


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
