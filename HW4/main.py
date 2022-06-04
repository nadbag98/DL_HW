import torch
from dataset import get_cifar_dls
from experiments import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.manual_seed(0)
    train_loader, test_loader = get_cifar_dls()

    exp_1(train_loader, test_loader, num_epochs=1000, lr=0.0005)


if __name__ == '__main__':
    main()
