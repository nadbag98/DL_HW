import torch
from experiments import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.manual_seed(0)

    # exp_1()

    exp_2()


if __name__ == '__main__':
    main()
