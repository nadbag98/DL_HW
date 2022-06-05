import torch
from experiments import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.manual_seed(0)

    # outputs train_test_loss_gap.png
    exp_1()

    # outputs train_loss_random_data.png
    # exp_2()

    # outputs generalization_half_random.png
    # exp_3()

    # outputs generalization_half_adversarial.png
    # exp_4()

if __name__ == '__main__':
    main()
