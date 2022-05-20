import torch
from dataset import get_california_dataset
from experiments import part2_exp, part3_exp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.manual_seed(0)
    train_loader, test_loader, in_dim, out_dim = get_california_dataset()
    depths_list = [2, 3, 4]
    num_epochs = 100
    hidden_width = 10
    part2_exp(depths_list, num_epochs=num_epochs, hidden_width=hidden_width,
              in_dim=in_dim, out_dim=out_dim, dl=train_loader)

    depths_list = [2, 3]
    hidden_width = 20
    part3_exp(depths_list, num_epochs=num_epochs, hidden_width=hidden_width,
              in_dim=in_dim, out_dim=out_dim, dl=train_loader)


if __name__ == '__main__':
    main()
