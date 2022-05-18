from tqdm import tqdm
import torch
from torch import nn
from dataset import get_california_dataset
from model import LinearNet
from training import train_epoch, test_epoch


def main(depth_list, verbose: bool = False, num_epochs: int = 100):
    torch.manual_seed(0)
    train_loader, test_loader, in_dim, out_dim = get_california_dataset()
    criterion = nn.MSELoss()
    results = dict()
    for depth in depth_list:
        curr_results = {"train_losses": [], "test_losses": [], "grad_magnitudes": []}
        net = LinearNet(N=depth, in_dim=in_dim, out_dim=out_dim)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
        for epoch in tqdm(range(1, num_epochs+1)):
            train_loss, mag = train_epoch(criterion=criterion,
                                     net=net,
                                     optim=optimizer,
                                     train_loader=train_loader)
            test_loss, mag2 = test_epoch(criterion=criterion,
                                        net=net,
                                        test_loader=test_loader)
            # for data in test_loader:
            #     hess = hessian(net.forward, data[0])
            # print(f"hess = {hess}")
            curr_results["train_losses"].append(train_loss)
            curr_results["test_losses"].append(test_loss)
            curr_results["grad_magnitudes"].append(mag)

            if verbose and epoch % 10 == 0:
                print(f"train loss at epoch {epoch}: {train_loss}")
                print(f"test loss at epoch {epoch}: {test_loss}")
                print(f"grad magnitude: {mag}")

        results[depth] = curr_results


if __name__ == '__main__':
    depths = [2]
    main(depths, verbose=True)
