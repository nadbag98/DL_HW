from tqdm import tqdm
import torch
from torch import nn
from model import LinearNet
from training import train_epoch
from plotting import plot_result

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def part2_exp(depth_list, verbose=False, num_epochs=100, hidden_width=10, in_dim=0, out_dim=0,
              dl: torch.utils.data.DataLoader = None):
    criterion = nn.MSELoss()
    results = dict()
    for depth in depth_list:
        curr_results = {"Train Loss": [], "Gradient Magnitude": []}
        net = LinearNet(N=depth, in_dim=in_dim, out_dim=out_dim, hidden_width=hidden_width)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
        for epoch in tqdm(range(1, num_epochs + 1)):
            train_loss, mag = train_epoch(criterion=criterion, net=net, optim=optimizer, train_loader=dl)
            curr_results["Train Loss"].append(train_loss)
            curr_results["Gradient Magnitude"].append(mag)
            if verbose and epoch % 10 == 0:
                print(f"train loss at epoch {epoch}: {train_loss}")
                print(f"grad magnitude: {mag}")
        results[depth] = curr_results

    plot_result(results=results, value_to_plot="Gradient Magnitude")
    plot_result(results=results, value_to_plot="Train Loss")
