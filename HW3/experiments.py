from tqdm import tqdm
import torch
import numpy as np
from scipy.linalg import fractional_matrix_power as mat_pow
from torch import nn, matmul
from model import LinearNet
from training import train_epoch, test_epoch
from plotting import plot_result
from hessian_eigenthings import compute_hessian_eigenthings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def part2_exp(depth_list, verbose=False, num_epochs=100, hidden_width=10, in_dim=0, out_dim=0,
              dl: torch.utils.data.DataLoader = None):
    criterion = nn.MSELoss()
    results = dict()
    for depth in depth_list:
        curr_results = {"Train Loss": [], "Gradient Magnitude": [], "Max Hessian Eigenval": [], "Min Hessian Eigenval": []}
        net = LinearNet(N=depth, in_dim=in_dim, out_dim=out_dim, hidden_width=hidden_width)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
        for epoch in tqdm(range(1, num_epochs + 1)):
            train_loss, mag = train_epoch(criterion=criterion, net=net, optim=optimizer, train_loader=dl)
            curr_results["Train Loss"].append(train_loss)
            curr_results["Gradient Magnitude"].append(mag)
            num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            # we use code from https://github.com/noahgolmant/pytorch-hessian-eigenthings
            # to compute eigenvals
            eigenvals, _ = compute_hessian_eigenthings(net, dl, criterion, num_params)
            curr_results["Max Hessian Eigenval"].append(float(np.max(eigenvals)))
            curr_results["Min Hessian Eigenval"].append(float(np.min(eigenvals)))
            if verbose and epoch % 10 == 0:
                print(f"train loss at epoch {epoch}: {train_loss}")
                print(f"grad magnitude: {mag}")
        results[depth] = curr_results

    plot_result(results=results, value_to_plot="Gradient Magnitude")
    plot_result(results=results, value_to_plot="Train Loss")
    plot_result(results=results, value_to_plot="Max Hessian Eigenval")
    plot_result(results=results, value_to_plot="Min Hessian Eigenval")


def part3_exp(depth_list, num_epochs=100, hidden_width=10, in_dim=0, out_dim=0,
              dl: torch.utils.data.DataLoader = None, lr=0.0001):
    criterion = nn.MSELoss()
    results = dict()
    for depth in depth_list:
        curr_results = {"Norm of Difference": [], "Actual Loss": [], "Analytical Loss": []}
        net = LinearNet(N=depth, in_dim=in_dim, out_dim=out_dim, hidden_width=hidden_width)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        W = net.get_e2e_vec()
        analytical_net = nn.Linear(in_dim, out_dim, bias=False)
        analytical_net.weight.data = W
        for epoch in tqdm(range(1, num_epochs + 1)):
            train_loss, _ = train_epoch(criterion=criterion, net=net, optim=optimizer, train_loader=dl)
            actual_e2e = net.get_e2e_vec()
            analytical_loss = test_epoch(analytical_net, criterion=criterion, test_loader=dl)
            analytical_grad = analytical_net.weight.grad.cpu()
            for j in range(1, depth):
                left_side = mat_pow(matmul(W, W.T).cpu(), (j-1)/depth)
                left_side = torch.tensor(left_side)
                right_side = mat_pow(matmul(W.T, W).cpu(), 1 - (j/depth))
                right_side = torch.tensor(right_side.real)
                update = matmul(matmul(left_side, analytical_grad), right_side)
                W -= lr * update.to(device)
            analytical_net.weight.data = W
            curr_results["Norm of Difference"].append(torch.linalg.norm(W - actual_e2e).item())
            curr_results["Actual Loss"].append(train_loss)
            curr_results["Analytical Loss"].append(analytical_loss)

        results[depth] = curr_results

    plot_result(results=results, value_to_plot="Norm of Difference")
    plot_result(results=results, value_to_plot="Actual Loss")
    plot_result(results=results, value_to_plot="Analytical Loss")
