import torch
from torch import nn
from torch.autograd.functional import hessian


def train_epoch(net: nn.Module,
                optim: torch.optim.Optimizer,
                criterion: nn.Module,
                train_loader: torch.utils.data.DataLoader):
    epoch_loss = 0.0
    grad_magnitude = 0.0
    for data in train_loader:
        inputs, targets = data
        optim.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs.view(-1), targets)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
    for parameter in net.parameters():
        grad_magnitude += torch.linalg.norm(parameter)
    return epoch_loss, grad_magnitude


def test_epoch(net: nn.Module,
               criterion: nn.Module,
               test_loader: torch.utils.data.DataLoader):
    epoch_loss = 0.0
    grad_magnitude = 0.0
    for data in test_loader:
        inputs, targets = data
        outputs = net(inputs)
        loss = criterion(outputs.view(-1), targets)
        loss.retain_grad()
        loss.backward(retain_graph=True)
        epoch_loss += loss.item()
    for parameter in net.parameters():
        grad_magnitude += torch.linalg.norm(parameter)
    return epoch_loss, grad_magnitude
