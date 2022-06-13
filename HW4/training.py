import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(net: nn.Module,
                optim: torch.optim.Optimizer,
                criterion: nn.Module,
                train_loader: torch.utils.data.DataLoader):
    epoch_loss = 0.0
    num_correct = 0
    for data in train_loader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        outputs = net(inputs)
        preds = torch.argmax(outputs, dim=1)
        num_correct += torch.sum(torch.eq(preds, targets)).item()
        loss = criterion(outputs, targets)
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    epoch_acc = num_correct / len(train_loader.dataset)
    return epoch_loss, epoch_acc


def test_epoch(net: nn.Module,
               criterion: nn.Module,
               test_loader: torch.utils.data.DataLoader):
    epoch_loss = 0.0
    num_correct = 0
    net.zero_grad()
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            preds = torch.argmax(outputs, dim=1)
            num_correct += torch.sum(torch.eq(preds, targets)).item()
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
    epoch_loss /= len(test_loader)
    epoch_acc = num_correct / len(test_loader.dataset)
    return epoch_loss, epoch_acc
