from tqdm import tqdm
import torch
import numpy as np
from torch import nn
from torch.utils import data
from model import *
from training import train_epoch, test_epoch
from plotting import plot_result

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def exp_1(train_dl: data.DataLoader, test_dl: data.DataLoader, num_epochs=100, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    model = CIFARNet()
    model.to(device)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss = train_epoch(criterion=criterion, net=model, optim=optimizer, train_loader=train_dl)
        test_loss = test_epoch(criterion=criterion, net=model, test_loader=test_dl)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plot_result(train_losses, test_losses, metric="loss")
