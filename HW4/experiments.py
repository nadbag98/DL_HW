from tqdm import tqdm, trange
import torch
import numpy as np
from torch import nn
from torch.utils import data
from model import *
from dataset import *
from training import train_epoch, test_epoch
from plotting import plot_result

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def exp_1(num_epochs=100, lr=0.00005):
    train_dl, test_dl = get_cifar_dls()
    criterion = nn.CrossEntropyLoss()
    model = CIFARNet()
    model.to(device)
    print("Number of parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    t = trange(num_epochs)
    for epoch in t:
        train_loss = train_epoch(criterion=criterion, net=model, optim=optimizer, train_loader=train_dl)
        test_loss = test_epoch(criterion=criterion, net=model, test_loader=test_dl)
        t.set_postfix_str(f"train loss={round(train_loss,3)}, test loss={round(test_loss,3)}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plot_result({"Train": train_losses, "Test": test_losses}, metric="loss", title="train_test_loss_gap")


def exp_2(num_epochs=150, lr=0.001):
    train_dl = get_random_dl()
    criterion = nn.CrossEntropyLoss()
    model = CIFARNet()
    model.to(device)
    print("Number of parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    t = trange(num_epochs)
    for epoch in t:
        train_loss = train_epoch(criterion=criterion, net=model, optim=optimizer, train_loader=train_dl)
        t.set_postfix_str(f"train loss={round(train_loss, 3)}")
        train_losses.append(train_loss)

    plot_result({"Train": train_losses}, metric="loss", title="train_loss_random_data")


def exp_3(num_epochs=100, lr=0.00033):
    train_dl, test_dl = get_half_random_dls()
    criterion = nn.CrossEntropyLoss()
    model = CIFARNet()
    model.to(device)
    print("Number of parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    t = trange(num_epochs)
    for epoch in t:
        train_loss = train_epoch(criterion=criterion, net=model, optim=optimizer, train_loader=train_dl)
        test_loss = test_epoch(criterion=criterion, net=model, test_loader=test_dl)
        t.set_postfix_str(f"train loss={round(train_loss, 3)}, test loss={round(test_loss, 3)}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plot_result({"Train": train_losses, "Test": test_losses}, metric="loss", title="generalization_half_random")


def exp_4(num_epochs=100, lr=0.00033):
    train_dl, test_dl = get_adverserial_cifar_dls()
    criterion = nn.CrossEntropyLoss()
    model = CIFARNet()
    model.to(device)
    print("Number of parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    t = trange(num_epochs)
    for epoch in t:
        train_loss = train_epoch(criterion=criterion, net=model, optim=optimizer, train_loader=train_dl)
        test_loss = test_epoch(criterion=criterion, net=model, test_loader=test_dl)
        t.set_postfix_str(f"train loss={round(train_loss, 3)}, test loss={round(test_loss, 3)}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plot_result({"Train": train_losses, "Test": test_losses}, metric="loss", title="generalization_half_adversarial")
