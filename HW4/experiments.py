from tqdm import trange
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
    train_accs = []
    test_accs = []
    t = trange(num_epochs)
    for epoch in t:
        train_loss, train_acc = train_epoch(criterion=criterion, net=model, optim=optimizer, train_loader=train_dl)
        test_loss, test_acc = test_epoch(criterion=criterion, net=model, test_loader=test_dl)
        t.set_postfix_str(f"train loss={round(train_loss,3)}, test loss={round(test_loss,3)}, "
                          f"train_acc={round(train_acc,3)}, test_acc={round(test_acc,3)}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    plot_result({"Train": train_losses, "Test": test_losses}, metric="loss", title="train_test_loss_gap")
    plot_result({"Train": train_accs, "Test": test_accs}, metric="accuracy", title="train_test_acc_gap")


def exp_2(num_epochs=150, lr=0.001):
    train_dl = get_random_dl()
    criterion = nn.CrossEntropyLoss()
    model = CIFARNet()
    model.to(device)
    print("Number of parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    train_accs = []
    t = trange(num_epochs)
    for epoch in t:
        train_loss, train_acc = train_epoch(criterion=criterion, net=model, optim=optimizer, train_loader=train_dl)
        t.set_postfix_str(f"train loss={round(train_loss, 3)}, "
                          f"train_acc={round(train_acc, 3)}")
        train_losses.append(train_loss)
        train_accs.append(train_acc)

    plot_result({"Train": train_accs}, metric="accuracy", title="train_acc_random_data")
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
    train_accs = []
    test_accs = []
    t = trange(num_epochs)
    for epoch in t:
        train_loss, train_acc = train_epoch(criterion=criterion, net=model, optim=optimizer, train_loader=train_dl)
        test_loss, test_acc = test_epoch(criterion=criterion, net=model, test_loader=test_dl)
        t.set_postfix_str(f"train loss={round(train_loss, 3)}, test loss={round(test_loss, 3)}, "
                          f"train_acc={round(train_acc, 3)}, test_acc={round(test_acc, 3)}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    plot_result({"Train": train_losses, "Test": test_losses}, metric="loss", title="generalization_half_random_loss")
    plot_result({"Train": train_accs, "Test": test_accs}, metric="accuracy", title="generalization_half_random_acc")


def exp_4(num_epochs=100, lr=0.00033):
    train_dl, test_dl = get_adverserial_cifar_dls()
    criterion = nn.CrossEntropyLoss()
    model = CIFARNet()
    model.to(device)
    print("Number of parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    t = trange(num_epochs)
    for epoch in t:
        train_loss, train_acc = train_epoch(criterion=criterion, net=model, optim=optimizer, train_loader=train_dl)
        test_loss, test_acc = test_epoch(criterion=criterion, net=model, test_loader=test_dl)
        t.set_postfix_str(f"train loss={round(train_loss, 3)}, test loss={round(test_loss, 3)}, "
                          f"train_acc={round(train_acc, 3)}, test_acc={round(test_acc, 3)}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    plot_result({"Train": train_losses, "Test": test_losses}, metric="loss",
                title="generalization_half_adversarial_loss")
    plot_result({"Train": train_accs, "Test": test_accs}, metric="accuracy",
                title="generalization_half_adversarial_acc")
