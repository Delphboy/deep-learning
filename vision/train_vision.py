import torch
import torch.nn as nn
import torch.nn.functional as F
from le_net import LeNet
from resnet import ResNet
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from torchvision.datasets.mnist import MNIST
from vision_transformer import VisionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate_le_net():
    transforms = trans.Compose(
        [
            trans.ToTensor(),
            trans.Normalize((0.1307,), (0.3081,)),
            trans.Resize((28, 28)),
        ]
    )
    train_dataset = MNIST(
        "../datasets/mnist", train=True, download=True, transform=transforms
    )
    test_dataset = MNIST(
        "../datasets/mnist", train=False, download=True, transform=transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = LeNet().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4 * 2)

    for epoch in range(10):
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item():.3f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        accuracies = []
        for i, (x, y) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            predictions = F.softmax(logits, dim=-1)
            predictions = predictions.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += len(y)
            acc = correct / total
            accuracies.append(acc)
            print(f"Batch: {i} | Accuracy: {acc}")

    print(f"Final accuracy: {sum(accuracies) / len(accuracies):.3f}")


def train_and_evaluate_resnet(size=50):
    assert size in [50, 101, 152], "Unsupported Resnet size"
    transforms = trans.Compose(
        [
            trans.ToTensor(),
            trans.Normalize((0.1307,), (0.3081,)),
            trans.Resize((28, 28)),
        ]
    )
    train_dataset = MNIST(
        "../datasets/mnist", train=True, download=True, transform=transforms
    )
    test_dataset = MNIST(
        "../datasets/mnist", train=False, download=True, transform=transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    if size == 50:
        model = ResNet.build_resnet_50(1, 10)
    if size == 101:
        model = ResNet.build_resnet_101(1, 10)
    if size == 152:
        model = ResNet.build_resnet_152(1, 10)

    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4 * 2)

    for epoch in range(10):
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item():.3f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        accuracies = []
        for i, (x, y) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            predictions = F.softmax(logits, dim=-1)
            predictions = predictions.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += len(y)
            acc = correct / total
            accuracies.append(acc)
            print(f"Batch: {i} | Accuracy: {acc}")

    print(f"Final accuracy: {sum(accuracies) / len(accuracies):.3f}")


def train_and_evaluate_vit():
    transforms = trans.Compose(
        [
            trans.ToTensor(),
            trans.Normalize((0.1307,), (0.3081,)),
            trans.Resize((28, 28)),
        ]
    )
    train_dataset = MNIST(
        "../datasets/mnist", train=True, download=True, transform=transforms
    )
    test_dataset = MNIST(
        "../datasets/mnist", train=False, download=True, transform=transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = VisionTransformer(512, 512, 512, 8).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(10):
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item():.3f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        accuracies = []
        for i, (x, y) in enumerate(test_dataloader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            predictions = F.softmax(logits, dim=-1)
            predictions = predictions.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += len(y)
            acc = correct / total
            accuracies.append(acc)
            print(f"Batch: {i} | Accuracy: {acc}")

    print(f"Final accuracy: {sum(accuracies) / len(accuracies):.3f}")


if __name__ == "__main__":
    # train_and_evaluate_le_net()
    # train_and_evaluate_resnet(50)
    train_and_evaluate_vit()
