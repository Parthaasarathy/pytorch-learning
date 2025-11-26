#!/usr/bin/env python3
"""
FashionMNIST training script (MLP model).
"""

import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    print(
        f"Test Error:\n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f}\n"
    )

    return test_loss, accuracy


def main():
    args = get_args()

    device = (
        torch.device("mps")
        if getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)

    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root=args.data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        test_loss, acc = test(test_loader, model, loss_fn, device)

        ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "test_loss": test_loss,
                "accuracy": acc,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(args.save_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
