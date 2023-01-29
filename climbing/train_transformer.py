from climbing.model_transformer import ClimbingSimpleViT
from climbing.dataset import ClimbDatasetTransformer
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from typing import Union, Literal

from pathlib import Path


def train(args, model, device, train_loader, optimizer, epoch, log_interval=10, loss_type="regression"):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if loss_type == "regression":
            loss = F.mse_loss(output, target[..., None])
        elif loss_type == "classification":
            target = target.long()
            loss = F.nll_loss(output, target)
        else:
            ValueError()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, loss_type="regression"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if loss_type == "regression":
                test_loss += F.mse_loss(output, target[..., None], reduction="sum").item()
            elif loss_type == "classification":
                target = target.long()
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if loss_type == "regression":
        print(f'\nTest set: Average loss: {test_loss:.4f}\n')
    elif loss_type == "classification":
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    train_kwargs = {'batch_size': 8}
    test_kwargs = {'batch_size': 8}

    model = ClimbingSimpleViT(
        hold_data_len=2+1,
        dim = 512,
        depth = 6,
        heads = 16,
        mlp_dim = 1024
    )
    if Path("_datasets/cache/climb_dataset_transformer.pt").exists():
        dataset = torch.load("_datasets/cache/climb_dataset_transformer.pt")
    else:
        dataset = ClimbDatasetTransformer()
        torch.save(dataset, "_datasets/cache/climb_dataset_transformer.pt")
    train_set_size = int(len(dataset)*0.9)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_set_size, test_set_size], generator=torch.Generator().manual_seed(123))
    train_loader = torch.utils.data.DataLoader(train_set,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    device="cpu"

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 25):
        train(model, device, train_loader, optimizer, epoch, log_interval=10)
        test(model, device, test_loader)
        # scheduler.step()

    torch.save(model.state_dict(), "kilter_transformer.pt")

if __name__ == "__main__":
    main()