import wandb
import torch
from pathlib import Path

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from typing import Dict

# import modelconfig.config as mcfg
import models.modeldef as modeldef
from data import load_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: nn.Module, optimizer: torch.optim, path: str, epoch: int):
    p = Path(path)

    if not p.exists():
        p.mkdir()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    checkpoint_name = "checkpoint_{}_epoch.pkl".format(epoch)
    checkpoint_path = p / checkpoint_name
    torch.save(checkpoint, checkpoint_path)


def train(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim,
    criterion: nn.modules.loss,
    model: nn.Module,
):
    model.to(device)
    for epoch in range(config.get("epochs")):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss
            if i % 1000 == 999:
                wandb.log({"loss": running_loss / 2000})
                running_loss = 0.0

        if epoch % config.get("checkpoint_interval") == 0:
            save_checkpoint(
                model=model, optimizer=optimizer, path="checkpoints", epoch=epoch
            )

        val_acc = val(model, val_loader)
        wandb.log({"Accuracy": val_acc})

    torch.save(
        model,
    )


def val(model: nn.Module, val_loader: DataLoader):
    model.to(device)
    model.eval()

    with torch.no_grad():
        running_acc = 0.0

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            equals = outputs.topk(1)[1].view(labels.shape) == labels
            running_acc += equals.sum().item()
        acc = running_acc / len(val_loader.dataset)

    return acc


if __name__ == "__main__":
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="pytorch template",
        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "architecture": "resnet18 with pretrained weights",
            "dataset": "cifar10",
            "epochs": 2,
            "checkpoint_interval": 10,
            "device": device,
        },
    )

    train_loader, val_loader = load_data.load()
    optimizer = modeldef.optimizer
    criterion = modeldef.loss_fn
    model = modeldef.realmodel

    wandb.watch(model)

    # Train the model
    train(
        wandb.config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        model=model,
    )

    wandb.finish()
