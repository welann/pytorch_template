import wandb
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from typing import Dict

# import modelconfig.config as mcfg
import models.modeldef as modeldef
from data import load_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
    config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer:torch.optim,
    criterion:nn.modules.loss,
    model: nn.Module,
):
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

        val_acc=val(model, val_loader)
        wandb.log({"Accuracy": val_acc})


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
            "architecture": "mlp",
            "dataset": "mnist",
            "epochs": 2,
            "device": device,
        },
    )

    train_loader, val_loader = load_data.load()
    optimizer = modeldef.optimizer
    criterion = modeldef.loss_fn
    model = modeldef.realmodel

    wandb.watch(model)

    # Train the model
    train(wandb.config,train_loader=train_loader,val_loader=val_loader,optimizer=optimizer,criterion=criterion,model=model)

    wandb.finish()
