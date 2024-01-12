import wandb
import random
import torch

from torch.autograd import Variable

# import modelconfig.config as mcfg
import models.modeldef as modeldef
from data import load_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def val(model,val_loader):
    model.to(device)
    model.eval()

    with torch.no_grad():
        running_acc = 0.0

        for (images, labels) in val_loader:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            equals = outputs.topk(1)[1].view(labels.shape) == labels          
            running_acc += equals.sum().item()
        acc = running_acc/len(val_loader.dataset) 
        print('\nAccuracy: {acc:.5f}') 
        
    return acc

if __name__ == '__main__':

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
        "device": device
        }
    )
    print(1)

    train_loader, val_loader=load_data.load()

    optimizer=modeldef.optimizer
    criterion=modeldef.loss_fn
    model=modeldef.realmodel
    print(2)
    wandb.watch(model)

    # Train the model
    for epoch in range(2):

        running_loss=0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss+=loss
            if i%2000==1999:
                wandb.log({"loss": running_loss/2000})


    print(3)
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
