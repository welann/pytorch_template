import torch
import torch.nn as nn
import torch.optim as optim



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(400, 120)   
        self.fc2 = nn.Linear(120, 84)   
        self.fc3 = nn.Linear(84,10)

        self.relu = nn.ReLU()   
        self.maxpool=nn.MaxPool2d(2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        
        x=x.view(x.size()[0],-1)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)

        return x



# Define model, loss function, and optimizer
realmodel = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(realmodel.parameters(), lr=0.001, momentum=0.9)




if __name__ == '__main__':
    print(realmodel)
    inputdata=torch.randn(1, 3, 32, 32)
    print(realmodel(inputdata))

