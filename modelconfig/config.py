import torch

input_size=784
hidden_size=50
output_size=10

epochs=10

num_workers=2

batch_size=8
learning_rate=0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")