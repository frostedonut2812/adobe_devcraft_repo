import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

class ProbabilityANN(nn.Module):
    def __init__(self, input_dim=53, hidden_dims=[64, 128, 64, 32, 16], output_dim=2):
        super(ProbabilityANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.fc5 = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.fc6 = nn.Linear(hidden_dims[4], output_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

if __name__=='__main__':
    model = ProbabilityANN()
    summary(model, (1,53), batch_size=2)    