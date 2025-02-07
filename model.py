import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# class ANN(nn.Module):
#     def __init__(self, input_dim):
#         super(ANN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc2 = nn.Linear(64, 32)
#         # One neuron per binary label (outputs a logit that we pass through sigmoid)
#         self.out1 = nn.Linear(32, 1)
#         self.out2 = nn.Linear(32, 1)
    
#     def forward(self, x):
#         # Make sure x is a float tensor (since our features are int/bool)
#         x = x.float()
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         # Apply sigmoid to get probabilities directly.
#         prob1 = torch.sigmoid(self.out1(x))
#         prob2 = torch.sigmoid(self.out2(x))
#         return prob1, prob2
    

# input_dim = 55

# model = ANN(input_dim)
# optimizer = optim.Adam(model.parameter(), lr=0.01)
# criterion = nn.BCELoss()

