import torch
from torch import nn
import torch.nn.functional as F

# Model NN2 and NN2
class NN1(nn.Module):
    def __init__(self, x_size, z_size):
        super(NN1, self).__init__()
        self.dense1 = nn.Linear(x_size, 50)
        self.dense2 = nn.Linear(50, 50)
        self.dense3 = nn.Linear(50, 50)
        self.dense4 = nn.Linear(50, z_size)
        
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))       
        x = self.dense4(x)
        
        return x
    
class NN2(nn.Module):
    def __init__(self, x_size, z_size):
        super(NN2, self).__init__()
        self.dense1 = nn.Linear(z_size, 50)
        self.dense2 = nn.Linear(50, 50)
        self.dense3 = nn.Linear(50, 50)
        self.dense4 = nn.Linear(50, x_size)
        
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))       
        x = self.dense4(x)
        
        return x

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std