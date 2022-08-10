import torch
from torch import nn

# Model NN2 and NN2
class NN(nn.Module):
    def __init__(self, num_hidden, hidden_size, in_size, out_size, activation):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        current_dim = in_size
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(current_dim, hidden_size))
            current_dim = hidden_size
        self.layers.append(nn.Linear(current_dim, out_size))

    def forward(self, x):
        # Normalize input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)

        # Denormalize output

        return x
    
# NN1 and NN2 combined
class Main_Network(nn.Module):
    def __init__(self, x_size, z_size, num_hidden, hidden_size, activation):
        super().__init__()
        self.net1 = NN(num_hidden, hidden_size, x_size, z_size, activation)
        self.net2 = NN(num_hidden, hidden_size, z_size, x_size, activation)
        
    def forward(self, x, z):
        output_xz = self.net1(x)    # Output from NN1
        output_xzx = self.net2(output_xz)    # Output from NN2 with NN1 as input       
        
        return output_xz, output_xzx
