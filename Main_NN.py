import torch
import NN
from NN import Normalize
from torch import nn

# NN1 and NN2 combined
class Main_Network(nn.Module):
    def __init__(self, x_size, z_size):
        super().__init__()
        self.net1 = NN.NN1(x_size, z_size)
        self.net2 = NN.NN2(x_size, z_size)
        
    def forward(self, x, z):
        output_xz = self.net1(x)    # Output from NN1
        output_xzx = self.net2(output_xz)    # Output from NN2 with NN1 as input       
        
        return output_xz, output_xzx