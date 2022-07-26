import torch
import data_generation as data
import numpy as np

class DataSet(torch.utils.data.Dataset):
    def __init__(self, system, M, K, a, b, N, samples):
        super().__init__()
        self.data = self.generate_data(system, M, K, a, b, N, samples)    # Generate synthetic data x, z, y
        self.data_length = self.data[0].shape[0]*self.data[0].shape[1]         # Total number of samples
        self.x_data = torch.from_numpy(self.data[0]).view(self.data_length, system.x_size)  # Convert to tensors and reshape
        self.z_data = torch.from_numpy(self.data[1]).view(self.data_length, system.z_size)
        self.ic = self.data[4]
        # Check if output is vector or scalar
        if self.data[2].ndim > 2:
            self.output_data = torch.from_numpy(self.data[2]).view(self.data[2].shape[0]*self.data[2].shape[1], self.data[2].shape[2])     # y data
        else:
            self.output_data = torch.from_numpy(self.data[2]).view(self.data[2].shape[0]*self.data[2].shape[1])
            
        self.time = torch.from_numpy(self.data[3])
        self.M = M
        self.K = K
        self.system = system
        
    def __len__(self):
        return self.data_length
    
    def generate_data(self,system, M, K, a, b, N, samples):
        ic = system.sample_ic(samples)
        #ic = np.random.dirichlet([1,1,1], size=samples)
        x_data, output, t = system.generate_data(ic, a, b, N)
        z_data = data.KKL_observer_data(M, K, output, a, b, N)
        
        return x_data, z_data, output, t, ic
    
    def normalize(self):
        self.mean_x = torch.mean(self.x_data, dim = 0)
        self.mean_z = torch.mean(self.z_data, dim = 0)
        self.mean_output = torch.mean(self.output_data, dim = 0)

        self.std_x = torch.std(self.x_data, dim = 0)
        self.std_z = torch.std(self.z_data, dim = 0)
        self.std_output = torch.std(self.output_data, dim = 0)

        self.x_data = (self.x_data - self.mean_x) / self.std_x
        self.z_data = (self.z_data - self.mean_z) / self.std_z
        self.output_data = (self.output_data - self.mean_output) / self.std_output       
        
    def __getitem__(self, idx):
        x = self.x_data[idx]
        z = self.z_data[idx]
        y = self.output_data[idx]
        return [x.float(), z.float(), y.float()]