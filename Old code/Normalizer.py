import torch

class Normalizer:
    def __init__(self, dataset):
        self.x_size = dataset.system.x_size
        self.z_size = dataset.system.z_size
        self.mean_x = dataset.mean_x
        self.std_x = dataset.std_x
        self.mean_z = dataset.mean_z
        self.std_z = dataset.std_z
        self.sys = dataset.system

    def check_sys(self, tensor):
        if tensor.size()[1] == self.sys.x_size:
            mean = self.mean_x
            std = self.std_x
        elif tensor.size()[1] == self.sys.z_size:
            mean = self.mean_z
            std = self.std_z
        else:
            raise Exception('Size of tensor unmatched with any system.')     

        return mean, std

    def Normalize(self, tensor):
        mean, std = self.check_sys(tensor)            
        return (tensor - mean) / std

    def Denormalize(self, tensor):
        mean, std = self.check_sys(tensor)           
        return tensor*std + mean        