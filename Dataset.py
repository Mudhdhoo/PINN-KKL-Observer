from random import sample
import torch
import data_generation as data

class DataSet(torch.utils.data.Dataset):
    """
    Dataset class to generate synthetic x, y and z data.
    The set is split up into data for normal loss and data for physics loss.

    -------- Parameters --------
    system: 
        A system object created from classes within Systems.py

    M:
        M matrix in the z dynamics.

    K:
        K matrix in the z dynamics.

    a:
        Start time of ODE solver.

    b:
        End time of ODE solver.

    samples:
        Numer of initial conditions to be sampled for data generation.

    limits_normal:
        Limits on the state sample space of initial conditions used to generate
        data to compute the normal loss.

    sample_mode: int
        Either 1 or 2. 
        If 1, the samples for the physics datapoints will be generated from a separate set of
        initial conditions. To set a different limit on the state sample space for the physics
        points, use the set_physics_limit method. 
        If 2, These points will be generated from the same initial conditions,
        with every other sample in a given trajectory being assigned as a physics datapoint.
        Default set to 2.
    """

    def __init__(self, system, M, K, a, b, N, samples, limits_normal, sample_mode = 2):
        super().__init__()
        self.sample_mode = sample_mode
        self.train_data = self.generate_data(system, M, K, a, b, N, samples, limits_normal, seed = 888)    # Generate synthetic data x, z, y
        self.data_length = self.train_data[0].shape[0]*self.train_data[0].shape[1]         # Total number of samples
        x = torch.from_numpy(self.train_data[0]).view(self.data_length, system.x_size)  # Convert to tensors and reshape
        z = torch.from_numpy(self.train_data[1]).view(self.data_length, system.z_size)
        # Check if output is vector or scalar
        if self.train_data[2].ndim > 2:
            y = torch.from_numpy(self.train_data[2]).view(self.train_data[2].shape[0]*self.train_data[2].shape[1], self.train_data[2].shape[2])     # y data
        else:
            y = torch.from_numpy(self.train_data[2]).view(self.train_data[2].shape[0]*self.train_data[2].shape[1])

        if sample_mode == 1:
            # ----------------------- Normal loss data ----------------------- 
            self.x_data = x
            self.z_data = z
            self.output_data = y
            self.ic = self.train_data[4]
            #-----------------------------------------------------------------

            # ----------------------- Physics loss data ----------------------- 
            self.limit_physics = limits_normal
            self.train_data_ph = self.generate_data(system, M, K, a, b, N ,samples, self.limit_physics, seed = 8888)
            self.data_length_ph = self.train_data_ph[0].shape[0]*self.train_data_ph[0].shape[1]         # Total number of samples
            self.x_data_ph = torch.from_numpy(self.train_data_ph[0]).view(self.data_length_ph, system.x_size)
            self.z_data_ph = torch.from_numpy(self.train_data_ph[1]).view(self.data_length_ph, system.z_size)
            self.ic_ph = self.train_data_ph[4]
            # Check if output is vector or scalar    
            if self.train_data_ph[2].ndim > 2:
                self.output_data_ph = torch.from_numpy(self.train_data_ph[2]).view(self.train_data_ph[2].shape[0]*self.train_data_ph[2].shape[1], self.train_data_ph[2].shape[2])     # y data
            else:
                self.output_data_ph = torch.from_numpy(self.train_data_ph[2]).view(self.train_data_ph[2].shape[0]*self.train_data_ph[2].shape[1])
            #--------------------------------------------------------------
        elif sample_mode == 2:
            self.data_length = int(self.data_length / 2)
            self.x_data = x[::2]
            self.z_data = z[::2]
            self.output_data = y[::2]
            self.x_data_ph = x[1::2]
            self.z_data_ph = z[1::2]
            self.output_data_ph = y[1::2]
        else:
            raise Exception('Sample mode must be either 1 or 2.')

        # ----------------------- Mean and standard deviation -----------------------     
        self.mean_x = torch.mean(self.x_data, dim = 0)
        self.mean_z = torch.mean(self.z_data, dim = 0)
        self.mean_output = torch.mean(self.output_data, dim = 0)
        self.std_x = torch.std(self.x_data, dim = 0)
        self.std_z = torch.std(self.z_data, dim = 0)
        self.std_output = torch.std(self.output_data, dim = 0)

        self.mean_x_ph = torch.mean(self.x_data_ph, dim = 0)
        self.mean_z_ph = torch.mean(self.z_data_ph, dim = 0)
        self.mean_output_ph = torch.mean(self.output_data_ph, dim = 0)
        self.std_x_ph = torch.std(self.x_data_ph, dim = 0)
        self.std_z_ph = torch.std(self.z_data_ph, dim = 0)
        self.std_output_ph = torch.std(self.output_data, dim = 0)
        #-----------------------------------------------------------------------------
        self.time = torch.from_numpy(self.train_data[3])
        self.M = M
        self.K = K
        self.system = system
        
    def __len__(self):
        return self.data_length
    
    def generate_data(self,system, M, K, a, b, N, samples, limits, seed):
        ic = system.sample_ic(limits, samples, seed = seed)
        x_data, output, t = system.generate_data(ic, a, b, N)
        z_data = data.KKL_observer_data(M, K, output, a, b, N)
        
        return x_data, z_data, output, t, ic

    def set_physics_limits(self, limits):
        """
        Sets the limits of the state sample space for the physics datapoints.
        Can only be used if sample_mode is 2.
        """
        if self.sample_mode == 2:
            self.limit_physics = limits
        else:
            raise Exception('Can only set limits if sample mode is 1.')  
    
    def normalize(self):
        """
        Old method to normalize all the data before training.
        Use the normalizer class instead.
        """
        self.x_data = (self.x_data - self.mean_x) / self.std_x
        self.z_data = (self.z_data - self.mean_z) / self.std_z
        self.output_data = (self.output_data - self.mean_output) / self.std_output       
        
    def __getitem__(self, idx):
        x = self.x_data[idx]
        z = self.z_data[idx]
        y = self.output_data[idx]
        x_ph = self.x_data_ph[idx]
        y_ph = self.output_data_ph[idx]
        return [x.float(), z.float(), y.float(), x_ph.float(), y_ph.float()]
      
