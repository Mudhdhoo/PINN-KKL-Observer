import torch
import numpy as np
import data_generation as data
from torch.autograd.functional import jacobian
import matplotlib as plt

class System_z:
    """
    Dynamics of the z-system for both autonomous and non-autonomous cases.
    Autonomous:
        z_dot = Mz(t) + Ky(t)
        
    Non-Autonomous:
        z_dot = Mz(t) + Ky(t) + phi(t, z(t))*(u(t) - u0(t))
        phi(t, z(t)) = dT/dx*g

    """
    def __init__(self, M, K, system):
        self.M = M
        self.K = K
        self.is_autonomous = True if system.input == None else False
        self.y_size = system.y_size

    def z_dynamics(self):
        if self.is_autonomous:
            if self.y_size > 1:
                z_dot = lambda y,z: np.matmul(self.M,z) + np.matmul(self.K, np.expand_dims(y,1))
            else:
                z_dot = lambda y,z: np.matmul(self.M,z) + self.K*y 
        
        else:
            if self.y_size > 1:
                z_dot = lambda y,q,z: np.matmul(self.M,z) + np.matmul(self.K, np.expand_dims(y,1)) + q
            else:
                z_dot = lambda y,q,z: np.matmul(self.M,z) + self.K*y + q 

        return z_dot

class Observer:
    def __init__(self, system, z_system, net):
        self.system = system
        self.z_system = z_system
        self.f = z_system.z_dynamics()
        self.T = net.net1
        self.T_inv = net.net2
    
    def simulate_NA(self, a, b, N, ic, u0, g):
        """
        Online simulation of observer for non-autonomous input-affine systems by the following steps:
        1. Generate y data from the system with input u.
        2. Use the y data to simulate observer dynamics:
            z_dot = Mz(t) + Ky(t) + phi(t, z(t))*(u(t) - u0(t))
            phi(t, z(t)) = dT/dx*g
        3. Use T_inv, the inverse of T to simulate:
            x_hat = T_inv(t, z)

        """
        x, y, t = self.system.generate_data(ic, a, b, N)     # Generate y data
        x = torch.from_numpy(np.reshape(x, (N+1,self.system.x_size)))
        u = self.system.input
        size_z = self.z_system.M.shape[0]
        h = (b-a) / N

        z = [[0]*size_z]
        v = np.array(z).T

        y = np.squeeze(y)
        if y.ndim > 2:
            y = y[1:, :]
        else:
            y = np.delete(y,0)

        for idx, output in enumerate(y):
            with torch.no_grad():
                x_hat = self.T_inv(torch.tensor(z[-1]).float())
            u_sub_u0 = u(t[idx]) - u0(t[idx])
            dTdx = jacobian(self.T, x_hat).numpy()
            dTdx_mul_g = np.matmul(dTdx, g)

            q = dTdx_mul_g*u_sub_u0     # phi*(u(t) - u0(t))

            k1 = self.f(output,q, v)
            k2 = self.f(output,q, v + h/2*k1)
            k3 = self.f(output,q, v + h/2*k2)
            k4 = self.f(output,q, v + h*k3)
            
            v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

            a = np.reshape(v.T, size_z)
            z.append(np.ndarray.tolist(a)) 

        z = np.array(z)
        z = torch.from_numpy(z).float()
        with torch.no_grad():
            x_hat = self.T_inv(z)

        error = torch.abs(x-x_hat)

        return x, x_hat, t, error

    def simulate(self, a, b, N, ic, mean, std, add_noise = False):
        """
        Online simulation of observer for autonomous systems by the following steps:
        1. Generate y data from system.
        2. Use the y data to simulate observer dynamics:
            z_dot = Mz(t) + Ky(t)
        3. Use T_inv, the inverse of T to simulate:
            x_hat = T_inv(t, z)

        """
        x, y, t = self.system.generate_data(ic, a, b, N)
        x = torch.from_numpy(np.reshape(x, (N+1,self.system.x_size)))
        if add_noise:
            noise = np.random.normal(0, 0.1, (y.shape[0], y.shape[1]))    # Adding Noise
            y = y + noise

        z = data.KKL_observer_data(self.z_system.M, self.z_system.K, y, a, b, N)
        z = torch.from_numpy(z).view(N+1,self.system.z_size).float()
        #z = ((z - mean) / std).float()      # Normalize input
        with torch.no_grad():
            x_hat = self.T_inv(z)

        error = torch.abs(x-x_hat)

        return x, x_hat, t, error
    
    def get_average_error(self, a, b, N, ic_samples, add_noise = False):
        error = 0
        for idx, ic in enumerate(ic_samples):
            sim = self.simulate(a, b, N, ic, 0,0, add_noise=add_noise)
            error += sim[3]
        error = error / idx
        time = sim[2]
        return error, time
