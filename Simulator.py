import torch
import numpy as np
import data_generation as data
from torch.autograd.functional import jacobian

def simulate(system, model, a, b, N, M, K, ic, mean = 0, std = 1, add_noise = False):
    x, y, t = system.generate_data(ic, a, b, N)
    print(y)
    x = torch.from_numpy(np.reshape(x, (N+1,system.x_size)))
    if add_noise == True:
        noise = np.random.normal(0, 1, (y.shape[0], y.shape[1]))    # Adding Noise
        y = y + noise
        
    z = data.KKL_observer_data(M, K, y, a, b, N)
    z = torch.from_numpy(z).view(N+1,system.z_size).float()
    #z = ((z - mean) / std).float()      # Normalize input
    with torch.no_grad():
        x_hat = model.net2(z)

    error = torch.abs(x-x_hat)

    return x, x_hat, t, error

def simulate_NA(system, ic, M, K, a, b, N, u0, T, T_inv, g):
    x, y, t = system.generate_data(ic, a, b, N)     # Generate y data

    u = system.input
    size_z = M.shape[0]
    h = (b-a) / N

    z = [[0]*size_z]
    v = np.array(z).T

    y = np.squeeze(y)
    if y.ndim > 2:
        y = y[1:, :]
        f = lambda y,q,z: np.matmul(M,z) + np.matmul(K, np.expand_dims(y,1)) + q
    else:
        f = lambda y,q,z: np.matmul(M,z) + K*y + q
        y = np.delete(y,0) 

    for idx, output in enumerate(y):
        with torch.no_grad():
            x_hat = T_inv(torch.tensor(z[-1]).float())
        u_sub_u0 = u(t[idx]) - u0(t[idx])
        dTdx = jacobian(T, x_hat).numpy()
        dTdx_mul_g = np.matmul(dTdx, g)

        q = dTdx_mul_g*u_sub_u0     # phi*(u(t) - u0(t))

        k1 = f(output,q, v)
        k2 = f(output,q, v + h/2*k1)
        k3 = f(output,q, v + h/2*k2)
        k4 = f(output,q, v + h*k3)
        
        v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

        a = np.reshape(v.T, size_z)
        z.append(np.ndarray.tolist(a)) 

    z = np.array(z)
    z = torch.from_numpy(z).float()
    with torch.no_grad():
        x_hat = T_inv(z)

    return x, x_hat, t      