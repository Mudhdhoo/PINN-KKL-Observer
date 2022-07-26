import torch
import matplotlib.pyplot as plt
import numpy as np
from smt.sampling_methods import LHS

class System:
    def __init__(self, function, output, sample_space):
        self.function = function
        self.output = output
        self.sample_space = sample_space
        
    # LHS Sampling
    def sample_ic(self,samples):
        return LHS(xlimits = self.sample_space, random_state = 0)(samples)
                     
    def simulate(self,a, b, N, v):
        x,t = RK4(self.function, a, b, N, v, self.input)
        return np.array(x), t
    
    def generate_data(self, ic, a, b, N):
        data = []
        output = []
        for i in range(0, np.size(ic, axis = 0)):
            x, t = self.simulate(a,b,N,ic[i])
            temp = []
            for j in x:
                temp.append(self.output(j))
            data.append(x)    
            output.append(np.array(temp))
           # plt.plot(x[:,0], x[:,1])
       # plt.show()
      
        return np.array(data), np.array(output), t   

# Runge-Kutta 4
def RK4(f, a, b, N, v, inputs):
    h = (b-a) / N
    x = [v]
    t = [a]
    u = 0
        
    for i in range(0,N):
        if inputs != None:
            #u = np.array([inputs(t[-1])])
            u = np.array(inputs(t[-1]))
        k1 = f(u, v)
        k2 = f(u, v + h/2*k1)
        k3 = f(u, v + h/2*k2)
        k4 = f(u, v + h*k3)
        
        v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x.append(np.ndarray.tolist(v)) 
        
        time = t[-1] + h
        t.append(time)
                 
    return x,np.array(t)

def KKL_observer_data(M, K, y, a, b, N):
    scalar_y = False
    data = []
    size_z = M.shape[0]
    h = (b-a) / N
    
    #Check if y is scalar or vector
    if y.ndim > 2:                                    # Reshape y from (m,) --> (m, 1) for matrix multiplication
        f = lambda y,z: np.matmul(M,z) + np.matmul(K, np.expand_dims(y,1))  
    else:
        f = lambda y,z: np.matmul(M,z) + K*y 
        scalar_y = True
        
    for output in y:
        x = [[0]*size_z]
        v = np.array(x).T
        if scalar_y == True:
            truncated_output = np.delete(output,0)    # Ignore the first output value as we already have the initial conditions
        else:
            truncated_output = output[1:, :]
        for i in truncated_output:
            k1 = f(i, v)
            k2 = f(i, v + h/2*k1)
            k3 = f(i, v + h/2*k2)
            k4 = f(i, v + h*k3)
        
            v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

            a = np.reshape(v.T, size_z)
            x.append(np.ndarray.tolist(a)) 
        data.append(np.array(x))

    return np.array(data)



             