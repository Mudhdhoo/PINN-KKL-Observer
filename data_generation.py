import numpy as np
        
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



             
