import numpy as np
from data_generation import System

"""
Systems are implemented by defining 6 essential parameters:

function: The system function describing its dynamics.

output: The measureable outputs of the system.

input: Input to the system for non-autonomous systems. If the system is autonomous, input is None.

x_size: Dimension of the system.

y_size: Dimension of the output.

z_size: Dimension of the transformed system.

"""

# Reverse Duffing Oscillator
class RevDuff(System):
    def __init__(self, sample_space):
        self.y_size = 1
        self.x_size = 2
        self.z_size = self.y_size*(self.x_size + 1)
        self.input = None
        super().__init__(self.function, self.output, sample_space)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
    
        x1_dot = x2**3
        x2_dot = -x1
    
        return np.array([x1_dot, x2_dot])
    
    def output(self, x):
        y = x[0]
        
        return y
        
# Network SIS
class SIS(System):
    def __init__(self, sample_space, A, B, G, C):
        self.A = A
        self.B = B
        self.G = G
        self.C = C
        self.x_size = self.A.shape[0]
        self.y_size = self.C.shape[0]
        self.z_size = self.y_size*(self.x_size + 1)
        self.function = lambda u, x: (B@A - G)@x - np.diag(x)@B@A@x    # x = np.array([a, b, c,....]])
        self.output = lambda x: C@x
        self.input = None
        super().__init__(self.function, self.output, sample_space)
        
# Van der Pol Oscillator
class VdP(System):
    def __init__(self, sample_space, my = 3):
        self.x_size = 2
        self.y_size = 1
        self.z_size = self.y_size*(self.x_size + 1)
        self.my = my
        self.input = None
        super().__init__(self.function, self.output, sample_space)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
            
        x1_dot = x2
        x2_dot = self.my*(1 - x1**2)*x2 - x1
            
        return np.array([x1_dot, x2_dot])
        
    def output(self, x):
        y = x[0]
        return y
        
# Polynomial system
class Polynomial(System):
    def __init__(self, sample_space):
        self.x_size = 2
        self.y_size = 1
        self.z_size = self.y_size*(self.x_size + 1)
        self.input = None
        super().__init__(self.function, self.output, sample_space)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
        
        x1_dot = x1 - (1/3)*x1**3 - x1*x2**2
        x2_dot = x1 - x2 - (1/3)*x2**3 - x2*x1**2
        
        return np.array([x1_dot, x2_dot])
    
    def output(self, x):
        y = x[0]
            
        return y
        
# Robot Arm
class RobotArm(System):
    def __init__(self, sample_space, A, B, C, f, u):
        self.x_size = 4
        self.y_size = 2
        self.z_size = self.y_size*(self.x_size + 1)      
        self.function = lambda u, x: A@x + f(x) + B@u
        self.output = lambda x: C@x
        self.input = u
        super().__init__(self.function, self.output, sample_space)        
        
# Chua's Circuit
class Chua(System):
    def __init__(self, sample_space, alpha, beta, gamma, a, b):
        self.x_size = 3
        self.y_size = 1
        self.z_size = self.y_size*(self.x_size + 1)
        self.g = lambda x: 0.5*(a - b)*(np.abs(x[0] + 1) - np.abs(x[0] - 1))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b
        self.input = None
        super().__init__(self.function, self.output, sample_space)  
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        
        x1_dot = self.alpha*(x2 - x1*(1 + self.b) - self.g(x))
        x2_dot = x1 - x2 + x3
        x3_dot = -self.beta*x2 - self.gamma*x3
        
        return np.array([x1_dot, x2_dot, x3_dot])
    
    def output(self, x):
        y = x[2]
        return y
    
# Rössler's System
class Rossler(System):
    def __init__(self, sample_space, a, b, c):
        self.x_size = 3
        self.y_size = 1
        self.z_size = self.y_size*(self.x_size + 1)
        self.input = None
        self.a = a
        self.b = b
        self.c = c
        super().__init__(self.function, self.output, sample_space)  
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1] 
        x3 = x[2]
        
        x1_dot = -(x2 + x3)
        x2_dot = x1 + self.a*x2
        x3_dot = self.b + x3*(x1 - self.c)
        
        return np.array([x1_dot, x2_dot, x3_dot])
    
    def output(self, x):
        y = x[0]
        return y

# SIR
class SIR(System):
    def __init__(self, sample_space, beta, gamma, N):
        self.x_size = 3
        self.y_size = 2
        self.z_size = self.y_size*(self.x_size + 1)
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.input = None
        super().__init__(self.function, self.output, sample_space)  
      
    def function(self, u, x):
        S = x[0]
        I = x[1]
        R = x[2]
        
        S_dot = -self.beta*I*S/self.N
        I_dot = self.beta*I*S/self.N - self.gamma*I
        R_dot = self.gamma*I
        
        return np.array([S_dot, I_dot, R_dot])
    
    def output(self, x):
        S = x[0]
        I = x[1]
        R = x[2]
        
        y = np.array([R, S+I+R])

        return y
    
# Non-Autonomous Reverse Duffing Oscillator
class RevDuff_NA(System):
    def __init__(self, sample_space, input):
        self.y_size = 1
        self.x_size = 2
        self.z_size = self.y_size*(self.x_size + 1)
        self.input = input
        super().__init__(self.function, self.output, sample_space)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
    
        x1_dot = x2**3
        x2_dot = -x1 + u
    
        return np.array([x1_dot, x2_dot])
    
    def output(self, x):
        y = x[0]
        
        return y

# Non-Autonomous Van der Pol Oscillator
class VdP_NA(System):
    def __init__(self, sample_space, input, my = 3):
        self.x_size = 2
        self.y_size = 1
        self.z_size = self.y_size*(self.x_size + 1)
        self.my = my
        self.input = input
        super().__init__(self.function, self.output, sample_space)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
        x1_dot = x2
        x2_dot = self.my*(1 - x1**2)*x2 - x1 + u
            
        return np.array([x1_dot, x2_dot])
        
    def output(self, x):
        y = x[0]
        return y

        
        
        
        
        
        
        
        
        
        
        
        
        