import torch
import Systems
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.io as spio
from Models1 import *
import Observer
from Observer import System_z, Observer

mat = spio.loadmat('/Applications/Programming/Machine Learning/DF Internship/MK_data/Duffing_ML_dim5', squeeze_me=True)
M = mat['M']
K = np.expand_dims(mat['L'], axis = 0).T

limits = np.array([[-1,1], [-1,1]])    # Sample space

model1 = RevDuff['M dim 5']       # Dim 5

sys = Systems.RevDuff(limits, add_noise=False)
z_sys = System_z(M, K, sys)
observer1 = Observer(sys, z_sys, model1)
a = 0
b = 50
#ic = np.random.rand(1,2)
ic = np.array([[0.6, 0.9]])

x1, x1_hat, t1, error1 = observer1.simulate(a, b, 5000, ic, 0,0, add_noise=False)

# --------------------- Animation --------------------- 
x11_hat = x1_hat[:,0].numpy()
x2_hat = x1_hat[:,1].numpy()
x11 = x1[:,0].numpy()
x2 = x1[:,1].numpy()

x1_est = []
x2_est = []
x1_truth = []
x2_truth = []

fig, ax = plt.subplots()
line1, = ax.plot([], [])
line2, = ax.plot([], [])
lim = 1.35
step = 5

def init():
	ax.set_xlim(-lim, lim)
	ax.set_ylim(-lim,lim)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend([line1, line2], ['Estimate', 'Ground Truth'])	
	plt.title('Reverse Duffing Oscillator')

def animation_frame(i):
	x1_est.append(x11_hat[i*step])
	x2_est.append(x2_hat[i*step])
	line1.set_xdata(x1_est)
	line1.set_ydata(x2_est)
	x1_truth.append(x11[i*step])
	x2_truth.append(x2[i*step])
	line2.set_xdata(x1_truth)
	line2.set_ydata(x2_truth)
	return line1, line2 

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 200, 1), init_func = init, interval=10, repeat = False)
#plt.show()

path = '/Applications/Programming/Machine Learning/DF Internship/Plots/Animations/test.gif'
writer = PillowWriter(fps=30)
animation.save(path, writer = writer)
