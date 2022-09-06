import torch
from torch import nn
import Systems
import numpy as np
from NN import Main_Network
from Dataset import DataSet
from Normalizer import Normalizer
from Trainer import Trainer
import scipy.io as spio
from torch.optim.lr_scheduler import ReduceLROnPlateau
import Observer
from Observer import System_z, Observer
import torch.nn.functional as F

def main():
    # --------------------- System Setup --------------------- 

    print('Generating Data.', '\n')
    a0 = 0.2
    b0 = 0.2
    c0 = 5.7
    limits = limits = np.array([[-1,1], [-1,1], [-1,1]])    # Sample space
    rossler = Systems.Rossler(a0, b0, c0)

    a = 0   # start
    b = 50  # end
    N = 1000        # Number of intervals for RK4
    num_ic = 50

    mat = spio.loadmat('/Applications/Programming/Machine Learning/DF Internship/MK_data/Chua_ML', squeeze_me=True)
    A = mat['M']
    B = np.expand_dims(mat['L'], axis = 0).T

    dataset = DataSet(rossler, A, B, a, b, N, num_ic, limits, PINN_sample_mode = 'split traj', data_gen_mode = 'negative forward')
    print('Dataset sucessfully generated.', '\n')

    # --------------------- Training Setup ---------------------
    torch.manual_seed(9)

    x_size = dataset.system.x_size
    z_size = dataset.system.z_size
    num_hidden = 5
    hidden_size = 50
    activation = F.relu
    normalizer = Normalizer(dataset)
    main_net = Main_Network(x_size, z_size, num_hidden, hidden_size, activation, normalizer)      

    epochs = 15
    learning_rate = 0.001
    batch_size = 32
    optimizer = torch.optim.Adam(main_net.parameters(), lr = learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 1, threshold = 0.0001, verbose = True)
    loss_fn = nn.MSELoss(reduction = 'mean')

    trainer = Trainer(dataset, epochs, optimizer, main_net, loss_fn, batch_size, scheduler = scheduler)
    trainer.train(with_pde = True)      # Choose with or without PDE
    File = '/Applications/Programming/Machine Learning/DF Internship/Experiments/Error_rossler/rossler_model'  
    torch.save(main_net, File)

    print('Training complete.', '\n')

    # --------------------- Generate Data ---------------------

    print('Running simulations.', '\n')
    np.random.seed(888)
    #rossler.toggle_noise()
    z_sys = System_z(A, B, rossler)
    observer = Observer(rossler, z_sys, main_net)

    ic = 100       # Number trajectories
    ic_samples_in = np.random.uniform(-1, 1, (ic,1,3))
    ic_samples_out = np.concatenate((np.random.uniform(1, 1.5, (int(ic/2),1,3)), np.random.uniform(-1, -1.5, (int(ic/2),1,3)))).reshape(ic, 3)
    np.random.shuffle(ic_samples_out[:,0])
    np.random.shuffle(ic_samples_out[:,1])
    np.random.shuffle(ic_samples_out[:,2])
    ic_samples_out = np.expand_dims(ic_samples_out, axis = 1)

    x_in, x_hat_in, _, _, t = observer.sim_multi(a, b, 5000, ic_samples_in)
    x_out, x_hat_out, _, _, t = observer.sim_multi(a, b, 5000, ic_samples_out)

    mdict = {'x_in':x_in, 
             'x_hat_in':x_hat_in,
             'x_out':x_out,
             'x_hat_out':x_hat_out,
             'time': t}

    spio.savemat('/Applications/Programming/Machine Learning/DF Internship/Experiments/duffing_error.mat',mdict)
    print('Simulations complete, data saved to directory.')
    
if __name__ =='__main__':
    main()


