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
    limits_normal = np.array([[-1, 1], [-1, 1]])    # Sample space for normal datapoints
    a = 0   # start
    b = 50  # end
    N = 1000          # Number of intervals for RK4
    num_ic = 50       # Number of initial conditions to be sampled

    #Load (A, B) from your directory
    mat = spio.loadmat('/Applications/Programming/Machine Learning/DF Internship/MK_data/Duffing_ML_dim5', squeeze_me=True)
    A = mat['M']
    B = np.expand_dims(mat['L'], axis = 0).T

    revduff = Systems.RevDuff(5, add_noise=False)
    dataset = DataSet(revduff, A, B, a, b, N, num_ic, limits_normal, PINN_sample_mode = 'split traj', data_gen_mode = 'backward sim')
    print('Dataset sucessfully generated.', '\n')

    # --------------------- Training Setup ---------------------
    torch.manual_seed(9)

    x_size = dataset.system.x_size
    z_size = dataset.system.z_size
    num_hidden = 3
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
    trainer.train(with_pde = True)
    File = '/Applications/Programming/Machine Learning/DF Internship/Experiments/Error_duffing/revduff'  
    torch.save(main_net, File)

    print('Training complete.', '\n')

    # --------------------- Generate Data ---------------------

    print('Running simulations.', '\n')
    np.random.seed(888)
    revduff.toggle_noise()
    z_sys = System_z(A, B, revduff)
    observer = Observer(revduff, z_sys, main_net)
    ic_samples = np.random.uniform(-1, 1, (100,1,2))
    errors, _, t = observer.get_average_error(a, b, 1000, ic_samples)

    mdict = {'errors': errors,
             'time': t}
    spio.savemat('/Applications/Programming/Machine Learning/DF Internship/Experiments/duffing_error.mat',mdict)
    print('Simulations complete, data saved to directory.')
    
if __name__ =='__main__':
    main()


