import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
import data_generation as data
import numpy as np
import matplotlib.pyplot as plt
from NN import NN1, NN2
import training_data
import Loss_Calculator as L
from Main_NN import Main_Network

class Trainer:
    def __init__(self, dataset, epochs, optimizer, net, loss_fn, batch_size, shuffle = True, scheduler = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.trainset = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net.to(self.device)
        self.loss_calculator = L.Loss_Calculator(loss_fn, self.net, self.dataset, self.device)
        print('Device:', self.device)
        
    def train(self):
        for epoch in range(self.epochs):
            loss_sum = 0
            for idx, data in enumerate(self.trainset):
                x, z, y = data
                x, z, y = x.to(self.device), z.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                z_hat, x_hat = self.net.forward(x, z)
                loss_normal = self.loss_calculator.calc_loss(x_hat, z_hat, x, z)
                loss_pde1 = self.loss_calculator.calc_pde_loss_xz(x, y, z_hat, self.dataset.system, self.dataset.M, self.dataset.K)
                loss_pde2 = self.loss_calculator.calc_pde_loss_zx(x, z_hat)
                loss = loss_normal + loss_pde1 + loss_pde2
                #loss = loss_normal
                loss_sum += loss
                loss.backward()
                self.optimizer.step()
                
            validation_loss = (loss_sum / idx).item()
            
            if self.scheduler == None:
                pass
            else:
                self.scheduler.step(validation_loss)
            
            print('Epoch:', epoch+1, 'Loss:', validation_loss)    # Average loss per epoch
                