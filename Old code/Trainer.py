import torch
import Loss_Calculator as L
from Loss_functions import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, dataset, epochs, optimizer, net, loss_fn, batch_size, shuffle = True, scheduler = None, reduction = 'mean'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.trainset = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net.to(self.device)
        self.loss_calculator = L.Loss_Calculator(loss_fn, self.net, self.dataset, self.device)
        self.normalizer = net.normalizer
        self.reduction = reduction
        print('Device:', self.device)
        
    def train(self, with_pde = True):
        M = self.dataset.M
        K = self.dataset.K
        sys = self.dataset.system
        pde1 = PdeLoss_xz(M, K, sys, self.loss_calculator, self.reduction)
        pde2 = PdeLoss_zx(self.loss_calculator, self.reduction)
        MSE = MSELoss(self.loss_calculator)
        for epoch in range(self.epochs):
            print(self.optimizer.state_dict())
            loss_sum = 0
            for idx, data in enumerate(self.trainset):
                x, z, y = data
                x, z, y = x.to(self.device), z.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                z_hat, x_hat, norm_z_hat, norm_x_hat = self.net(x, z)
                if self.normalizer != None:
                    label_x = self.normalizer.Normalize(x).float()
                    label_z = self.normalizer.Normalize(z).float()
                else:
                    label_x = x
                    label_z = z

                #loss_normal = self.loss_calculator.calc_loss(norm_x_hat, norm_z_hat, label_x, label_z)      # MSE Loss      
                loss_normal = MSE(norm_x_hat, norm_z_hat, label_x, label_z)

                if with_pde:
                    #loss_pde1 = self.loss_calculator.calc_pde_loss_xz(x, y, z_hat, self.dataset.system, self.dataset.M, self.dataset.K)     # PDE Loss
                    #loss_pde2 = self.loss_calculator.calc_pde_loss_zx(x, z_hat)     # PDE Loss
                    loss_pde1 = pde1(x, y, z_hat)
                    loss_pde2 = pde2(x, z_hat)
                    loss = loss_normal + loss_pde1 + loss_pde2
                else:
                    loss = loss_normal
    
                loss_sum += loss
                loss.backward()
                self.optimizer.step()
                
            training_loss = (loss_sum / idx).item()
            
            if self.scheduler == None:
                pass
            else:
                self.scheduler.step(training_loss)
            
            print('Epoch:', epoch+1, 'Loss:', training_loss)    # Average loss per epoch
                
