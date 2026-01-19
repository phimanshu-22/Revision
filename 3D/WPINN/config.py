import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from scipy.stats import qmc

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Global device configuration
global device, device1
device = torch.device('cuda:0')
# device1 = torch.device('cuda:0')
torch.manual_seed(101)

torch.set_default_dtype(torch.float32)

class DataConfig:
    def __init__(self):
        # Sample sizes
        self.n_collocation = 10000
        self.n_validation = 10000
        self.n_boundary = 2000
        self.n_test = 20
        
        # Domain bounds
        self.x_lower = 0
        self.x_upper = 1
        self.y_lower = 0
        self.y_upper = 1
        self.z_lower = 0
        self.z_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""
        # Collocation points

        # sampler = qmc.Sobol(d = 2, scramble = True, seed = 501)
        # sobol_sequence_collocation = sampler.random(n = self.n_collocation)
        # sobol_sequence_boundary = sampler.random(n = self.n_boundary)

        # x_collocation = torch.tensor(sobol_sequence_collocation[:,0].flatten()*(self.x_upper - self.x_lower) + self.x_lower).float().to(device)
        # y_collocation = torch.tensor(sobol_sequence_collocation[:,1].flatten()*(self.y_upper - self.y_lower) + self.y_lower).float().to(device)

        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_collocation = (torch.rand(self.n_collocation) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        z_collocation = (torch.rand(self.n_collocation) * (self.z_upper - self.z_lower) + self.z_lower).to(self.device)

        x_bc = (torch.rand(self.n_boundary) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_bc = (torch.rand(self.n_boundary) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        z_bc = (torch.rand(self.n_boundary) * (self.z_upper - self.z_lower) + self.z_lower).to(self.device)
        
        x_bc_l = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_u = self.x_upper * torch.ones(self.n_boundary).to(self.device)
        y_bc_l = self.y_lower * torch.ones(self.n_boundary).to(self.device)
        y_bc_u = self.y_upper * torch.ones(self.n_boundary).to(self.device)
        z_bc_l = self.z_lower * torch.ones(self.n_boundary).to(self.device)
        z_bc_u = self.z_upper * torch.ones(self.n_boundary).to(self.device)
        

        # Validation points
        x_validation = (torch.rand(self.n_validation) * (self.x_upper - self.x_lower) + self.x_lower)
        y_validation = (torch.rand(self.n_validation) * (self.y_upper - self.y_lower) + self.y_lower)
        z_validation = (torch.rand(self.n_validation) * (self.z_upper - self.z_lower) + self.z_lower)

        # Testing and Plotting points
        # xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        # ytest = torch.linspace(self.y_lower, self.y_upper, self.n_test)
            
        # x_grid, y_grid = torch.meshgrid(xtest, ytest)
        # x_test = x_grid.reshape(-1)
        # y_test = y_grid.reshape(-1)
        
        return {
            'domain': (self.x_lower, self.x_upper, self.y_lower, self.y_upper, self.z_lower, self.z_upper),  
            'collocation': (self.n_collocation, x_collocation, y_collocation, z_collocation),
            'validation': (x_validation, y_validation, z_validation),
            'boundary': (x_bc, y_bc, z_bc, x_bc_l, x_bc_u, y_bc_l, y_bc_u, z_bc_l, z_bc_u),
            # 'test': (self.n_test, x_test, y_test)
        }
    

config = DataConfig()
points = config.generate_training_points()

# Access the points as needed
x_lower, x_upper, y_lower, y_upper, z_lower, z_upper = points['domain']
n_collocation, x_collocation, y_collocation, z_collocation = points['collocation']
x_validation, y_validation, z_validation = points['validation']
x_bc, y_bc, z_bc, x_bc_l, x_bc_u, y_bc_l, y_bc_u, z_bc_l, z_bc_u = points['boundary']
# n_test, x_test, y_test = points['test']
