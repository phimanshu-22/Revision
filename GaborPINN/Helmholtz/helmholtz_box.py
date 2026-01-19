import torch
import math

b1 = 1.0
b2 = 8.0
c  = 1.0

def exact_u(x, y):
    # x,y: (N,1)
    return torch.sin(b1 * math.pi * x) * torch.sin(b2 * math.pi * y)

def rhs_f(x, y):
    coeff = - (math.pi ** 2) * (b1 ** 2 + b2 ** 2) + c ** 2
    return coeff * exact_u(x, y)

def on_boundary_mask(x, y):
    # |x|=1 or |y|=1 (with small tolerance)
    eps = 1e-6
    return (torch.abs(x - 1.0) < eps) | (torch.abs(x + 1.0) < eps) | \
           (torch.abs(y - 1.0) < eps) | (torch.abs(y + 1.0) < eps)
