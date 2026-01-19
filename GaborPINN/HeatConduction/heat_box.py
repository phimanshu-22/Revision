# heat_box.py
import torch
import math

e = 0.15

def analytical(x, t):
    et = 2*t - 1
    ex = torch.exp(1.0 / (et**2 + e))
    return (1 - x**2) * ex

def right_side(x, t):
    et = 2*t - 1
    ex = torch.exp(1.0 / (et**2 + e))
    return 2 * ex * (1 + 2*et*(x**2 - 1) / (et**2 + e)**2)
