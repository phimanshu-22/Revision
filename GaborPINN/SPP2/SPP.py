import torch

def get_epsilon(device):
    return torch.tensor(2**-10, dtype=torch.float64, device=device)

def analytical(t, device):
    e = get_epsilon(device)
    return t**2 + 2 - torch.exp(-t/e)

def right_side(t, u, device):
    e = get_epsilon(device)
    expo = torch.exp(-t/e)
    res = (2 - expo + t**2)**2 + (3 + t)*(2*t + expo/e) \
          + e*(2 - expo/e**2) - torch.sin(2 - expo + t**2)
    return res

def initial_condition(device):
    t0 = torch.tensor(0.0, dtype=torch.float64, device=device)
    return analytical(t0, device)

def initial_derivative(device):
    e = get_epsilon(device)
    return 1.0 / e
