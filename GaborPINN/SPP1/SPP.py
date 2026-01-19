import torch

def get_epsilon(device):
    return torch.tensor(2**-10, dtype=torch.float64, device=device)

def analytical(x, device):
    """Exact solution: u(x) = (exp(-x) - exp(-x/ε)) / (exp(-1) - exp(-1/ε))"""
    e = get_epsilon(device)
    numerator = torch.exp(-x) - torch.exp(-x / e)
    denominator = torch.exp(torch.tensor(-1.0, device=device)) - torch.exp(-1.0 / e)
    return numerator / denominator

def pde_residual_coeff(x, device):
    """Returns coefficient (1 + ε) for the first derivative term."""
    e = get_epsilon(device)
    return 1.0 + e

def boundary_condition_left(device):
    """u(0) = 0"""
    return torch.tensor(0.0, dtype=torch.float64, device=device)

def boundary_condition_right(device):
    """u(1) = 1"""
    return torch.tensor(1.0, dtype=torch.float64, device=device)
