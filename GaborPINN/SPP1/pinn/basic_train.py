from utils.encoding import get_embedder
from utils.utils import seed_torch
from model_zoo import build_model
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from SPP import analytical, get_epsilon, pde_residual_coeff, boundary_condition_left, boundary_condition_right


def compute_derivatives(net, embedding_fn, x):
    """Compute u, u_x, u_xx for 1D BVP."""
    t_dummy = torch.zeros_like(x)
    s_dummy = torch.zeros_like(x)
    xts = torch.cat([x, t_dummy, s_dummy], dim=1)  # [x, t=0, s=0]
    xts.requires_grad_(True)
    
    z = embedding_fn(xts) if embedding_fn is not None else xts
    u = net(z)
    
    grads = torch.autograd.grad(u, xts, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]  # du/dx
    
    u_xx = torch.autograd.grad(u_x, xts, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]  # d²u/dx²
    
    return u, u_x, u_xx


def prepare_input(x, embedding_fn, device):
    """Prepare input tensor (x, t=0, s=0) and apply embedding."""
    xts = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1).to(device)
    return embedding_fn(xts) if embedding_fn is not None else xts


def evaluate(cfg, net, embedding_fn):
    """Evaluate model on spatial domain [0,1]."""
    device = torch.device(cfg.pinn.device)
    net = net.double()
    net.eval()
    
    nx = cfg.pinn.nx
    x_lin = torch.linspace(0.0, 1.0, nx, dtype=torch.float64, device=device).view(-1, 1)
    z = prepare_input(x_lin, embedding_fn, device)
    
    with torch.no_grad():
        u_pred = net(z)
    
    u_true = analytical(x_lin, device)
    rel_l2 = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    
    print(f"Relative L2 error: {rel_l2:.4e}")
    
    # Plot comparison
    plt.figure()
    plt.plot(x_lin.cpu().numpy(), u_true.cpu().numpy(), label='Exact')
    plt.plot(x_lin.cpu().numpy(), u_pred.cpu().numpy(), '--', label='Predicted')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('1D BVP Solution')
    os.makedirs(cfg.visualize_path, exist_ok=True)
    plt.savefig(os.path.join(cfg.visualize_path, "bvp1d_solution.png"), dpi=150)
    plt.close()
    
    return rel_l2


def train(cfg, net, embedding_fn):
    """Train PINN for singularly perturbed BVP."""
    device = torch.device(cfg.pinn.device)
    seed_torch(cfg.pinn.seed)
    net = net.to(device).double()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.pinn.step_size, gamma=0.9)
    sw = SummaryWriter(cfg.tensorboard_path)
    
    N_int, N_bc = 10000, 2000
    min_loss = 1e20
    start_time = time.time()
    
    # Prepare evaluation grid
    nx = cfg.pinn.nx
    x_eval = torch.linspace(0.0, 1.0, nx, dtype=torch.float64, device=device).view(-1, 1)
    z_eval = prepare_input(x_eval, embedding_fn, device)
    u_true_eval = analytical(x_eval, device)
    
    for epoch in range(cfg.pinn.epochs):
        net.train()
        
        # Interior: PDE residual ε*u_xx + (1+ε)*u_x + u = 0
        x_int = torch.rand(N_int, 1, dtype=torch.float64, device=device)
        u_int, u_x, u_xx = compute_derivatives(net, embedding_fn, x_int)
        
        e = get_epsilon(device)
        coeff = pde_residual_coeff(x_int, device)
        res_pde = e * u_xx + coeff * u_x + u_int
        loss_pde = torch.mean(res_pde**2)
        
        # Boundary conditions: u(0) = 0, u(1) = 1
        x_bc_left = torch.zeros(N_bc // 2, 1, dtype=torch.float64, device=device)
        x_bc_right = torch.ones(N_bc // 2, 1, dtype=torch.float64, device=device)
        
        z_bc_left = prepare_input(x_bc_left, embedding_fn, device)
        z_bc_right = prepare_input(x_bc_right, embedding_fn, device)
        
        u_bc_left = net(z_bc_left)
        u_bc_right = net(z_bc_right)
        
        u_bc_left_true = boundary_condition_left(device).expand_as(u_bc_left)
        u_bc_right_true = boundary_condition_right(device).expand_as(u_bc_right)
        
        loss_bc_left = torch.mean((u_bc_left - u_bc_left_true)**2)
        loss_bc_right = torch.mean((u_bc_right - u_bc_right_true)**2)
        loss_bc = loss_bc_left + loss_bc_right
        
        loss = loss_pde + loss_bc  
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Periodic evaluation
        if epoch % cfg.pinn.print_loss_every == 0:
            net.eval()
            with torch.no_grad():
                u_pred_eval = net(z_eval)
                rel_l2 = (torch.norm(u_pred_eval - u_true_eval) / torch.norm(u_true_eval)).item()
            
            sw.add_scalar('train/total', loss.item(), epoch)
            sw.add_scalar('train/pde', loss_pde.item(), epoch)
            sw.add_scalar('train/bc', loss_bc.item(), epoch)
            sw.add_scalar('train/rel_l2', rel_l2, epoch)
            
            print(f"Epoch {epoch}: loss={loss.item():.3e}, pde={loss_pde.item():.3e}, "
                  f"bc={loss_bc.item():.3e}, L2={rel_l2:.3e}")
            
            if loss.item() < min_loss:
                min_loss = loss.item()
                os.makedirs(cfg.checkpoint_path, exist_ok=True)
                torch.save({'net': net.state_dict()}, 
                          os.path.join(cfg.checkpoint_path, 'best_net.pth'))
    
    print(f"Training time: {(time.time() - start_time) / 60:.2f} min")


def main(cfg):
    """Main entry point."""
    embedding_fn, input_cha = get_embedder(cfg.pinn.encoding_config)
    
    net = build_model(
        cfg.pinn.model_type, cfg.net_name, input_cha,
        cfg.pinn.out_channels, hidden_layers=cfg.pinn.hidden_layers,
        input_scale=cfg.pinn.scale, n_layers=len(cfg.pinn.hidden_layers)
    )
    
    macs, params = get_model_complexity_info(net, (input_cha,), as_strings=True,
                                             print_per_layer_stat=False)
    print(f"MACs: {macs}, Params: {params}")
    
    if cfg.task_type == 'train':
        train(cfg, net, embedding_fn)
        
        # Load best checkpoint and evaluate
        device = torch.device(cfg.pinn.device)
        ckpt_path = os.path.join(cfg.checkpoint_path, 'best_net.pth')
        if os.path.exists(ckpt_path):
            net.load_state_dict(torch.load(ckpt_path, map_location=device)['net'])
            net.to(device)
            rel_l2 = evaluate(cfg, net, embedding_fn)
            print(f"Final L2 error: {rel_l2:.4e}")
    else:
        # Test mode
        state = torch.load(cfg.pinn.state_dict_file)
        net.load_state_dict(state['net'])
        net.to(cfg.pinn.device)
        evaluate(cfg, net, embedding_fn)


def pinn_ent(cfg):
    """Legacy wrapper for compatibility."""
    main(cfg)
