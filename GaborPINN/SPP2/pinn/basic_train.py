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
from SPP import analytical, right_side, initial_condition, get_epsilon, initial_derivative


def compute_derivatives(net, embedding_fn, t):
    """Compute u, u_t, u_tt for 1D ODE."""
    x_dummy = torch.zeros_like(t)
    s_dummy = torch.zeros_like(t)
    xts = torch.cat([t, x_dummy, s_dummy], dim=1)
    xts.requires_grad_(True)
    
    z = embedding_fn(xts) if embedding_fn is not None else xts
    u = net(z)
    
    grads = torch.autograd.grad(u, xts, torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 0:1]
    
    u_tt = torch.autograd.grad(u_t, xts, torch.ones_like(u_t), create_graph=True)[0][:, 0:1]
    
    return u, u_t, u_tt


def prepare_input(t, embedding_fn, device):
    """Prepare input tensor (t, x=0, s=0) and apply embedding."""
    xts = torch.cat([t, torch.zeros_like(t), torch.zeros_like(t)], dim=1).to(device)
    return embedding_fn(xts) if embedding_fn is not None else xts


def evaluate(cfg, net, embedding_fn):
    """Evaluate model on time domain [0,1]."""
    device = torch.device(cfg.pinn.device)
    net = net.double()
    net.eval()
    
    nt = cfg.pinn.nx
    t_lin = torch.linspace(0.0, 1.0, nt, dtype=torch.float64, device=device).view(-1, 1)
    z = prepare_input(t_lin, embedding_fn, device)
    
    with torch.no_grad():
        u_pred = net(z)
    
    u_true = analytical(t_lin, device)
    rel_l2 = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    
    print(f"Relative L2 error: {rel_l2:.4e}")
    
    # Plot comparison
    plt.figure()
    plt.plot(t_lin.cpu().numpy(), u_true.cpu().numpy(), label='Exact')
    plt.plot(t_lin.cpu().numpy(), u_pred.cpu().numpy(), '--', label='Predicted')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('u')
    plt.title('1D ODE Solution')
    os.makedirs(cfg.visualize_path, exist_ok=True)
    plt.savefig(os.path.join(cfg.visualize_path, "ode1d_solution.png"), dpi=150)
    plt.close()
    
    return rel_l2


def train(cfg, net, embedding_fn):
    """Train PINN for stiff nonlinear ODE."""
    device = torch.device(cfg.pinn.device)
    seed_torch(cfg.pinn.seed)
    net = net.to(device).double()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.pinn.step_size, gamma=0.9)
    sw = SummaryWriter(cfg.tensorboard_path)
    
    N_int, N_ic = 1000, 500
    min_loss = 1e20
    start_time = time.time()
    
    # Prepare evaluation grid
    nt = cfg.pinn.nx
    t_eval = torch.linspace(0.0, 1.0, nt, dtype=torch.float64, device=device).view(-1, 1)
    z_eval = prepare_input(t_eval, embedding_fn, device)
    u_true_eval = analytical(t_eval, device)
    
    for epoch in range(cfg.pinn.epochs):
        net.train()
        
        # Interior: PDE residual ε*u_tt + (3+t)*u_t + u² - sin(u) = f(t)
        t_int = torch.rand(N_int, 1, dtype=torch.float64, device=device)
        u_int, u_t, u_tt = compute_derivatives(net, embedding_fn, t_int)
        f_int = right_side(t_int, u_int, device)
        e = get_epsilon(device)
        res_pde = e * u_tt + (3 + t_int) * u_t + u_int**2 - torch.sin(u_int) - f_int
        loss_pde = torch.mean(res_pde**2)
        
        # Initial conditions: u(0) and u'(0)
        t_ic = torch.zeros(N_ic, 1, dtype=torch.float64, device=device)
        u_ic, u_t_ic, _ = compute_derivatives(net, embedding_fn, t_ic)
        
        u_ic_true = initial_condition(device).expand_as(u_ic)
        loss_u_ic = torch.mean((u_ic - u_ic_true)**2)
        
        u_t_ic_true = initial_derivative(device).expand_as(u_t_ic)
        loss_ut_ic = torch.mean((u_t_ic - u_t_ic_true)**2)
        
        loss = loss_pde + loss_u_ic + loss_ut_ic
        
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
            sw.add_scalar('train/u_ic', loss_u_ic.item(), epoch)
            sw.add_scalar('train/ut_ic', loss_ut_ic.item(), epoch)
            sw.add_scalar('train/rel_l2', rel_l2, epoch)
            
            print(f"Epoch {epoch}: loss={loss.item():.3e}, pde={loss_pde.item():.3e}, "
                  f"u_ic={loss_u_ic.item():.3e}, ut_ic={loss_ut_ic.item():.3e}, L2={rel_l2:.3e}")
            
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
