from utils.encoding import get_embedder
from utils.utils import seed_torch
from model_zoo import build_model
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from heat_box import analytical, right_side


def compute_derivatives(net, embedding_fn, x, t):
    """Compute u, u_t, u_xx for heat equation."""
    s = torch.zeros_like(x)
    xts = torch.cat([x, t, s], dim=1)
    xts.requires_grad_(True)
    
    z = embedding_fn(xts) if embedding_fn is not None else xts
    u = net(z)
    
    grads = torch.autograd.grad(u, xts, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    u_xx = torch.autograd.grad(u_x, xts, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    
    return u, u_t, u_xx


def prepare_input(x, t, embedding_fn, device):
    """Prepare input tensor (x, t, s=0) and apply embedding."""
    s = torch.zeros_like(x)
    xts = torch.cat([x, t, s], dim=1).to(device)
    return embedding_fn(xts) if embedding_fn is not None else xts


def evaluate(cfg, net, embedding_fn):
    """Evaluate model on full (x,t) domain."""
    device = torch.device(cfg.pinn.device)
    net.eval()
    
    nx, nt = cfg.pinn.nx, cfg.pinn.nz
    x_lin = torch.linspace(-1.0, 1.0, nx)
    t_lin = torch.linspace(0.0, 1.0, nt)
    X, T = torch.meshgrid(x_lin, t_lin, indexing='ij')
    
    x_flat = X.reshape(-1, 1).to(device)
    t_flat = T.reshape(-1, 1).to(device)
    z = prepare_input(x_flat, t_flat, embedding_fn, device)
    
    with torch.no_grad():
        u_pred = net(z).reshape(nx, nt)
    
    u_true = analytical(X.to(device), T.to(device))
    rel_l2 = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    
    print(f"Relative L2 error: {rel_l2:.4e}")
    
    # Plot error
    err = (u_pred - u_true).abs().cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.imshow(err.T, origin='lower', extent=[-1, 1, 0, 1], aspect='auto')
    plt.colorbar(label='|Error|')
    plt.xlabel('x'); plt.ylabel('t'); plt.title('Absolute Error')
    os.makedirs(cfg.visualize_path, exist_ok=True)
    plt.savefig(os.path.join(cfg.visualize_path, "heat_error.png"), dpi=150)
    plt.close()
    
    return rel_l2


def train(cfg, net, embedding_fn):
    """Train PINN for heat equation."""
    device = torch.device(cfg.pinn.device)
    seed_torch(cfg.pinn.seed)
    net = net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.pinn.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.pinn.step_size, gamma=cfg.pinn.gamma)
    sw = SummaryWriter(cfg.tensorboard_path)
    
    N_int, N_bc, N_ic = 20000, 2000, 2000
    min_loss = 1e20
    start_time = time.time()
    
    # Prepare evaluation grid
    nx, nt = cfg.pinn.nx, cfg.pinn.nz
    x_eval_lin = torch.linspace(-1.0, 1.0, nx)
    t_eval_lin = torch.linspace(0.0, 1.0, nt)
    X_eval, T_eval = torch.meshgrid(x_eval_lin, t_eval_lin, indexing='ij')
    x_eval = X_eval.reshape(-1, 1).to(device)
    t_eval = T_eval.reshape(-1, 1).to(device)
    z_eval = prepare_input(x_eval, t_eval, embedding_fn, device)
    u_true_eval = analytical(X_eval.to(device), T_eval.to(device))
    
    for epoch in range(cfg.pinn.epochs):
        net.train()
        
        # Interior: PDE residual
        x_int = (2 * torch.rand(N_int, 1) - 1.0).to(device)
        t_int = torch.rand(N_int, 1).to(device)
        u_int, u_t, u_xx = compute_derivatives(net, embedding_fn, x_int, t_int)
        f_int = right_side(x_int, t_int).to(device)
        loss_pde = torch.mean((u_t - u_xx - f_int) ** 2)
        
        # Boundary: x = ±1
        t_bc = torch.rand(N_bc, 1).to(device)
        x_bc = torch.cat([-torch.ones(N_bc // 2, 1), torch.ones(N_bc // 2, 1)]).to(device)
        t_bc = torch.cat([t_bc[:N_bc // 2], t_bc[:N_bc // 2]]).to(device)
        z_bc = prepare_input(x_bc, t_bc, embedding_fn, device)
        u_bc = net(z_bc)
        u_bc_true = analytical(x_bc, t_bc)
        loss_bc = torch.mean((u_bc - u_bc_true) ** 2)
        
        # Initial: t = 0
        x_ic = (2 * torch.rand(N_ic, 1) - 1.0).to(device)
        t_ic = torch.zeros_like(x_ic).to(device)
        z_ic = prepare_input(x_ic, t_ic, embedding_fn, device)
        u_ic = net(z_ic)
        u_ic_true = analytical(x_ic, t_ic)
        loss_ic = torch.mean((u_ic - u_ic_true) ** 2)
        
        loss = loss_pde + loss_bc + loss_ic
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Periodic evaluation
        if epoch % cfg.pinn.print_loss_every == 0:
            net.eval()
            with torch.no_grad():
                u_pred_eval = net(z_eval).reshape(nx, nt)
                rel_l2 = (torch.norm(u_pred_eval - u_true_eval) / torch.norm(u_true_eval)).item()
            
            sw.add_scalar('train/total', loss.item(), epoch)
            sw.add_scalar('train/pde', loss_pde.item(), epoch)
            sw.add_scalar('train/bc', loss_bc.item(), epoch)
            sw.add_scalar('train/ic', loss_ic.item(), epoch)
            sw.add_scalar('train/rel_l2', rel_l2, epoch)
            
            print(f"Epoch {epoch}: loss={loss.item():.3e}, pde={loss_pde.item():.3e}, "
                  f"bc={loss_bc.item():.3e}, ic={loss_ic.item():.3e}, L2={rel_l2:.3e}")
            
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
