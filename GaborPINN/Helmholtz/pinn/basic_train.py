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
from helmholtz_box import rhs_f, exact_u


def compute_laplacian(net, embedding_fn, x, y):
    """Compute u and Laplacian(u) = u_xx + u_yy."""
    s = torch.zeros_like(x)
    xyz = torch.cat([x, y, s], dim=1)
    xyz.requires_grad_(True)
    
    z = embedding_fn(xyz) if embedding_fn is not None else xyz
    u = net(z)
    
    grads = torch.autograd.grad(u, xyz, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    
    u_xx = torch.autograd.grad(u_x, xyz, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyz, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    
    return u, u_xx + u_yy


def prepare_input(x, y, embedding_fn, device):
    """Prepare input tensor (x, y, s=0) and apply embedding."""
    xyz = torch.cat([x, y, torch.zeros_like(x)], dim=1).to(device)
    return embedding_fn(xyz) if embedding_fn is not None else xyz


def evaluate(cfg, net, embedding_fn):
    """Evaluate model on full (x,y) domain."""
    device = torch.device(cfg.pinn.device)
    net.eval()
    
    nx, ny = cfg.pinn.nx, cfg.pinn.nz
    x_lin = torch.linspace(-1.0, 1.0, nx)
    y_lin = torch.linspace(-1.0, 1.0, ny)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing='ij')
    
    x_flat = X.reshape(-1, 1).to(device)
    y_flat = Y.reshape(-1, 1).to(device)
    z = prepare_input(x_flat, y_flat, embedding_fn, device)
    
    with torch.no_grad():
        u_pred = net(z).reshape(nx, ny)
    
    u_true = exact_u(X.to(device), Y.to(device))
    rel_l2 = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    
    print(f"Relative L2 error: {rel_l2:.4e}")
    
    # Plot comparison
    u_pred_np = u_pred.cpu().numpy()
    u_true_np = u_true.cpu().numpy()
    err_np = (u_pred - u_true).abs().cpu().numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    im0 = axs[0].imshow(u_true_np.T, origin='lower', extent=[-1, 1, -1, 1])
    axs[0].set_title("Exact u")
    fig.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(u_pred_np.T, origin='lower', extent=[-1, 1, -1, 1])
    axs[1].set_title("Predicted u")
    fig.colorbar(im1, ax=axs[1])
    
    im2 = axs[2].imshow(err_np.T, origin='lower', extent=[-1, 1, -1, 1])
    axs[2].set_title("|Error|")
    fig.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    os.makedirs(cfg.visualize_path, exist_ok=True)
    plt.savefig(os.path.join(cfg.visualize_path, "helmholtz_results.png"), dpi=150)
    plt.close()
    
    return rel_l2


def train(cfg, net, embedding_fn):
    """Train PINN for Helmholtz equation: Δu + u = f."""
    device = torch.device(cfg.pinn.device)
    seed_torch(cfg.pinn.seed)
    net = net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.pinn.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.pinn.step_size, gamma=cfg.pinn.gamma)
    sw = SummaryWriter(cfg.tensorboard_path)
    
    N_int, N_bnd = 20000, 4000
    min_loss = 1e20
    start_time = time.time()
    
    # Prepare evaluation grid (once before loop)
    nx, ny = cfg.pinn.nx, cfg.pinn.nz
    x_lin = torch.linspace(-1.0, 1.0, nx)
    y_lin = torch.linspace(-1.0, 1.0, ny)
    X_eval, Y_eval = torch.meshgrid(x_lin, y_lin, indexing='ij')
    x_eval = X_eval.reshape(-1, 1).to(device)
    y_eval = Y_eval.reshape(-1, 1).to(device)
    z_eval = prepare_input(x_eval, y_eval, embedding_fn, device)
    u_true_eval = exact_u(X_eval.to(device), Y_eval.to(device))
    
    for epoch in range(cfg.pinn.epochs):
        net.train()
        
        # Interior: PDE residual Δu + u = f
        x_int = (2 * torch.rand(N_int, 1) - 1.0).to(device)
        y_int = (2 * torch.rand(N_int, 1) - 1.0).to(device)
        u_int, lap_u = compute_laplacian(net, embedding_fn, x_int, y_int)
        f_int = rhs_f(x_int, y_int).to(device)
        loss_pde = torch.mean((lap_u + u_int - f_int) ** 2)
        
        # Boundary: 4 edges (x=±1, y=±1)
        n_edge = N_bnd // 4
        xb = 2 * torch.rand(n_edge, 1) - 1.0
        yb = 2 * torch.rand(n_edge, 1) - 1.0
        
        x_b = torch.cat([torch.ones_like(yb), -torch.ones_like(yb), xb, xb]).to(device)
        y_b = torch.cat([yb, yb, torch.ones_like(xb), -torch.ones_like(xb)]).to(device)
        
        z_b = prepare_input(x_b, y_b, embedding_fn, device)
        u_b = net(z_b)
        u_b_true = exact_u(x_b, y_b)
        loss_bc = torch.mean((u_b - u_b_true) ** 2)
        
        loss = loss_pde + loss_bc
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # ========== ADD L2 ERROR COMPUTATION ==========
        if epoch % cfg.pinn.print_loss_every == 0:
            net.eval()
            with torch.no_grad():
                u_pred_eval = net(z_eval).reshape(nx, ny)
                rel_l2 = (torch.norm(u_pred_eval - u_true_eval) / torch.norm(u_true_eval)).item()
            
            sw.add_scalar('train/loss', loss.item(), epoch)
            sw.add_scalar('train/pde', loss_pde.item(), epoch)
            sw.add_scalar('train/bc', loss_bc.item(), epoch)
            sw.add_scalar('train/rel_l2', rel_l2, epoch)  # Log L2 to TensorBoard
            
            print(f"Epoch {epoch}: loss={loss.item():.3e}, pde={loss_pde.item():.3e}, "
                  f"bc={loss_bc.item():.3e}, L2={rel_l2:.3e}")  # ← Now prints L2!
            
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
