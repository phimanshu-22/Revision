from config import*
from models import*
from Poisson import*

def pinn_loss(model):
    x_interior = x_collocation.clone().detach().requires_grad_(True)
    y_interior = y_collocation.clone().detach().requires_grad_(True)
    z_interior = z_collocation.clone().detach().requires_grad_(True)

    u = model(x_interior, y_interior, z_interior) 

    grads = torch.autograd.grad(u, [x_interior, y_interior, z_interior], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y, u_z = grads

    u_xx = torch.autograd.grad(u_x, x_interior, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_interior, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z_interior, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]

    u_pred_bc_x_l = model(x_bc_l, y_bc, z_bc)
    u_pred_bc_x_u = model(x_bc_u, y_bc, z_bc)
    u_pred_bc_y_l = model(x_bc, y_bc_l, z_bc)
    u_pred_bc_y_u = model(x_bc, y_bc_u, z_bc)
    u_pred_bc_z_l = model(x_bc, y_bc, z_bc_l)
    u_pred_bc_z_u = model(x_bc, y_bc, z_bc_u)

    rhs_col = rhs.reshape(-1, 1)

    pde_loss = torch.mean((u_xx + u_yy + u_zz - rhs_col) ** 2)

    bc_loss = torch.mean((u_pred_bc_x_l - u_bc_x_l) ** 2) +\
              torch.mean((u_pred_bc_x_u - u_bc_x_u) ** 2) +\
              torch.mean((u_pred_bc_y_l - u_bc_y_l) ** 2) + \
              torch.mean((u_pred_bc_y_u - u_bc_y_u) ** 2) + \
              torch.mean((u_pred_bc_z_l - u_bc_z_l) ** 2) + \
              torch.mean((u_pred_bc_z_u - u_bc_z_u) ** 2)

    loss = pde_loss + bc_loss

    return loss, pde_loss, bc_loss


global itr
itr = 0

def loss_print(model, total_loss, pde_loss, bc_loss, iteration):
    numerical = model(x_validation, y_validation, z_validation)
    errL2 = torch.norm(exact_validation-numerical) / torch.norm(exact_validation)
    errMax = torch.max(torch.abs(exact_validation-numerical))

    print(f'Epoch[{iteration}]  '
            f'Total Loss: {total_loss.item():.6f}, '
            f'PDE Loss: {pde_loss.item():.6f}, '
            f'BC Loss: {bc_loss.item():.6f}\n\t\t'
            f'RelativeL2: {errL2},\t'
            f'Max: {errMax}\n')

def train_pinn(model, optimizer1, optimizer2, num_epochs, num_prints):

    print("\n======================== ADAM phase Training ========================\n")


    for epoch in tqdm(range(num_epochs)):

        # total_loss, pde_loss, bc_loss = mmpinn_loss(model)

        # numerical = torch.mv(Wval, c.cpu()) + b.cpu()
        # errL2 =  torch.norm(exact_validation-numerical) / torch.norm(exact_validation)
        # errMax = torch.max(torch.abs(exact_validation-numerical))

        # with open(log_filename, "a") as f:
        #     f.write(f"{total_loss.item():.6f},{pde_loss.item():.6f},{ic_loss.item():.6f},{bc_loss.item():.6f},{errL2.item():.6f},{errMax.item():.6f}\n")
        
        optimizer1.zero_grad()
        loss, pde_loss, bc_loss = pinn_loss(model)
        loss.backward()
        optimizer1.step()
    
        if epoch % ((num_epochs-1)/num_prints) == 0:  # Print every num_prints epochs
            loss_print(model, loss, pde_loss, bc_loss, epoch)
    
    torch.cuda.empty_cache()

    print("\n============== starting LBFGS phase Training ==============\n")

    def closure_fn():
        global itr
        optimizer2.zero_grad()
        loss, pde_loss, bc_loss = pinn_loss(model)
        loss.backward()

        if itr % 1000 == 0:
            loss_print(model, loss, pde_loss, bc_loss, itr)
        itr += 1

        return loss


    print(f"\n---- LBFGS Stage ----\n")
    optimizer2.step(closure_fn)


