from config import*
from models import*
from Wfamily import*
from Poisson import*


x_interior = x_collocation.clone()
y_interior = y_collocation.clone()
z_interior = z_collocation.clone()
def wpinn_loss(model):   
    # PDE loss at collocation points

    global c,b
    c, b = model(x_interior, y_interior, z_interior)

    u = torch.mv(Wfamily, c) + b
    u_yy = torch.mv(DW2y, c)
    u_xx = torch.mv(DW2x, c)
    u_zz = torch.mv(DW2z, c)
    
    u_pred_bc_x_l = torch.mv(Wbc_x_l, c) + b
    u_pred_bc_x_u = torch.mv(Wbc_x_u, c) + b
    u_pred_bc_y_l = torch.mv(Wbc_y_l, c) + b
    u_pred_bc_y_u = torch.mv(Wbc_y_u, c) + b
    u_pred_bc_z_l = torch.mv(Wbc_z_l, c) + b
    u_pred_bc_z_u = torch.mv(Wbc_z_u, c) + b
    
    pde_loss = torch.mean((u_yy + u_xx + u_zz - rhs) ** 2)
    
    bc_loss = torch.mean((u_pred_bc_x_l - u_bc_x_l) ** 2) +\
              torch.mean((u_pred_bc_x_u - u_bc_x_u) ** 2) +\
              torch.mean((u_pred_bc_y_l - u_bc_y_l) ** 2) + \
              torch.mean((u_pred_bc_y_u - u_bc_y_u) ** 2) + \
              torch.mean((u_pred_bc_z_l - u_bc_z_l) ** 2) + \
              torch.mean((u_pred_bc_z_u - u_bc_z_u) ** 2)
    
    total_loss = pde_loss + bc_loss
    
    return total_loss, pde_loss, bc_loss


global itr
itr = 0

def loss_print(model, total_loss, pde_loss, bc_loss, iteration):
    numerical = torch.mv(Wval, c.cpu()) + b.cpu()
    errL2 =  torch.norm(exact_validation-numerical) / torch.norm(exact_validation)
    errMax = torch.max(torch.abs(exact_validation-numerical))

    print(f'Epoch[{iteration}]  '
            f'Total Loss: {total_loss.item():.6f}, '
            f'PDE Loss: {pde_loss.item():.6f}, '
            f'BC Loss: {bc_loss.item():.6f}\n\t\t'
            f'RelativeL2: {errL2},\t'
            f'Max: {errMax}\n')

def train_wpinn(model, optimizer1, optimizer2, num_epochs, num_prints):

    print("\n======================== ADAM phase Training ========================\n")

    def closure():
        global itr
        optimizer2.zero_grad()
        total_loss, pde_loss, bc_loss = wpinn_loss(model)

        # numerical = torch.mv(Wval, c.cpu()) + b.cpu()
        # errL2 =  torch.norm(exact_validation-numerical) / torch.norm(exact_validation)
        # errMax = torch.max(torch.abs(exact_validation-numerical))

        # with open(log_filename, "a") as f:
        #     f.write(f"{total_loss.item():.6f},{pde_loss.item():.6f},{ic_loss.item():.6f},{bc_loss.item():.6f},{errL2.item():.6f},{errMax.item():.6f}\n")

        total_loss.backward()

        if itr % 1000 == 0:
            loss_print(model, total_loss, pde_loss, bc_loss, itr)

        itr += 1

        return total_loss
    

    for epoch in tqdm(range(num_epochs)):

        optimizer1.zero_grad()
        total_loss, pde_loss, bc_loss = wpinn_loss(model)

        # numerical = torch.mv(Wval, c.cpu()) + b.cpu()
        # errL2 =  torch.norm(exact_validation-numerical) / torch.norm(exact_validation)
        # errMax = torch.max(torch.abs(exact_validation-numerical))

        # with open(log_filename, "a") as f:
        #     f.write(f"{total_loss.item():.6f},{pde_loss.item():.6f},{ic_loss.item():.6f},{bc_loss.item():.6f},{errL2.item():.6f},{errMax.item():.6f}\n")
        

        total_loss.backward()
        optimizer1.step()
    
        if epoch % ((num_epochs-1)/num_prints) == 0:  # Print every num_prints epochs
            loss_print(model, total_loss, pde_loss, bc_loss, epoch)
    
    torch.cuda.empty_cache()

    print("\n============== starting LBFGS phase Training ==============\n")
    loss = optimizer2.step(closure)

    return c,b

