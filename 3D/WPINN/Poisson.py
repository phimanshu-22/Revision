from config import*


def analytical(x,y,z):
    n = 3
    pi = torch.pi
    sx = torch.sin(n * pi * x)
    sy = torch.sin(n * pi * y)
    sz = torch.sin(n * pi * z)

    return sx * sy * sz


def right_side(x,y,z):
    n = 3
    pi = torch.pi
    sx = torch.sin(n * pi * x)
    sy = torch.sin(n * pi * y)
    sz = torch.sin(n * pi * z)

    return -3 * ((n*pi)**2) * sx * sy * sz



u_bc_x_l = analytical(x_bc_l, y_bc, z_bc)
u_bc_x_u = analytical(x_bc_u, y_bc, z_bc)
u_bc_y_l = analytical(x_bc, y_bc_l, z_bc)
u_bc_y_u = analytical(x_bc, y_bc_u, z_bc)
u_bc_z_l = analytical(x_bc, y_bc, z_bc_l)
u_bc_z_u = analytical(x_bc, y_bc, z_bc_u)


rhs = right_side(x_collocation, y_collocation, z_collocation)
exact_validation = analytical(x_validation, y_validation, z_validation)
# exact_test = analytical(x_test, y_test, z_test).reshape(n_test, n_test).numpy()
