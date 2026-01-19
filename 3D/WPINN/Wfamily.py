from config import*

def waveley_family(Jx, Jy, Jz, a):

    family = torch.tensor([(2**jx,2**jy,2**jz,kx,ky,kz) for jx in Jx for jy in Jy for jz in Jz
                                               for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1) 
                                               for ky in range(int(torch.floor((y_lower-a)*2**(jy))), int(torch.ceil((y_upper+a)*2**(jy))) + 1)
                                               for kz in range(int(torch.floor((z_lower-a)*2**(jz))), int(torch.ceil((z_upper+a)*2**(jz))) + 1)])
    
    

    return len(family), family.to(device)


def gaussian(x, y, z, jx, jy, jz, kx, ky, kz):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    Z = jz[:, None] * z[None, :] - kz[:, None]
    return - X * Y * Z * torch.exp(-(X**2 + Y**2 + Z**2)/2)


def D2xgaussian(x, y, z, jx, jy, jz, kx, ky, kz):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    Z = jz[:, None] * z[None, :] - kz[:, None]
    return (jx[:, None]**2) * X * Y * Z * (3 - X**2) * torch.exp(-(X**2 + Y**2 + Z**2)/2)

def D2ygaussian(x, y, z, jx, jy, jz, kx, ky, kz):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    Z = jz[:, None] * z[None, :] - kz[:, None]
    return (jy[:, None]**2) * X * Y * Z * (3 - Y**2) * torch.exp(-(X**2 + Y**2 + Z**2)/2)

def D2zgaussian(x, y, z, jx, jy, jz, kx, ky, kz):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    Z = jz[:, None] * z[None, :] - kz[:, None]
    return (jz[:, None]**2) * X * Y * Z * (3 - Z**2) * torch.exp(-(X**2 + Y**2 + Z**2)/2)


# def print_sparsity(W):
#     nnz = torch.count_nonzero(W > 1e-12).item()  # threshold for "nonzero"
#     sparsity = 1 - nnz / W.numel()
#     print(f"{W.shape}: {nnz:,} nnz ({sparsity:.1%} sparse)")


print("\n======================== Computing Wavelet Family ========================\n")

Jx = torch.arange(-5.0,5.0)
Jy = torch.arange(-5.0,5.0)
Jz = torch.arange(-5.0,5.0)
a = 0.0

len_family, family = waveley_family(Jx, Jy, Jz, a)
print("family_len: ", len(family)) 

jx = family[:, 0]#.cpu()
jy = family[:, 1]#.cpu()
jz = family[:, 2]#.cpu()
kx = family[:, 3]#.cpu() 
ky = family[:, 4]#.cpu() 
kz = family[:, 5]#.cpu()



Wfamily =  gaussian(x_collocation, y_collocation, z_collocation, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()
DW2x = D2xgaussian(x_collocation, y_collocation, z_collocation, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()
DW2y = D2ygaussian(x_collocation, y_collocation, z_collocation, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()
DW2z = D2zgaussian(x_collocation, y_collocation, z_collocation, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()


Wbc_x_l = gaussian(x_bc_l, y_bc, z_bc, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()
Wbc_x_u = gaussian(x_bc_u, y_bc, z_bc, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()
Wbc_y_l = gaussian(x_bc, y_bc_l, z_bc, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()
Wbc_y_u = gaussian(x_bc, y_bc_u, z_bc, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()
Wbc_z_l = gaussian(x_bc, y_bc, z_bc_l, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()
Wbc_z_u = gaussian(x_bc, y_bc, z_bc_u, jx, jy, jz, kx, ky, kz).T
torch.cuda.empty_cache()

Wval = gaussian(x_validation, y_validation, z_validation, jx.cpu(), jy.cpu(), jz.cpu(), kx.cpu(), ky.cpu(), kz.cpu()).T
# Wtest = gaussian(x_test, y_test, z_test, jx.cpu(), jy.cpu(), jz.cpu(), kx.cpu(), ky.cpu(), kz.cpu()).T
torch.cuda.empty_cache()