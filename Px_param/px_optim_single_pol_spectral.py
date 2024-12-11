import torch
from torch import nn
import matplotlib.pyplot as plt

import torcwa
from utils.utils import *

def cost_function(dens, options, wavelengths, layers, target, pol, geom, sim_dtype):
    
    t = torch.zeros_like(target)

    for i in range(len(wavelengths)):
        # Build layers
        eps =  options["mat 2"][i] + (options["mat 1"] - options["mat 2"][i])*(1 - dens)
        layers[0] = {"t": options["t"], "eps": eps}
        options["lam"] = wavelengths[i]

        t_s, t_p = trans_at_angle_comp(layers, options["theta"], options["phi"], options, 
                                       geom, sim_dtype)
        if pol == "s":
            t[i] = t_s
        elif pol == "p":
            t[i] = t_p
        else:
            raise Exception("Invalid polarisation")

    cost = torch.sum((t - target) ** 2)
    return torch.sqrt(cost/len(wavelengths))

def pixel_optim_pol(seed, wavelengths, target, pol, layers, options, sim_dtype, geo_dtype, device):

    # Starting seed for random number generation
    torch.manual_seed(seed)

    # If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
    # If you need accurate operation, you have to disable the flag below.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.autograd.set_detect_anomaly(True) # Check for problems with gradient

    # Setup TORCWA geometry
    geom = torcwa.rcwa_geo()
    geom.dtype = geo_dtype
    geom.device = device
    geom.Lx = options["Lx"]
    geom.Ly = options["Ly"]
    geom.nx = options["nx"]
    geom.ny = options["ny"]
    geom.grid()
    geom.edge_sharpness = 500.
    x_axis = geom.x.cpu()
    y_axis = geom.y.cpu()

    # Starting guess using seed
    x = torch.linspace(-options["Lx"]/2, options["Lx"]/2, options["nx"])
    y = torch.linspace(-options["Ly"]/2, options["Ly"]/2, options["ny"])

    xx, yy = torch.meshgrid(x, y, indexing = "ij")
    gamma = torch.rand((options["nx"], options["ny"]), dtype = geo_dtype, device = device)
    gamma = filter(gamma, 40, xx, yy, geo_dtype, device)

    # Velocity and momentum for ADAM
    mt = torch.zeros_like(gamma)
    vt = torch.zeros_like(gamma)

    beta = options["beta increase factor"]**(torch.linspace(1, options["num iterations"], options["num iterations"], 
                                                            device = device)/options["beta increase step"])
    beta = 0.5*beta # scale beta down for the pixel approach

    iter = 0
    cost_hist = []
    norm_kappa_hist = []

    # Main optimisation loop
    while iter < options["num iterations"]:
        #print(f"Iteration {iter+1}")

        gamma.requires_grad_(True)

        # Perform blurring
        if options["blur radius"] is not None:
            gamma_blur = filter(gamma, options["blur radius"], xx, yy, geo_dtype, device)
            kappa_norm = projection(gamma_blur, beta[iter], 0.5, geo_dtype, device)
        else:
            kappa_norm = projection(gamma, beta[iter], 0.5, geo_dtype, device)

        cost = cost_function(kappa_norm, options, wavelengths, layers, target, pol, geom, sim_dtype)

        # Work out gradient of cost function w.r.t density with backpropagation
        cost.backward()

        with torch.no_grad(): # Disables gradient calculation
            grad = gamma.grad
            gamma.grad = None

            # Check for NaN
            if True in torch.isnan(grad):
                print("NaN detected in gradient")
                plt.imshow(kappa_norm.detach().cpu().numpy())
                plt.show()
                
            # Update density with ADAM
            gamma, mt, vt = update_with_adam(options["alpha"], options["beta 1"], options["beta 2"], 
                                             options["epsilon"], grad, mt, vt, iter, gamma)

            # Force BCs
            gamma[gamma < 0] = 0
            gamma[gamma > 1] = 1

            # Update history
            cost_hist.append(cost.detach().cpu().numpy())
            norm_kappa_hist.append(kappa_norm.detach().cpu().numpy())

            #print(f"cost: {cost.detach().cpu():.5f}\n-------------------------------")

            iter = iter + 1


    # Evaluate final performance
    with torch.no_grad():
        eps =  options["mat 2"] + (options["mat 1"] - options["mat 2"])*(1 - kappa_norm)
    
        layers[0] = {"t": options["t"], "eps": eps}
        t = torch.zeros_like(target)

        for i in range(len(wavelengths)):
            t_s, t_p = trans_at_angle_comp(layers, options["theta"], options["phi"], options, 
                                           geom, sim_dtype)
            if pol == "s":
                t[i] = t_s**2
            elif pol == "p":
                t[i] = t_p**2
            else:
                raise Exception("Invalid polarisation")

    return kappa_norm.detach().cpu().numpy(), cost_hist, norm_kappa_hist, t.detach().cpu().numpy()