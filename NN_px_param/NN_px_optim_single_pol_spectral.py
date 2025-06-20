import torch
from torch import nn
import matplotlib.pyplot as plt

import torcwa
from utils.utils import *
from utils.neural_network_architectures import NeuralNetwork
from utils.neural_network_architectures import train_loop_single_spectral as train_loop

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

def NN_px_optim_pol(seed, wavelengths, target, pol, layers, options, sim_dtype, geo_dtype, device):

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

    # Work out the shape of the input vector into NN
    N, M = (options["N NN"], options["M NN"])
    N, M = (options["N NN"], options["M NN"])
    model = NeuralNetwork(N, M, options["ker size"],
                          scale = options["scaling"],
                          channels = options["channels"],
                          offset = options["offset"],
                          dense_channels = options["dense channels"],
                          blur = options["NN blur"],
                          device = device).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=options["alpha NN"], 
                                betas = [options["beta 1"],options["beta 2"]],
                                eps = options["epsilon"])

    # Input vector of zeros into the NN
    X = torch.zeros(1, 1, N, M, device=device)

    beta = options["beta increase factor"]**(torch.linspace(1, options["num iterations"], options["num iterations"], 
                                                            device = device)/options["beta increase step"])

    kappa_hist = []
    cost_hist = []

    for t in range(options["num NN"]):
        train_loop(model, cost_function, optimiser, X, beta[t], kappa_hist, cost_hist, 
                   options, wavelengths, layers, target, pol, geom, sim_dtype)

    model.eval()
    design = model(X)

    # LMpx training
    x = torch.linspace(-options["Lx"]/2, options["Lx"]/2, options["nx"])
    y = torch.linspace(-options["Ly"]/2, options["Ly"]/2, options["ny"])
    xx, yy = torch.meshgrid(x, y, indexing = "ij")
    
    gamma = options["NN px fact"] * design.detach()

    # Velocity and momentum for ADAM
    mt = torch.zeros_like(gamma)
    vt = torch.zeros_like(gamma)

    iter = options["num NN"] - 1
    while iter < options["num iterations"]:

        gamma.requires_grad_(True)

        # Perform blurring
        gamma_blur = filter(gamma, options["blur NN px"], xx, yy, geo_dtype, device)
        kappa_norm = torch.special.expit(beta[iter] * gamma_blur)

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
            gamma, mt, vt = update_with_adam(options["alpha NN px"], options["beta 1"], options["beta 2"], 
                                             options["epsilon"], grad, mt, vt, iter, gamma)

            # Normalise gamma
            gamma = (gamma - torch.mean(gamma))/torch.sqrt(torch.var(gamma) + 1e-5)

            # Update history
            cost_hist.append(cost.detach().cpu().numpy())
            kappa_hist.append(kappa_norm.detach().cpu().numpy())

            iter = iter + 1

    # Evaluate final performance
    with torch.no_grad():
        t = torch.zeros_like(target)

        for i in range(len(wavelengths)):
            # Build layers
            eps =  options["mat 2"][i] + (options["mat 1"] - options["mat 2"][i])*(1 - kappa_norm)
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

    return design.detach().cpu().numpy(), cost_hist, kappa_hist, t.detach().cpu().numpy()