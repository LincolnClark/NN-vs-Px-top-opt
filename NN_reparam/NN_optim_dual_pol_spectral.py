import torch
from torch import nn
import matplotlib.pyplot as plt

import torcwa
from utils.utils import *
from NN_reparam.neural_network_architectures import NeuralNetwork
from NN_reparam.neural_network_architectures import train_loop_dual_spectral as train_loop

def cost_function(dens, options, wavelengths, layers, targets, targetp, geom, sim_dtype):
    ts = torch.zeros_like(targets)
    tp = torch.zeros_like(targetp)

    for i in range(len(wavelengths)):

        # Build layers
        eps =  options["mat 2"][i] + (options["mat 1"] - options["mat 2"][i])*(1 - dens)
        layers[0] = {"t": options["t"], "eps": eps}
        options["lam"] = wavelengths[i]

        t_s, t_p = trans_at_angle_comp(layers, options["theta"], options["phi"], options, 
                                       geom, sim_dtype)
        ts[i] = t_s ** 2
        tp[i] = t_p ** 2

    cost = torch.sum((ts - targets) ** 2+ (tp - targetp) ** 2)/2
    return torch.sqrt(cost/len(wavelengths))

def NN_optim_pol(seed, wavelengths, targets, targetp, layers, options, sim_dtype, geo_dtype, device):
    
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
    model = NeuralNetwork(N, M, options["ker size"],
                          scale = options["scaling"],
                          channels = options["channels"],
                          offset = options["offset"],
                          dense_channels = options["dense channels"]).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=options["alpha NN"], 
                                betas = [options["beta 1"],options["beta 2"]],
                                eps = options["epsilon"])

    # Input vector of zeros into the NN
    X = torch.zeros(1, 1, N, M, device=device)

    beta = options["beta increase factor"]**(torch.linspace(1, options["num iterations"], options["num iterations"], 
                                                            device = device)/options["beta increase step"])

    kappa_hist = []
    cost_hist = []

    for t in range(options["num iterations"]):
        #print(f"Iteration {t+1}")

        train_loop(model, cost_function, optimiser, X, beta[t], kappa_hist, cost_hist, 
                   options, wavelengths, layers, targets, targetp, geom, sim_dtype)
    #print("Done!")

    model.eval()
    design = model(X)
    design = torch.special.expit(beta[-1] * design)

    # Evaluate final performance
    with torch.no_grad():
        ts = torch.zeros_like(targets)
        tp = torch.zeros_like(targetp)
        for i in range(len(wavelengths)):
            # Build layers
            eps =  options["mat 2"][i] + (options["mat 1"] - options["mat 2"][i])*(1 - design)
            layers[0] = {"t": options["t"], "eps": eps}
            options["lam"] = wavelengths[i]

            t_s, t_p = trans_at_angle_comp(layers, options["theta"], options["phi"], options, 
                                        geom, sim_dtype)
            ts[i] = t_s ** 2
            tp[i] = t_p ** 2

    return design.detach().cpu().numpy(), cost_hist, kappa_hist, ts.detach().cpu().numpy(), tp.detach().cpu().numpy()