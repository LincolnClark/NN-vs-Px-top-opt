import torch
from torch import nn
import matplotlib.pyplot as plt

import torcwa
from utils.utils import *
from NN_reparam.neural_network_architectures import NeuralNetwork
from NN_reparam.neural_network_architectures import train_loop_single_angle as train_loop

def cost_function(dens, options, angles, layers, target, pol, geom, sim_dtype):
    # Build layers
    eps =  options["mat 2"] + (options["mat 1"] - options["mat 2"])*(1 - dens)
    
    layers[0] = {"t": options["t"], "eps": eps}
    t = torch.zeros_like(target)
    for i in range(len(angles)):
        t_s, t_p = trans_at_angle_comp(layers, angles[i], options["phi"], options, 
                                       geom, sim_dtype)
        if pol == "s":
            t[i] = t_s
        elif pol == "p":
            t[i] = t_p
        else:
            raise Exception("Invalid polarisation")

    cost = torch.sum((t - target) ** 2)
    return torch.sqrt(cost/len(angles))

def NN_optim_pol(seed, lam, angles, target, pol, layers, options, sim_dtype, geo_dtype, device):
    
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

    for t in range(options["num iterations"]):
        train_loop(model, cost_function, optimiser, X, beta[t], kappa_hist, cost_hist, 
                   options, angles, layers, target, pol, geom, sim_dtype)

    model.eval()
    design = model(X)
    design = torch.special.expit(beta[-1] * design)

    # Final performance
    # Evaluate final performance
    with torch.no_grad():
        eps =  options["mat 2"] + (options["mat 1"] - options["mat 2"])*(1 - design)
    
        layers[0] = {"t": options["t"], "eps": eps}
        t = torch.zeros_like(target)
        for i in range(len(angles)):
            t_s, t_p = trans_at_angle_comp(layers, angles[i], options["phi"], options, 
                                        geom, sim_dtype)
            if pol == "s":
                t[i] = t_s
            elif pol == "p":
                t[i] = t_p

    return design.detach().cpu().numpy(), cost_hist, kappa_hist, t.detach().cpu().numpy()