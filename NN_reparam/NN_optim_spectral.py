import torch
from torch import nn
import matplotlib.pyplot as plt

import torcwa
from utils.utils import *

def cost_function(dens, options, wavelengths, layers, targets, targetp, geom, sim_dtype):
    # Build layers
    # TODO: Dispersion
    eps =  options["mat 2"] + (options["mat 1"] - options["mat 2"])*(1 - dens)
    
    layers[0] = {"t": options["t"], "eps": eps}
    ts = torch.zeros_like(targets)
    tp = torch.zeros_like(targetp)

    for i in range(len(wavelengths)):
        options["lam"] = wavelengths[i]

        t_s, t_p = trans_at_angle_comp(layers, options["theta"], options["phi"], options, 
                                       geom, sim_dtype)
        ts[i] = t_s ** 2
        tp[i] = t_p ** 2

    cost = torch.sum((ts - targets) ** 2+ (tp - targetp) ** 2)/2
    return torch.sqrt(cost/len(wavelengths))

class BiasLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bias_value = torch.randn((1))
        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return x + self.bias_layer
    
class RandInput(torch.nn.Module):
    def __init__(self, len) -> None:
        super().__init__()
        bias_value = torch.randn((len))
        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return x + self.bias_layer
    
class NeuralNetwork(nn.Module):
    def __init__(self, n, m, scale):
        super().__init__()

        self.flatten = nn.Flatten()

        self.nn_modules = nn.ModuleList([
            RandInput(n*m),
            nn.Linear(n * m, n * m),
            nn.Tanh(),
            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(1, 256, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),

            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(256, 128, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),

            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(128, 64, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),

            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(64, 32, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),

            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(32, 16, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),
            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(16, 1, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer()]
        )

        self.normalise_loc = []#[3, 8, 13, 18, 22]
        self.upscale_loc = [7, 12, 17, 22]
        self.n_modules = len(self.nn_modules)
        self.n_functionals = len(self.normalise_loc) + len(self.upscale_loc)

        self.n = n
        self.m = m
        self.scale = scale


    def forward(self, x):
        x = self.flatten(x)
        x = self.nn_modules[0](x)
        x = self.nn_modules[1](x)
        x = torch.reshape(x, (1, 1, self.n, self.m))

        j = 2
        for i in range(2, self.n_modules + self.n_functionals):
            #if i in self.normalise_loc:
            #    x = nn.functional.normalize(x)
            if i in self.upscale_loc:
                x = nn.functional.interpolate(x, mode = "bilinear", scale_factor = self.scale)
            else:
                x = self.nn_modules[j](x)
                j = j + 1

        x = x[0, 0, :, :]

        return x
    
def train_loop(model, loss_fn, optimiser, x, beta, hist, c_hist, 
               options, wavelengths, layers, targets, targetp, geom, sim_dtype):

    # Set the model to training mode
    model.train()
    
    # Forward simulation
    gamma = model(x)

    if True in torch.isnan(gamma):
        print("NaN in gamma")
    # Binarisation
    kappa = torch.special.expit(beta * gamma)

    cost = loss_fn(kappa, options, wavelengths, layers, targets, targetp, geom, sim_dtype)

    # Backpropagation
    cost.backward()
    optimiser.step()
    optimiser.zero_grad()

    hist.append(kappa.detach().cpu().numpy())
    c_hist.append(cost.detach().cpu().numpy())

    #print(f"cost: {cost.detach().cpu():.5f}\n-------------------------------")
    

def NN_optim_spectral(seed, wavelengths, targets, targetp, layers, options, sim_dtype, geo_dtype, device):
    
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
    model = NeuralNetwork(N, M, options["t NN"]).to(device)

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

    # Final performance
    # Evaluate final performance
    with torch.no_grad():
        eps =  options["mat 2"] + (options["mat 1"] - options["mat 2"])*(1 - design)
    
        layers[0] = {"t": options["t"], "eps": eps}
        ts = torch.zeros_like(targets)
        tp = torch.zeros_like(targetp)

        for i in range(len(wavelengths)):
            options["lam"] = wavelengths[i]

            t_s, t_p = trans_at_angle_comp(layers, options["theta"], options["phi"], options, 
                                        geom, sim_dtype)
            ts[i] = t_s ** 2
            tp[i] = t_p ** 2

    return design.detach().cpu().numpy(), cost_hist, kappa_hist, ts.detach().cpu().numpy(), tp.detach().cpu().numpy()