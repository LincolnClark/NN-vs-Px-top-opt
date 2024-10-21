import torch
import time
from NN_reparam.NN_optim_pol_dependent import NN_optim_pol
from Pixel_param.pixel_optim_pol_dependent import pixel_optim_pol
from utils.material import SiO2
from utils.plot_compare import *
from utils.measurement import length_scale, convergence_time, binarisation_level

def quadratic_OTF(angles, NA):
    kx = torch.sin(torch.deg2rad(angles))
    return kx**2 / NA**2

def identity_OTF(angles, val):
    return val * torch.ones(angles.shape)

def run_ang_benchmark(lam, patterned_material, t, angles, targets, targetp, 
                      period, layers, label, res_folder):
    """
    lam - wavelength in free space (nm)
    patterend_material - Material class for patterned material
    t - thickness of patterned layer (nm)
    angles - angles to evaluate OTF at (deg)
    targets - target OTF for s polarised light
    target- - target OTF for p polarised light
    period - (Lx, Ly) period along x and y
    layers - list of layers, [None] for no extra layers
    label - string labelling current benchmark
    res_folder - string for filepath to save in
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    seed = 39 # Starting seed for random number generation

    options = {
            # Geometry and material
            "t": t, # thickness of the design domain
            "mat 1": 1., # permittivity when density is 0
            "mat 2": patterned_material.eps(lam), # permittivity when density is 1
            "superstrate": 1., # Superstrate permittivity
            "substrate": SiO2.eps(lam), # substrate permittivity
            "Lx": period[0], # Period along x
            "Ly": period[1], # Period along y

            # Illumination
            "lam": lam, # Wavelegnth in free space
            "theta": 1e-8,
            "phi": 1e-8,

            # RCWA settings
            "M": 9, # Number of Fourier coefficients along x
            "N": 9, # Number of Fourier coefficients along y
            "nx": 240, # number of x points for design domain
            "ny": 240, # Number of y points for design domain
            "N NN": 15,
            "M NN": 15,
            "t NN": 2,
            
            # Optimisation settings
            "num iterations": 300,
            "beta increase factor": 1.2,
            "beta increase step": 20, # Number of iterations between thresholding increases
            "eta norm": 0.5,

            # Robustness options
            "blur radius": 30,

            # ADAM optimiser settings
            "alpha": 0.01, # max step size
            "alpha NN": 0.001,
            "beta 1": 0.9, # decay rate of 1st moment
            "beta 2": 0.999, # decay rate of 2nd moment
            "epsilon": 1e-6 # factor to avoid divison by 0
        }

    sim_dtype = torch.complex64
    geo_dtype = torch.float32

    # (seed, lam, angles, targets, targetp, layers, options, sim_dtype, geo_dtype, device
    t = time.time()
    NN_des, NN_cost, NN_des_hist, NN_ts, NN_tp  = NN_optim_pol(seed, lam, angles, 
                                                               targets, targetp, 
                                                               layers, options, 
                                                               sim_dtype, geo_dtype, device)
    NN_time = time.time() - t

    t = time.time()
    px_des, px_cost, px_des_hist, px_ts, px_tp = pixel_optim_pol(seed, lam, angles, 
                                                                 targets, targetp, 
                                                                 layers, options, 
                                                                 sim_dtype, geo_dtype, device)
    px_time = time.time() - t

    # Plots
    compare_cost(NN_cost, px_cost, f"{res_folder}{label}_cost_compare.png")
    compare_final_designs(NN_des, px_des, f"{res_folder}{label}_design_compare.png")
    compare_performances(NN_ts, NN_tp, px_ts, px_tp, targets, targetp,
                         angles, f"{res_folder}{label}_performance_compare.png")
    animate_history(NN_des_hist, f"{res_folder}{label}_kappa_NN_ani.gif")
    animate_history(px_des_hist, f"{res_folder}{label}_kappa_px_ani.gif")

    # Calculate the benchmark parameters
    NN_final_cost = NN_cost[-1]
    px_final_cost = px_cost[-1]
    NN_fs_solid, NN_fs_void = length_scale(NN_des > 0.5, period[0]/NN_des.shape[0])
    px_fs_solid, px_fs_void = length_scale(px_des > 0.5, period[0]/px_des.shape[0])
    NN_conv_time = convergence_time(NN_cost)
    px_conv_time = convergence_time(px_cost)
    NN_bin_level = binarisation_level(NN_des)
    px_bin_level = binarisation_level(px_des)

    return ((NN_final_cost, NN_time, NN_fs_solid, NN_fs_void, NN_conv_time, NN_bin_level), 
            (px_final_cost, px_time, px_fs_solid, px_fs_void, px_conv_time, px_bin_level))