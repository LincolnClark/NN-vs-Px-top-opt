import torch
import time
import pickle
import numpy as np

from NN_param.NN_optim_dual_pol_OTF import NN_optim_pol as NN_pol_dep
from LMpx_param.LMpx_optim_dual_pol_OTF import pixel_optim_pol as LMpx_pol_dep
from Px_param.px_optim_dual_pol_OTF import pixel_optim_pol as px_pol_dep
from NN_px_param.NN_px_optim_dual_pol_OTF import NN_px_optim_pol as NN_px_pol_dep

from NN_param.NN_optim_single_pol_OTF import NN_optim_pol as NN_no_pol
from LMpx_param.LMpx_optim_single_pol_OTF import pixel_optim_pol as LMpx_no_pol
from Px_param.px_optim_single_pol_OTF import pixel_optim_pol as px_no_pol
from NN_px_param.NN_px_optim_single_pol_OTF import NN_px_optim_pol as NN_px_no_pol

from utils.material import SiO2
from utils.plotting import *
from utils.measurement import length_scale, convergence_time

def params(t, patterned_material, lam, period, blur):
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
            "nx": 320, # number of x points for design domain
            "ny": 320, # Number of y points for design domain

            # NN options
            "ker size" : 5,
            "channels": [128, 64, 32, 16, 1],
            "dense channels": 16,
            "scaling": [1, 2, 2, 3, 1],
            "offset": [True, True, True, False, False],
            "N NN": 20,
            "M NN": 20,

            # Optimisation settings
            "num iterations": 300,
            "num NN": 150, # Number of NN iterations for NNpx method
            "NN px fact": 0.5,
            "beta increase factor": 1.4,
            "beta increase step": 15, # Number of iterations between thresholding increases
            "eta norm": 0.5,

            # Robustness options
            "blur radius": blur,
            "NN blur": None,
            "blur NN px": 0.1 * lam,

            # ADAM optimiser settings
            "alpha": 0.03, # max step size
            "alpha NN": 0.001,
            "alpha NN px": 0.02,
            "beta 1": 0.9, # decay rate of 1st moment
            "beta 2": 0.999, # decay rate of 2nd moment
            "epsilon": 1e-8 # factor to avoid divison by 0
        }
    
    options["nx"] = options["N NN"] * np.prod(options["scaling"])
    options["ny"] = options["M NN"] * np.prod(options["scaling"])

    sim_dtype = torch.complex64
    geo_dtype = torch.float32

    return device, seed, options, sim_dtype, geo_dtype

def quadratic_OTF(angles, NA):
    kx = torch.sin(torch.deg2rad(angles))
    return kx**2 / NA**2

def identity_OTF(angles, val):
    return val * torch.ones(angles.shape)

def run_pol_dependent_ang_benchmark(lam, patterned_material, t, angles, targets, targetp, 
                                    period, layers, label, res_folder, blur_frac = [None]):
    """
    Perform optimisations for benchmark that optimises for both s and p polarised light
    ARGUMENTS:
        lam - wavelength in free space (nm)
        patterend_material - Material class for patterned material
        t - thickness of patterned layer (nm)
        angles - angles to evaluate OTF at (deg)
        targets - target OTF for s polarised light
        targetp- - target OTF for p polarised light
        period - (Lx, Ly) period along x and y
        layers - list of layers, [None] for no extra layers
        label - string labelling current benchmark
        res_folder - string for filepath to save in
        blur_frac - list of amount of blurring (fraction of wl)
    RETURNS:
        Benchmark parameters, NN, NNpx, LMpx, px
    """

    # Convert blur from % of wl to nm
    blur = [i * lam if i is not None else None for i in blur_frac]

    device, seed, options, sim_dtype, geo_dtype = params(t, patterned_material, lam, period, blur)

    # ========================================================================================================
    # Run neural network optimisation
    t = time.time()
    NN_des, NN_cost, NN_des_hist, NN_ts, NN_tp  = NN_pol_dep(seed, lam, angles, 
                                                             targets, targetp, 
                                                             layers, options, 
                                                             sim_dtype, geo_dtype, device)
    NN_time = time.time() - t
    plot_final_design(NN_des, "Neural Network", f"{res_folder}{label}_NN_design.png")
    animate_history(NN_des_hist, f"{res_folder}{label}_NN_ani.gif")
    plot_otf_perfomance_dual_pol(NN_ts, NN_tp, targets, targetp, angles, "NN Optimisation",
                                 f"{res_folder}{label}_NN_performance.png")
    plot_cost(NN_cost, "NN Optimisation", f"{res_folder}{label}_NN_cost_evolution.png")

    # ========================================================================================================
    # Run neural network + pixel optimisation
    t = time.time()
    NN_px_des, NN_px_cost, NN_px_des_hist, NN_px_ts, NN_px_tp  = NN_px_pol_dep(seed, lam, angles, 
                                                                               targets, targetp, 
                                                                               layers, options, 
                                                                               sim_dtype, geo_dtype, device)
    NN_px_time = time.time() - t
    plot_final_design(NN_px_des, "NN + px ", f"{res_folder}{label}_NNpx_design.png")
    animate_history(NN_px_des_hist, f"{res_folder}{label}_NNpx_ani.gif")
    plot_otf_perfomance_dual_pol(NN_px_ts, NN_px_tp, targets, targetp, angles, "NNpx Optimisation",
                                 f"{res_folder}{label}_NNpx_performance.png")
    plot_cost(NN_px_cost, "NNpx Optimisation", f"{res_folder}{label}_NNpx_cost_evolution.png")

    # ========================================================================================================
    # Run latent matrix pixel optimisation
    LMpx_times = []
    LMpx_designs = []
    LMpx_costs = []
    LMpx_des_hists = []
    LMpx_tss = []
    LMpx_tps = []
    
    for bl in blur:
        options["blur radius"] = bl
        if bl is not None:
            lab = f"{bl / lam:.0%}"
        else:
            lab = "0%"
        t = time.time()
        LMpx_des, LMpx_cost, LMpx_des_hist, LMpx_ts, LMpx_tp = LMpx_pol_dep(seed, lam, angles, 
                                                                            targets, targetp, 
                                                                            layers, options, 
                                                                            sim_dtype, geo_dtype, device)
        LMpx_times.append(time.time() - t)
        LMpx_designs.append(LMpx_des)
        LMpx_costs.append(LMpx_cost)
        LMpx_des_hists.append(LMpx_des_hist)
        LMpx_tss.append(LMpx_ts)
        LMpx_tps.append(LMpx_tp)

        plot_final_design(LMpx_des, f"LMpx {lab} blur", f"{res_folder}{label}_LMpx_{lab}_design.png")
        animate_history(LMpx_des_hist, f"{res_folder}{label}_LMpx_{lab}_ani.gif")
        plot_otf_perfomance_dual_pol(LMpx_ts, LMpx_tp, targets, targetp, angles, f"LMpx {lab} blur",
                                     f"{res_folder}{label}_LMpx_{lab}_performance.png")
        plot_cost(LMpx_cost, f"LMpx {lab} blur", f"{res_folder}{label}_LMpx_{lab}_cost_evolution.png")

    # ========================================================================================================
    # Run constrained pixel optimisation
    px_times = []
    px_designs = []
    px_costs = []
    px_des_hists = []
    px_tss = []
    px_tps = []
    
    for bl in blur:
        options["blur radius"] = bl
        if bl is not None:
            lab = f"{bl / lam:.0%}"
        else:
            lab = "0%"
        t = time.time()
        px_des, px_cost, px_des_hist, px_ts, px_tp = px_pol_dep(seed, lam, angles, 
                                                                targets, targetp, 
                                                                layers, options, 
                                                                sim_dtype, geo_dtype, device)
        px_times.append(time.time() - t)
        px_designs.append(px_des)
        px_costs.append(px_cost)
        px_des_hists.append(px_des_hist)
        px_tss.append(px_ts)
        px_tps.append(px_tp)

        plot_final_design(px_des, f"px {lab} blur", f"{res_folder}{label}_px_{lab}_design.png")
        animate_history(px_des_hist, f"{res_folder}{label}_px_{lab}_ani.gif")
        plot_otf_perfomance_dual_pol(px_ts, px_tp, targets, targetp, angles, f"px {lab} blur",
                                     f"{res_folder}{label}_px_{lab}_performance.png")
        plot_cost(px_cost, f"px {lab} blur", f"{res_folder}{label}_px_{lab}_cost_evolution.png")

    # ========================================================================================================
    # Save everything into pickle files
    with open(f"{res_folder}{label}_NN_data.pickle", "wb") as file:
        pkl_data = {
            "blur": blur,
            "cost history": NN_cost,
            "final designs": NN_des,
            "perf s pol": NN_ts,
            "perf p pol": NN_tp,
            "target s": targets,
            "target p": targetp,
            "angles": angles,
        }
        pickle.dump(pkl_data, file)

    with open(f"{res_folder}{label}_NNpx_data.pickle", "wb") as file:
        pkl_data = {
            "blur": blur,
            "cost history": NN_px_cost,
            "final designs":  NN_px_des,
            "perf s pol": NN_px_ts,
            "perf p pol": NN_px_ts,
            "target s": targets,
            "target p": targetp,
            "angles": angles,
        }
        pickle.dump(pkl_data, file)

    with open(f"{res_folder}{label}_LMpx_data.pickle", "wb") as file:
        pkl_data = {
            "blur": blur,
            "cost history": LMpx_costs,
            "final designs": LMpx_designs,
            "perf s pol": LMpx_tss,
            "perf p pol": LMpx_tps,
            "target s": targets,
            "target p": targetp,
            "angles": angles,
        }
        pickle.dump(pkl_data, file)

    with open(f"{res_folder}{label}_px_data.pickle", "wb") as file:
        pkl_data = {
            "blur": blur,
            "cost history": px_costs,
            "final designs": px_designs,
            "perf s pol": px_tss,
            "perf p pol": px_tps,
            "target s": targets,
            "target p": targetp,
            "angles": angles,
        }
        pickle.dump(pkl_data, file)

    # ========================================================================================================
    # Calculate the benchmark parameters
    NN_final_cost = NN_cost[-1]
    NN_px_final_cost = NN_px_cost[-1]
    LMpx_final_cost = [i[-1] for i in LMpx_costs]
    px_final_cost = [i[-1] for i in px_costs]

    NN_fs_solid, NN_fs_void = length_scale(NN_des > 0.5, period[0]/NN_des.shape[0])
    NN_px_fs_solid, NN_px_fs_void = length_scale(NN_px_des > 0.5, period[0]/NN_px_des.shape[0])
    LMpx_fs_solid = []
    LMpx_fs_void = []
    px_fs_solid = []
    px_fs_void = []
    for i in range(len(blur)):
        fssolid, fssvoid = length_scale(LMpx_designs[i] > 0.5, period[0]/LMpx_designs[i].shape[0])
        LMpx_fs_solid.append(fssolid)
        LMpx_fs_void.append(fssvoid)
        fssolid, fssvoid = length_scale(px_designs[i] > 0.5, period[0]/px_designs[i].shape[0])
        px_fs_solid.append(fssolid)
        px_fs_void.append(fssvoid)
        
    NN_conv_time = convergence_time(NN_cost)
    NN_px_conv_time = convergence_time(NN_px_cost)
    LMpx_conv_time = []
    px_conv_time = []
    for i in range(len(blur)):
        LMpx_conv_time.append(convergence_time(LMpx_costs[i]))
        px_conv_time.append(convergence_time(px_costs[i]))

    return ((NN_final_cost, NN_time, NN_fs_solid, NN_fs_void, NN_conv_time),
            (NN_px_final_cost, NN_px_time, NN_px_fs_solid, NN_px_fs_void, NN_px_conv_time),
            (LMpx_final_cost, LMpx_times, LMpx_fs_solid, LMpx_fs_void, LMpx_conv_time),
            (px_final_cost, px_times, px_fs_solid, px_fs_void, px_conv_time))


def run_pol_ind_ang_benchmark(lam, patterned_material, t, angles, target, pol, 
                              period, layers, label, res_folder, blur_frac = [None]):
    """
    Perform optimisations for benchmark that optimises for a single polarisation
    ARGUMENTS:
        lam - wavelength in free space (nm)
        patterend_material - Material class for patterned material
        t - thickness of patterned layer (nm)
        angles - angles to evaluate OTF at (deg)
        target- - target OTF
        pol - "s" or "p"
        period - (Lx, Ly) period along x and y
        layers - list of layers, [None] for no extra layers
        label - string labelling current benchmark
        res_folder - string for filepath to save in
        blur_frac - list of amount of blurring (fraction of wl)
    RETURNS:
        Benchmark parameters, NN, NNpx, LMpx, px
    """

    # Convert blur from % of wl to nm
    blur = [i * lam if i is not None else None for i in blur_frac]

    device, seed, options, sim_dtype, geo_dtype = params(t, patterned_material, lam, period, blur)

    # ========================================================================================================
    # Run neural network optimisation
    t = time.time()
    NN_des, NN_cost, NN_des_hist, NN_t = NN_no_pol(seed, lam, angles, 
                                                    target, pol, 
                                                    layers, options, 
                                                    sim_dtype, geo_dtype, device)
    NN_time = time.time() - t
    plot_final_design(NN_des, "Neural Network", f"{res_folder}{label}_NN_design.png")
    animate_history(NN_des_hist, f"{res_folder}{label}_NN_ani.gif")
    plot_otf_perfomance_single_pol(NN_t, target, angles, "NN Optimisation",
                                 f"{res_folder}{label}_NN_performance.png")
    plot_cost(NN_cost, "NN Optimisation", f"{res_folder}{label}_NN_cost_evolution.png")

    # ========================================================================================================
    # Run neural network + pixel optimisation
    t = time.time()
    NN_px_des, NN_px_cost, NN_px_des_hist, NN_px_t = NN_px_no_pol(seed, lam, angles, 
                                                                  target, pol, 
                                                                  layers, options, 
                                                                  sim_dtype, geo_dtype, device)
    NN_px_time = time.time() - t
    plot_final_design(NN_px_des, "NN + px ", f"{res_folder}{label}_NNpx_design.png")
    animate_history(NN_px_des_hist, f"{res_folder}{label}_NNpx_ani.gif")
    plot_otf_perfomance_single_pol(NN_px_t, target, angles, "NNpx Optimisation",
                                 f"{res_folder}{label}_NNpx_performance.png")
    plot_cost(NN_px_cost, "NNpx Optimisation", f"{res_folder}{label}_NNpx_cost_evolution.png")

    # ========================================================================================================
    # Run latent matrix pixel optimisation
    LMpx_times = []
    LMpx_designs = []
    LMpx_costs = []
    LMpx_des_hists = []
    LMpx_ts = []
    
    for bl in blur:
        options["blur radius"] = bl
        if bl is not None:
            lab = f"{bl / lam:.0%}"
        else:
            lab = "0%"
        t = time.time()
        LMpx_des, LMpx_cost, LMpx_des_hist, LMpx_t = LMpx_no_pol(seed, lam, angles, 
                                                                 target, pol, 
                                                                 layers, options, 
                                                                 sim_dtype, geo_dtype, device)
        LMpx_times.append(time.time() - t)
        LMpx_designs.append(LMpx_des)
        LMpx_costs.append(LMpx_cost)
        LMpx_des_hists.append(LMpx_des_hist)
        LMpx_ts.append(LMpx_t)

        plot_final_design(LMpx_des, f"LMpx {lab} blur", f"{res_folder}{label}_LMpx_{lab}_design.png")
        animate_history(LMpx_des_hist, f"{res_folder}{label}_LMpx_{lab}_ani.gif")
        plot_otf_perfomance_single_pol(LMpx_t, target, angles, f"LMpx {lab} blur",
                                       f"{res_folder}{label}_LMpx_{lab}_performance.png")
        plot_cost(LMpx_cost, f"LMpx {lab} blur", f"{res_folder}{label}_LMpx_{lab}_cost_evolution.png")

    # ========================================================================================================
    # Run constrained pixel optimisation
    px_times = []
    px_designs = []
    px_costs = []
    px_des_hists = []
    px_ts = []
    
    for bl in blur:
        options["blur radius"] = bl
        if bl is not None:
            lab = f"{bl / lam:.0%}"
        else:
            lab = "0%"
        t = time.time()
        px_des, px_cost, px_des_hist, px_t = px_no_pol(seed, lam, angles, 
                                                       target, pol, 
                                                       layers, options, 
                                                       sim_dtype, geo_dtype, device)
        px_times.append(time.time() - t)
        px_designs.append(px_des)
        px_costs.append(px_cost)
        px_des_hists.append(px_des_hist)
        px_ts.append(px_t)

        plot_final_design(px_des, f"px {lab} blur", f"{res_folder}{label}_px_{lab}_design.png")
        animate_history(px_des_hist, f"{res_folder}{label}_px_{lab}_ani.gif")
        plot_otf_perfomance_single_pol(px_t, target, angles, f"px {lab} blur",
                                       f"{res_folder}{label}_px_{lab}_performance.png")
        plot_cost(px_cost, f"px {lab} blur", f"{res_folder}{label}_px_{lab}_cost_evolution.png")

    # ========================================================================================================
    # Save everything into a pickle files
    with open(f"{res_folder}{label}_NN_data.pickle", "wb") as file:
        pkl_data = {
            "blur": blur,
            "cost history": NN_cost,
            "final designs": NN_des,
            #"design history": NN_des_hist,
            "perf": NN_t,
            "target": target,
            "angles": angles,
        }
        pickle.dump(pkl_data, file)

    with open(f"{res_folder}{label}_NNpx_data.pickle", "wb") as file:
        pkl_data = {
            "blur": blur,
            "cost history": NN_px_cost,
            "final designs": NN_px_des,
            #"design history": NN_px_hists,
            "perf": NN_px_t,
            "target": target,
            "angles": angles,
        }
        pickle.dump(pkl_data, file)

    with open(f"{res_folder}{label}_LMpx_data.pickle", "wb") as file:
        pkl_data = {
            "blur": blur,
            "cost history": LMpx_costs,
            "final designs": LMpx_designs,
            #"design history": LMpx_des_hists,
            "perf": LMpx_ts,
            "target": target,
            "angles": angles,
        }
        pickle.dump(pkl_data, file)

    with open(f"{res_folder}{label}_px_data.pickle", "wb") as file:
        pkl_data = {
            "blur": blur,
            "cost history": px_costs,
            "final designs": px_designs,
            #"design history": px_des_hists,
            "perf": px_ts,
            "target": target,
            "angles": angles,
        }
        pickle.dump(pkl_data, file)

    # ========================================================================================================
    # Calculate the benchmark parameters
    NN_final_cost = NN_cost[-1]
    NN_px_final_cost = NN_px_cost[-1]
    LMpx_final_cost = [i[-1] for i in LMpx_costs]
    px_final_cost = [i[-1] for i in px_costs]

    NN_fs_solid, NN_fs_void = length_scale(NN_des > 0.5, period[0]/NN_des.shape[0])
    NN_px_fs_solid, NN_px_fs_void = length_scale(NN_px_des > 0.5, period[0]/NN_px_des.shape[0])
    LMpx_fs_solid = []
    LMpx_fs_void = []
    px_fs_solid = []
    px_fs_void = []
    for i in range(len(blur)):
        fssolid, fssvoid = length_scale(LMpx_designs[i] > 0.5, period[0]/LMpx_designs[i].shape[0])
        LMpx_fs_solid.append(fssolid)
        LMpx_fs_void.append(fssvoid)
        fssolid, fssvoid = length_scale(px_designs[i] > 0.5, period[0]/px_designs[i].shape[0])
        px_fs_solid.append(fssolid)
        px_fs_void.append(fssvoid)
        
    NN_conv_time = convergence_time(NN_cost)
    NN_px_conv_time = convergence_time(NN_px_cost)
    LMpx_conv_time = []
    px_conv_time = []
    for i in range(len(blur)):
        LMpx_conv_time.append(convergence_time(LMpx_costs[i]))
        px_conv_time.append(convergence_time(px_costs[i]))

    return ((NN_final_cost, NN_time, NN_fs_solid, NN_fs_void, NN_conv_time),
            (NN_px_final_cost, NN_px_time, NN_px_fs_solid, NN_px_fs_void, NN_px_conv_time),
            (LMpx_final_cost, LMpx_times, LMpx_fs_solid, LMpx_fs_void, LMpx_conv_time),
            (px_final_cost, px_times, px_fs_solid, px_fs_void, px_conv_time))