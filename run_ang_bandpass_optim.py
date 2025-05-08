import torch
import numpy as np
import pandas as pd
import copy

from Benchmarks.angular_ang_bp_benchmark import run_pol_dependent_ang_benchmark, run_pol_ind_ang_benchmark
from utils.material import aSi

def add_results_to_dicts(NNdict, NNpxdict, LMpxdict, pxdict, results, label, blur, wl, na, period):
    NN_res, NNpx_res, LMpx_res, px_res = results

    # Add NN results
    NNdict["Label"].append("Neural Network")
    NNdict["Benchmark"].append(label)
    NNdict["Wavelength (nm)"].append(wl)
    NNdict["NA"].append(na)
    NNdict["Final cost value"].append(NN_res[0])
    NNdict["Time (s)"].append(NN_res[1])
    NNdict["Min feature size solid (nm)"].append(NN_res[2])
    NNdict["Min Feature size void (nm)"].append(NN_res[3])
    NNdict["Convergence time (# iterations)"].append(NN_res[4])
    NNdict["Px (nm)"].append(period[0])
    NNdict["Py (nm)"].append(period[1])

    # Add NNpx results
    NNpxdict["Label"].append("Neural Network")
    NNpxdict["Benchmark"].append(label)
    NNpxdict["Wavelength (nm)"].append(wl)
    NNpxdict["NA"].append(na)
    NNpxdict["Final cost value"].append(NNpx_res[0])
    NNpxdict["Time (s)"].append(NNpx_res[1])
    NNpxdict["Min feature size solid (nm)"].append(NNpx_res[2])
    NNpxdict["Min Feature size void (nm)"].append(NNpx_res[3])
    NNpxdict["Convergence time (# iterations)"].append(NNpx_res[4])
    NNpxdict["Px (nm)"].append(period[0])
    NNpxdict["Py (nm)"].append(period[1])

    # Do pixel based results
    for i in range(len(blur)):
        LMpxdict[i]["Label"].append(f"LMpx {blur[i]}% blur")
        LMpxdict[i]["Benchmark"].append(label)
        LMpxdict[i]["Wavelength (nm)"].append(wl)
        LMpxdict[i]["NA"].append(na)
        LMpxdict[i]["Final cost value"].append(LMpx_res[0][i])
        LMpxdict[i]["Time (s)"].append(LMpx_res[1][i])
        LMpxdict[i]["Min feature size solid (nm)"].append(LMpx_res[2][i])
        LMpxdict[i]["Min Feature size void (nm)"].append(LMpx_res[3][i])
        LMpxdict[i]["Convergence time (# iterations)"].append(LMpx_res[4][i])
        LMpxdict[i]["Px (nm)"].append(period[0])
        LMpxdict[i]["Py (nm)"].append(period[1])

        pxdict[i]["Label"].append(f"px {blur[i]}% blur")
        pxdict[i]["Benchmark"].append(label)
        pxdict[i]["Wavelength (nm)"].append(wl)
        pxdict[i]["NA"].append(na)
        pxdict[i]["Final cost value"].append(px_res[0][i])
        pxdict[i]["Time (s)"].append(px_res[1][i])
        pxdict[i]["Min feature size solid (nm)"].append(px_res[2][i])
        pxdict[i]["Min Feature size void (nm)"].append(px_res[3][i])
        pxdict[i]["Convergence time (# iterations)"].append(px_res[4][i])
        pxdict[i]["Px (nm)"].append(period[0])
        pxdict[i]["Py (nm)"].append(period[1])
    return

def save_results(NN, NNpx, LMpx, px, blur_level, csv_folder):
    NN_OTF_df = pd.DataFrame(NN)
    NN_px_OTF_df = pd.DataFrame(NNpx)
    LMpx_OTF_df = [pd.DataFrame(LMpx[i]) for i in range(len(blur_level))]
    px_OTF_df = [pd.DataFrame(px[i]) for i in range(len(blur_level))]

    NN_OTF_df.to_csv(f"{csv_folder}NN_ang_bandpass.csv")
    NN_px_OTF_df.to_csv(f"{csv_folder}NNpx_ang_bandpass.csv")
    for i in range(len(blur_level)):
        LMpx_OTF_df[i].to_csv(f"{csv_folder}LMpx_blur{blur_level[i]}_ang_bandpass.csv")
        px_OTF_df[i].to_csv(f"{csv_folder}px_blur{blur_level[i]}_ang_bandpass.csv")

if __name__ == "__main__":

    #torch.use_deterministic_algorithms(True)

    result_folder = "./Benchmark_results/ang_bandpass/"
    csv_folder = "./Benchmark_results/ang_bandpass/"

    N_ANGLES = 11
    
    # ========================================================================================================
    blur_level = [None, 0.05, 0.1, 0.2] 

    lams = [1550]
    mats = [aSi]
    thicknesses = [550]
    periods = [(1500, 1500)]
    labels = ["1550nm_aSi_ang_bandpass"]

    num_aperture = [0.15]

    # ========================================================================================================
    # Create dictionaries to store results
    results_dict = {
        "Label": [],
        "Benchmark": [],
        "Wavelength (nm)": [],
        "NA": [],
        "Px (nm)": [],
        "Py (nm)": [],
        "Final cost value": [],
        "Time (s)": [],
        "Min feature size solid (nm)": [],
        "Min Feature size void (nm)": [],
        "Convergence time (# iterations)": [],
    }
    NN_res = copy.deepcopy(results_dict)
    NNpx_res = copy.deepcopy(results_dict)
    LMpx_res = [copy.deepcopy(results_dict) for i in blur_level]
    px_res = [copy.deepcopy(results_dict) for i in blur_level]

    dev = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {dev} device")

    # ===================================================================================================================
    print(f"Running s and p bandpass")

    for i in range(len(lams)):

        # Determine the target OTFs
        angles = torch.linspace(-15, 15, N_ANGLES)
        target1 = torch.zeros_like(angles)
        target1[torch.logical_and(angles >= 5, angles <= 10)] = 1.0
        target2 = torch.flip(target1)

        res = run_pol_dependent_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, target1,
                                              target2, periods[i], [None], 
                                              f"OTF_pol_bandpass_{labels[i]}_ang_bandpass", result_folder, 
                                              blur_level)
        # Save benchmark results to dictionary
        add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "s, p pol", blur_level, lams[i], 0.17, periods[i])

    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)

    print(f"Running pol insensitive bandpass")

    for i in range(len(lams)):

        # Determine the target OTFs
        angles = torch.linspace(-20, 20, N_ANGLES)
        target = torch.zeros_like(angles)
        target[torch.logical_and(angles >= 5, angles <= 10)] = 1.0

        res = run_pol_dependent_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, target,
                                              target, periods[i], [None], 
                                              f"OTF_pol_insens_{labels[i]}_ang_bandpass", result_folder, 
                                              blur_level)
        # Save benchmark results to dictionary
        add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "pol insensitive", blur_level, lams[i], 0.17, periods[i])

    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)

    print(f"Running s bandpass")

    for i in range(len(lams)):

        # Determine the target OTFs
        angles = torch.linspace(-15, 15, N_ANGLES)
        target = torch.zeros_like(angles)
        target[torch.logical_and(angles >= 5, angles <= 10)] = 1.0

        res = run_pol_ind_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, target,
                                        "s", periods[i], [None], 
                                        f"OTF_s_pol_{labels[i]}_ang_bandpass", result_folder, 
                                        blur_level)
        # Save benchmark results to dictionary
        add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "s polarisation", blur_level, lams[i], 0.17, periods[i])

    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)

