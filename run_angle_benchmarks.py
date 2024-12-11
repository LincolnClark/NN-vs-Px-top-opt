import torch
import numpy as np
import pandas as pd
import copy

from Benchmarks.angular_benchmark import quadratic_OTF, identity_OTF, run_pol_dependent_ang_benchmark, run_pol_ind_ang_benchmark
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
        LMpxdict[i]["Label"].append(f"LMpx {blur[i]}nm blur")
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

        pxdict[i]["Label"].append(f"px {blur[i]}nm blur")
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

    NN_OTF_df.to_csv(f"{csv_folder}NN_OTF_benchmark.csv")
    NN_px_OTF_df.to_csv(f"{csv_folder}NNpx_OTF_benchmark.csv")
    for i in range(len(blur_level)):
        LMpx_OTF_df[i].to_csv(f"{csv_folder}LMpx_blur{blur_level[i]}_OTF_benchmark.csv")
        px_OTF_df[i].to_csv(f"{csv_folder}px_blur{blur_level[i]}_OTF_benchmark.csv")

if __name__ == "__main__":

    #torch.use_deterministic_algorithms(True)

    result_folder = "./Benchmark_results/plots/"
    csv_folder = "./Benchmark_results/"

    # OTF Benchmarks
    N_ANGLES = 5
    
    # Real benchmark set
    """
    lams = [800, 900, 980, 1300, 1550]
    mats = [aSi, aSi, aSi, aSi, aSi]
    thicknesses = [250, 300, 330, 500, 550]
    periods = [(530, 530), (600, 600), (650, 650), (860, 860), (1000, 1000)]
    labels = ["800nm_aSi", "900nm_aSi", "980nm_aSi", "1300nm_aSi", "1550nm_aSi"]
    num_aperture = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    blur_level = [None, 30, 60]"""

    # Test set
    
    lams = [1550]
    mats = [aSi]
    thicknesses = [550]
    periods = [(1000, 1000)]
    labels = ["1550nm_aSi"]

    num_aperture = [0.15]
    blur_level = [None, 45]

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

    for i in range(len(lams)):

        for na in num_aperture:
            print(f"Running s pol optimisation for {lams[i]} nm and NA of {na}")

            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            target = quadratic_OTF(angles, na)

            res = run_pol_ind_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, target,
                                            "s", periods[i], [None], 
                                            f"OTF_s_pol_{labels[i]}_NA{na}", result_folder, 
                                            blur_level)
            # Save benchmark results to dictionary
            add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "s polarisation", blur_level, lams[i], na, periods[i])

    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)

    for i in range(len(lams)):

        for na in num_aperture:
            print(f"Running p pol optimisation for {lams[i]} nm and NA of {na}")
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            target = quadratic_OTF(angles, na)

            res = run_pol_ind_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, target,
                                            "p", periods[i], [None], 
                                            f"OTF_p_pol_{labels[i]}_NA{na}", result_folder, 
                                            blur_level)
            # Save benchmark results to dictionary
            add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "p polarisation", blur_level, lams[i], na, periods[i])
    
    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)

    for i in range(len(lams)):

        for na in num_aperture:
            print(f"Running pol insensitive optimisation for {lams[i]} nm and NA of {na}")
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = quadratic_OTF(angles, na)

            res = run_pol_dependent_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                                                  targetp, periods[i], [None], 
                                                  f"OTF_pol_insensitive_{labels[i]}_NA{na}", result_folder, 
                                                  blur_level)
            # Save benchmark results to dictionary
            add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "Polarisation Insensitive", blur_level, lams[i], na, periods[i])

    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)

    for i in range(len(lams)):

        for na in num_aperture:
            print(f"Running s pol quadratic, p pol constant optimisation for {lams[i]} nm and NA of {na}")
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = identity_OTF(angles, 0.8)

            res = run_pol_dependent_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                                                  targetp, periods[i], [None], 
                                                  f"OTF_pol_sens_s_{labels[i]}_NA{na}", result_folder, 
                                                  blur_level)
            # Save benchmark results to dictionary
            add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "s quadratic, p constant", blur_level, lams[i], na, periods[i])

    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)
            
    for i in range(len(lams)):

        for na in num_aperture:
            print(f"Running p pol quadratic, s pol constant optimisation for {lams[i]} nm and NA of {na}")
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targetp = quadratic_OTF(angles, na)
            targets = identity_OTF(angles, 0.8)

            res = run_pol_dependent_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                                                  targetp, periods[i], [None], 
                                                  f"OTF_pol_sens_p_{labels[i]}_NA{na}", result_folder, 
                                                  blur_level)
            # Save benchmark results to dictionary
            add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "s constant, p quadratic", blur_level, lams[i], na, periods[i])

    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)