import torch
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from Benchmarks.spectral_benchmark import bandpass, bandstop, constant_value, run_pol_dependent_spec_benchmark, run_pol_ind_spec_benchmark
from utils.material import aSi

def add_results_to_dicts(NNdict, NNpxdict, LMpxdict, pxdict, results, label, blur, wl, period, width):
    NN_res, NNpx_res, LMpx_res, px_res = results

    # Add NN results
    NNdict["Label"].append("Neural Network")
    NNdict["Benchmark"].append(label)
    NNdict["Wavelength (nm)"].append(wl)
    NNdict["Final cost value"].append(NN_res[0])
    NNdict["Time (s)"].append(NN_res[1])
    NNdict["Min feature size solid (nm)"].append(NN_res[2])
    NNdict["Min Feature size void (nm)"].append(NN_res[3])
    NNdict["Convergence time (# iterations)"].append(NN_res[4])
    NNdict["Px (nm)"].append(period[0])
    NNdict["Py (nm)"].append(period[1])
    NNdict["Width (nm)"].append(width)

    # Add NNpx results
    NNpxdict["Label"].append("Neural Network")
    NNpxdict["Benchmark"].append(label)
    NNpxdict["Wavelength (nm)"].append(wl)
    NNpxdict["Final cost value"].append(NNpx_res[0])
    NNpxdict["Time (s)"].append(NNpx_res[1])
    NNpxdict["Min feature size solid (nm)"].append(NNpx_res[2])
    NNpxdict["Min Feature size void (nm)"].append(NNpx_res[3])
    NNpxdict["Convergence time (# iterations)"].append(NNpx_res[4])
    NNpxdict["Px (nm)"].append(period[0])
    NNpxdict["Py (nm)"].append(period[1])
    NNpxdict["Width (nm)"].append(width)

    # Do pixel based results
    for i in range(len(blur)):
        LMpxdict[i]["Label"].append(f"LMpx {blur[i]}nm blur")
        LMpxdict[i]["Benchmark"].append(label)
        LMpxdict[i]["Wavelength (nm)"].append(wl)
        LMpxdict[i]["Final cost value"].append(LMpx_res[0][i])
        LMpxdict[i]["Time (s)"].append(LMpx_res[1][i])
        LMpxdict[i]["Min feature size solid (nm)"].append(LMpx_res[2][i])
        LMpxdict[i]["Min Feature size void (nm)"].append(LMpx_res[3][i])
        LMpxdict[i]["Convergence time (# iterations)"].append(LMpx_res[4][i])
        LMpxdict[i]["Px (nm)"].append(period[0])
        LMpxdict[i]["Py (nm)"].append(period[1])
        LMpxdict[i]["Width (nm)"].append(width)

        pxdict[i]["Label"].append(f"px {blur[i]}nm blur")
        pxdict[i]["Benchmark"].append(label)
        pxdict[i]["Wavelength (nm)"].append(wl)
        pxdict[i]["Final cost value"].append(px_res[0][i])
        pxdict[i]["Time (s)"].append(px_res[1][i])
        pxdict[i]["Min feature size solid (nm)"].append(px_res[2][i])
        pxdict[i]["Min Feature size void (nm)"].append(px_res[3][i])
        pxdict[i]["Convergence time (# iterations)"].append(px_res[4][i])
        pxdict[i]["Px (nm)"].append(period[0])
        pxdict[i]["Py (nm)"].append(period[1])
        pxdict[i]["Width (nm)"].append(width)
    return

def save_results(NN, NNpx, LMpx, px, blur_level, csv_folder):
    NN_spec_df = pd.DataFrame(NN)
    NN_px_spec_df = pd.DataFrame(NNpx)
    LMpx_spec_df = [pd.DataFrame(LMpx[i]) for i in range(len(blur_level))]
    px_spec_df = [pd.DataFrame(px[i]) for i in range(len(blur_level))]

    NN_spec_df.to_csv(f"{csv_folder}NN_spectral_benchmark.csv")
    NN_px_spec_df.to_csv(f"{csv_folder}NNpx_spectral_benchmark.csv")
    for i in range(len(blur_level)):
        LMpx_spec_df[i].to_csv(f"{csv_folder}LMpx_blur{blur_level[i]}_spectral_benchmark.csv")
        px_spec_df[i].to_csv(f"{csv_folder}px_blur{blur_level[i]}_spectral_benchmark.csv")

if __name__ == "__main__":

    #torch.use_deterministic_algorithms(True)

    result_folder = "./Benchmark_results/plots/"
    csv_folder = "./Benchmark_results/"

    # Spectral Benchmarks
    N_WL = 15
    
    # Real benchmark set
    """"""

    # Test set
    
    lams = [1550]
    mats = [aSi]
    thicknesses = [550]
    periods = [(1000, 1000)]
    labels = ["1550nm_aSi"]

    dl = [200, 400]
    widths = [50, 100]
    blur_level = [None, 45]

    # Create dictionaries to store results
    results_dict = {
        "Label": [],
        "Benchmark": [],
        "Wavelength (nm)": [],
        "Px (nm)": [],
        "Py (nm)": [],
        "Width (nm)": [],
        "Final cost value": [],
        "Time (s)": [],
        "Min feature size solid (nm)": [],
        "Min Feature size void (nm)": [],
        "Convergence time (# iterations)": []
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

        for j in range(len(widths)):
            print(f"Running s pol optimisation for {lams[i]} nm and width of {widths[j]} nm")

            
            # Determine the target spectrum
            wavelengths = torch.linspace(lams[i] - dl[j], lams[i] + dl[j], N_WL)
            target = bandpass(wavelengths, 1.0, lams[i], widths[j])

            res = run_pol_ind_spec_benchmark(lams[i], mats[i], thicknesses[i], wavelengths, target,
                                            "s", periods[i], [None], 
                                            f"spectral_s_pol_{labels[i]}_bandpass_wdth{widths[j]}", result_folder, 
                                            blur_level)
            # Save benchmark results to dictionary
            add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "s polarisation bandpass", blur_level, lams[i], periods[i], widths[j])

    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)
            
    for i in range(len(lams)):

        for j in range(len(widths)):
            print(f"Running pol independent optimisation for {lams[i]} nm and width of {widths[j]} nm")

            
            # Determine the target spectrum
            wavelengths = torch.linspace(lams[i] - dl[j], lams[i] + dl[j], N_WL)
            targets = bandpass(wavelengths, 1, lams[i], widths[j])
            targetp = bandpass(wavelengths, 1, lams[i], widths[j])

            res = run_pol_dependent_spec_benchmark(lams[i], mats[i], thicknesses[i], wavelengths, targets,
                                                   targetp, periods[i], [None], 
                                                   f"spectral_pol_ind_{labels[i]}_bandpass_wdth{widths[j]}", result_folder, 
                                                   blur_level)
            # Save benchmark results to dictionary
            add_results_to_dicts(NN_res, NNpx_res, LMpx_res, px_res, res, "pol independent bandpass", blur_level, lams[i], periods[i], widths[j])

    # Update results csv
    save_results(NN_res, NNpx_res, LMpx_res, px_res, blur_level, csv_folder)