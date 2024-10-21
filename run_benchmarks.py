import torch
import numpy as np
import pandas as pd

from Benchmarks.angular_benchmark import quadratic_OTF, identity_OTF, run_ang_benchmark
from Benchmarks.spectral_benchmark import bandpass, bandstop, constant_value, run_spec_benchmark
from utils.material import aSi, Si3N4

def add_results_to_dict(results, dic, label):
    dic["Benchmark"].append(label)
    dic["Final cost value"].append(results[0])
    dic["Time (s)"].append(results[1])
    dic["Min feature size solid (nm)"].append(results[2])
    dic["Min Feature size void (nm)"].append(results[3])
    dic["Convergence time (# iterations)"].append(results[4])
    dic["Binarisation level"].append(results[5])
    return

if __name__ == "__main__":

    result_folder = "./Benchmark_results/plots/"
    csv_folder = "./Benchmark_results/"

    # OTF Benchmarks
    N_ANGLES = 5

    lams = [650.0, 900.0]
    mats = [Si3N4, aSi]
    thicknesses = [600, 300]
    periods = [(450, 450), (550, 550)]
    labels = ["650nm_SiN", "900nm_aSi"]

    num_aperture = [0.1, 0.2]

    # Create dictionaries to store results
    NN_OTF_results = {
        "Benchmark": [],
        "Final cost value": [],
        "Time (s)": [],
        "Min feature size solid (nm)": [],
        "Min Feature size void (nm)": [],
        "Convergence time (# iterations)": [],
        "Binarisation level": []
    }
    px_OTF_results = {
        "Benchmark": [],
        "Final cost value": [],
        "Time (s)": [],
        "Min feature size solid (nm)": [],
        "Min Feature size void (nm)": [],
        "Convergence time (# iterations)": [],
        "Binarisation level": []
    }

    print("Angle Insensitive")
    for i in range(len(lams)):

        for na in num_aperture:
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = quadratic_OTF(angles, na)

            NN_res, px_res = run_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                                               targetp, periods[i], [None], 
                                               f"OTF_pol_insensitive_{labels[i]}_NA{na}", result_folder)
            # Save benchmark results to dictionary
            add_results_to_dict(NN_res, NN_OTF_results, f"{labels[i]}_NA{na}_pol_insens")
            add_results_to_dict(px_res, px_OTF_results, f"{labels[i]}_NA{na}_pol_insens")

    print("Angle Sensitive 1")
    for i in range(len(lams)):

        for na in num_aperture:
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = identity_OTF(angles, 0.8)

            NN_res, px_res = run_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                                               targetp, periods[i], [None], 
                                               f"OTF_pol_sensitive_s_{labels[i]}_NA{na}", result_folder)
            # Save benchmark results to dictionary
            add_results_to_dict(NN_res, NN_OTF_results, f"{labels[i]}_NA{na}_pol_sens_s")
            add_results_to_dict(px_res, px_OTF_results, f"{labels[i]}_NA{na}_pol_sens_s")
            
    print("Angle Sensitive 2")
    for i in range(len(lams)):

        for na in num_aperture:
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = identity_OTF(angles, 0.8)

            NN_res, px_res = run_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                                               targetp, periods[i], [None], 
                                               f"OTF_pol_sensitive_p_{labels[i]}_NA{na}", result_folder)
            # Save benchmark results to dictionary
            add_results_to_dict(NN_res, NN_OTF_results, f"{labels[i]}_NA{na}_pol_sens_p")
            add_results_to_dict(px_res, px_OTF_results, f"{labels[i]}_NA{na}_pol_sens_p")

    # Save results to csv file
    NN_OTF_df = pd.DataFrame(NN_OTF_results)
    px_OTF_df = pd.DataFrame(px_OTF_results)
    NN_OTF_df.to_csv(f"{csv_folder}NN_OTF_benchamrk.csv")
    NN_OTF_df.to_csv(f"{csv_folder}px_OTF_benchamrk.csv")

    # =================================================================================================================
    # Spectral benchmarks
    N_WL = 21
    mats = [Si3N4, aSi]
    thicknesses = [600, 400]
    periods = [(450, 450), (850, 850)]
    labels = ["650nm_SiN", "1550nm_aSi"]

    lam = [650, 1550]
    lam_optim = [(500, 800), [1200, 1700]]
    widths = [50, 50]

     # Create dictionaries to store results
    NN_spectrum_results = {
        "Benchmark": [],
        "Final cost value": [],
        "Time (s)": [],
        "Min feature size solid (nm)": [],
        "Min Feature size void (nm)": [],
        "Convergence time (# iterations)": [],
        "Binarisation level": []
    }
    px_spectrum_results = {
        "Benchmark": [],
        "Final cost value": [],
        "Time (s)": [],
        "Min feature size solid (nm)": [],
        "Min Feature size void (nm)": [],
        "Convergence time (# iterations)": [],
        "Binarisation level": []
    }

    print("Spectral pol insensitive bandpass")
    for i in range(len(mats)):
        # Determine the target Spectrums
        wavelengths = torch.linspace(lam_optim[i][0], lam_optim[i][1], N_WL)
        targets = bandpass(wavelengths, 0.9, lam[i], widths[i])
        targetp = bandpass(wavelengths, 0.9, lam[i], widths[i])

        NN_res, px_res = run_spec_benchmark(mats[i], thicknesses[i], wavelengths, targets,
                                            targetp, periods[i], [None], 
                                            f"Spectrum_pol_insensitive_{labels[i]}_bandpass{lam[i]}", result_folder)
        # Save benchmark results to dictionary
        add_results_to_dict(NN_res, NN_spectrum_results, f"{labels[i]}_bandpass{lam[i]}_pol_insens")
        add_results_to_dict(px_res, px_spectrum_results, f"{labels[i]}_bandpass{lam[i]}_pol_insens")
        
    print("Spectral pol sensitive bandpass")
    for i in range(len(mats)):
        # Determine the target Spectrums
        wavelengths = torch.linspace(lam_optim[i][0], lam_optim[i][1], N_WL)
        targets = bandpass(wavelengths, 0.9, lam[i], widths[i])
        targetp = constant_value(wavelengths, 0.9)

        NN_res, px_res = run_spec_benchmark(mats[i], thicknesses[i], wavelengths, targets,
                                            targetp, periods[i], [None], 
                                            f"Spectrum_pol_sensitive_s_{labels[i]}_bandpass{lam[i]}", result_folder)
        # Save benchmark results to dictionary
        add_results_to_dict(NN_res, NN_spectrum_results, f"{labels[i]}_bandpass{lam[i]}_pol_sens_s")
        add_results_to_dict(px_res, px_spectrum_results, f"{labels[i]}_bandpass{lam[i]}_pol_sens_s")

    # Save results to csv file
    NN_spectral_df = pd.DataFrame(NN_spectrum_results)
    px_spectral_df = pd.DataFrame(px_spectrum_results)
    NN_spectral_df.to_csv(f"{csv_folder}NN_spectral_benchamrk.csv")
    NN_spectral_df.to_csv(f"{csv_folder}px_spectral_benchamrk.csv")