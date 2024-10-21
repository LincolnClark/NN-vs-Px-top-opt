import torch
import numpy as np

from Benchmarks.angular_benchmark import quadratic_OTF, identity_OTF, run_ang_benchmark
from Benchmarks.spectral_benchmark import bandpass, bandstop, constant_value, run_spec_benchmark
from utils.material import aSi, Si3N4

if __name__ == "__main__":

    result_folder = "./Benchmark_results/"

    # OTF Benchmarks
    N_ANGLES = 5

    lams = [650.0, 900.0]
    mats = [Si3N4, aSi]
    thicknesses = [600, 300]
    periods = [(450, 450), (550, 550)]
    labels = ["650nm_SiN", "900nm_aSi"]

    num_aperture = [0.1, 0.2]

    # Angle insensitive optimisation
    print("Angle Insensitive")
    for i in range(len(lams)):

        for na in num_aperture:
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = quadratic_OTF(angles, na)

            run_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                              targetp, periods[i], [None], 
                              f"OTF_pol_insensitive_{labels[i]}_NA{na}", result_folder)

    print("Angle Sensitive 1")
    for i in range(len(lams)):

        for na in num_aperture:
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = identity_OTF(angles, 0.8)

            run_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                              targetp, periods[i], [None], 
                              f"OTF_pol_sensitive_s_{labels[i]}_NA{na}", result_folder)
            
    print("Angle Sensitive 2")
    for i in range(len(lams)):

        for na in num_aperture:
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = identity_OTF(angles, 0.8)

            run_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                              targetp, periods[i], [None], 
                              f"OTF_pol_sensitive_p_{labels[i]}_NA{na}", result_folder)
            
    # Spectral benchmarks
    N_WL = 21
    mats = [Si3N4, aSi]
    thicknesses = [600, 400]
    periods = [(450, 450), (850, 850)]
    labels = ["650nm_SiN", "1550nm_aSi"]

    lam = [650, 1550]
    lam_optim = [(500, 800), [1200, 1700]]
    widths = [50, 50]

    print("Spectral pol insensitive bandpass")
    for i in range(len(mats)):
        # Determine the target Spectrums
        wavelengths = torch.linspace(lam_optim[i][0], lam_optim[i][1], N_WL)
        targets = bandpass(wavelengths, 0.9, lam[i], widths[i])
        targetp = bandpass(wavelengths, 0.9, lam[i], widths[i])

        run_spec_benchmark(mats[i], thicknesses[i], wavelengths, targets,
                           targetp, periods[i], [None], 
                           f"Spectrum_pol_insensitive_{labels[i]}_bandpass{lam[i]}", result_folder)
        
    print("Spectral pol sensitive bandpass")
    for i in range(len(mats)):
        # Determine the target Spectrums
        wavelengths = torch.linspace(lam_optim[i][0], lam_optim[i][1], N_WL)
        targets = bandpass(wavelengths, 0.9, lam[i], widths[i])
        targetp = constant_value(wavelengths, 0.9)

        run_spec_benchmark(mats[i], thicknesses[i], wavelengths, targets,
                           targetp, periods[i], [None], 
                           f"Spectrum_pol_sensitive_s_{labels[i]}_bandpass{lam[i]}", result_folder)
            