import torch
import numpy as np

from Benchmarks.angular_benchmark import quadratic_OTF, identity_OTF, run_ang_benchmark
from utils.material import aSi, Si3N4

if __name__ == "__main__":

    # OTF Benchmarks
    N_ANGLES = 5

    lams = torch.tensor([600, 800, 900, 1550])
    mats = [Si3N4, aSi, aSi, aSi]
    thicknesses = [500, 300, 300, 400]
    periods = [(400, 400), (500, 500), (550, 550), (850, 850)]
    labels = ["600nm_SiN", "800nm_aSi", "900nm_aSi", "1550nm_aSi"]

    num_aperture = [0.05, 0.1, 0.2]
    result_folder = "./Benchmark_results/"

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
                              f"pol_insensitive_{labels[i]}_NA{na*100}", result_folder)

    print("Angle Sensitive 1")
    for i in range(len(lams)):

        for na in num_aperture:
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = identity_OTF(angles, 0.8)

            run_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                              targetp, periods[i], [None], 
                              f"pol_sensitive_s_{labels[i]}_NA{na*100}", result_folder)
            
    print("Angle Sensitive 2")
    for i in range(len(lams)):

        for na in num_aperture:
            # Determine the target OTFs
            angles = torch.linspace(0, np.rad2deg(np.arcsin(na)), N_ANGLES)
            targets = quadratic_OTF(angles, na)
            targetp = identity_OTF(angles, 0.8)

            run_ang_benchmark(lams[i], mats[i], thicknesses[i], angles, targets,
                              targetp, periods[i], [None], 
                              f"pol_sensitive_p_{labels[i]}_NA{na*100}", result_folder)
            

    # Spectral benchmarks