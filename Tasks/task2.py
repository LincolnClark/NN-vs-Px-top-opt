import torch
from utils.material import aSi, SiO2, void
   
seed = 39393939 # Starting seed for random number generation
lam = 900 # Wavelength (nm)

# Angles to optimise over
angles = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

# Target transmission for s and p polarised light
targets = torch.tensor([i**2/100 for i in angles])
targetp = torch.tensor([0.8 for i in angles])

# Setup layers, in case extra thin films involved
layers = [None]

options = {
        # Geometry and material
        "t": 300, # thickness of the design domain
        "mat 1": 1., # permittivity when density is 0
        "mat 2": aSi.eps(lam), # permittivity when density is 1
        "superstrate": 1., # Superstrate permittivity
        "substrate": SiO2.eps(lam), # substrate permittivity
        "Lx": 500, # Period along x
        "Ly": 500, # Period along y

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
        "edge blur": 8,
        "edge weight": 0.25,
        "start rob": 350,
        "rob weight": 0.25,

        # ADAM optimiser settings
        "alpha": 0.01, # max step size
        "alpha NN": 0.001,
        "beta 1": 0.9, # decay rate of 1st moment
        "beta 2": 0.999, # decay rate of 2nd moment
        "epsilon": 1e-6 # factor to avoid divison by 0
    }

sim_dtype = torch.complex64
geo_dtype = torch.float32