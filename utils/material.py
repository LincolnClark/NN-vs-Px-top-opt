import numpy as np
import pandas as pd

j = complex(0,1)

class Material:
    
    def __init__(self, n, k, label = None):
        """
        Class for holding material properties
        VARIABLES:
        n - real part of the refractive index. Function of wavelength
        k - imaginary part of refractive index. Function of wavelength
        label - label for the data
        """
        self.n = n
        self.k = k
        self.label = label

        return

    def N(self, lam):
        """Returns the complex valued refractive index as a function of wavelength. N = n + ik convention"""
        return self.n(lam) + j*self.k(lam)
    
    def eps(self, lam):
        """Return the complex permittivity"""
        return np.power(self.N(lam), 2)

def void_n(lam):
    return 1
def void_k(lam):
    return 0
void = Material(void_n, void_k, label = "n=1 void material")
    
Si_data = pd.read_csv("./utils/materials/Si.csv")
def Si_n(lam):
    wl = Si_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, Si_data["n"])
def Si_k(lam):
    wl = Si_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, Si_data["k"])
Si = Material(Si_n, Si_k, label = "Schinke et al., DOI: https://doi.org/10.1063/1.4923379")

aSi_data = pd.read_csv("./utils/materials/aSi.csv")
def aSi_n(lam):
    wl = aSi_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, aSi_data["n"])
def aSi_k(lam):
    wl = aSi_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, aSi_data["k"])
aSi = Material(aSi_n, aSi_k, label = "Pierce and Spicer, DOI: https://doi.org/10.1103/PhysRevB.5.3017")

SiO2_data = pd.read_csv("./utils/materials/SiO2.csv")
def SiO2_n(lam):
    wl = SiO2_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, SiO2_data["n"])
def SiO2_k(lam):
    return 0
SiO2 = Material(SiO2_n, SiO2_k, label = "Malitson, DOI: https://doi.org/10.1364/JOSA.55.001205")

Au_data = pd.read_csv("./utils/materials/Au.csv")
def Au_n(lam):
    wl = Au_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, Au_data["n"])
def Au_k(lam):
    wl = Au_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, Au_data["k"])
Au = Material(Au_n, Au_k, label = "Johnson and Christy, DOI: https://doi.org/10.1103/PhysRevB.6.4370")

Ag_data = pd.read_csv("./utils/materials/Ag.csv")
def Ag_n(lam):
    wl = Ag_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, Ag_data["n"])
def Ag_k(lam):
    wl = Ag_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, Ag_data["k"])
Ag = Material(Ag_n, Ag_k, label = "Johnson and Christy, DOI: https://doi.org/10.1103/PhysRevB.6.4370")

TiO2_data = pd.read_csv("./utils/materials/TiO2.csv")
def TiO2_n(lam):
    wl = TiO2_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, TiO2_data["n"])
def TiO2_k(lam):
    wl = TiO2_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, TiO2_data["k"])
TiO2 = Material(TiO2_n, TiO2_k, label = "Zhukovsky et al., DOI: https://doi.org/10.1103/PhysRevLett.115.177402")

Si3N4_data = pd.read_csv("./utils/materials/Si3N4.csv")
def Si3N4_n(lam):
    wl = Si3N4_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, Si3N4_data["n"])
def Si3N4_k(lam):
    return 0
Si3N4 = Material(Si3N4_n, Si3N4_k, label = "Luke et al. DOI: https://doi.org/10.1364/OL.40.004823")

Al_data = pd.read_csv("./utils/materials/Al.csv")
def Al_n(lam):
    wl = Al_data["wl"].to_numpy()
    return np.interp(lam, wl * 1000, Al_data["n"])
def Al_k(lam):
    wl = Al_data["wl"].to_numpy()
    return np.interp(lam, wl * 1000, Al_data["k"])
Al = Material(Al_n, Al_k, label = "Rakic DOI: https://doi.org/10.1364/AO.34.004755")

VO2_data = pd.read_csv("./utils/materials/VO2 Wan et al.csv")
def VO2_n_met(lam):
    wl = VO2_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, VO2_data["n hot"])
def VO2_n_ins(lam):
    wl = VO2_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, VO2_data["n cold"])
def VO2_k_met(lam):
    wl = VO2_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, VO2_data["k hot"])
def VO2_k_ins(lam):
    wl = VO2_data["wl"].to_numpy()
    return np.interp(lam, wl*1000, VO2_data["k cold"])

VO2_met = Material(VO2_n_met, VO2_k_met, label = "Metallic, Wan et al, DOI: https://doi.org/10.48550/arXiv.1901.02517")
VO2_ins = Material(VO2_n_ins, VO2_k_ins, label = "Insulating, Wan et al. DOI: https://doi.org/10.48550/arXiv.1901.02517")