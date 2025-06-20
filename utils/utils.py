import numpy as np
import torch
import torcwa
import matplotlib.pyplot as plt
import matplotlib.animation as ani

def ramp(num_iter, start, end, device):

    ramp = torch.ones(num_iter, device = device)
    ramp[:start] = 0.
    for i in range(start, end):
        ramp[i] = 1/(end - start)*i - start/(end - start)

    return ramp

#===========================================================================
# Calculate amplitude transmittance for an angle
#===========================================================================

def trans_at_angle_comp(layers, theta, phi, options, geom, sim_dtype):
    """Calculate the amplitude transmittance at specific angle
    INPUTS:
    layers - list of dictionaries describing the layers
              't' thickness of the layer
              'eps' permittivity distribution
    theta - elevation angle of incidence in degrees
    phi - azimuthal angle of incidence in degrees
    """

    order = [options["M"], options["N"]]
    lamb0 = torch.tensor(options["lam"], dtype = geom.dtype, 
                         device = geom.device)    # nm
    inc_ang = theta*(np.pi/180)   # radian
    azi_ang = phi*(np.pi/180)    # radian
    L = [options["Lx"], options["Ly"]]

    # Setup the simulation 
    sim = torcwa.rcwa(freq = 1/lamb0, order = order, L = L,
                      dtype = sim_dtype, device = geom.device)
    
    sim.add_input_layer(eps = options["superstrate"])
    sim.add_output_layer(eps = options["substrate"])
    sim.set_incident_angle(inc_ang = inc_ang, azi_ang = azi_ang)

    # Build the layers
    for i in range(len(layers)):
        sim.add_layer(thickness = layers[i]["t"], eps = layers[i]["eps"])
    
    sim.solve_global_smatrix()
    Sss = sim.S_parameters(orders = [0,0] , direction = 'forward', 
                           port='transmission' , polarization='ss', 
                           ref_order=[0,0])
    Ssp = sim.S_parameters(orders = [0,0] , direction = 'forward', 
                           port='transmission' , polarization='sp', 
                           ref_order=[0,0])
    Sps = sim.S_parameters(orders = [0,0] , direction = 'forward', 
                           port='transmission' , polarization='ps', 
                           ref_order=[0,0])
    Spp = sim.S_parameters(orders = [0,0] , direction = 'forward', 
                           port='transmission' , polarization='pp', 
                           ref_order=[0,0])

    return torch.abs(Sss), torch.abs(Spp)


#===========================================================================
# Filtering, projection and edge finding
#===========================================================================

def projection(gamma, beta, eta, dtype, device):
    """Projection funtion used in topoloigy optimisation, limits to a 
    Heaviside step function as beta->infinity
    INPUTS:
    gamma - values being modified
    beta - scaling factor, largr more step function like
    eta - position of the step (0.5 is good)
    """
    num = torch.tanh(beta*eta) + torch.tanh(beta*(gamma - eta))
    den = torch.tanh(beta*eta) + torch.tanh(beta*(1-eta))
    return num/den

def filter(A, r, xx, yy, dtype, device):
    # Create xx and yy on device
    xx = xx.to(device)
    yy = yy.to(device)

    # Create filter
    w = r - torch.sqrt(xx**2 + yy**2)
    w = w.long()
    w[w < 0] = 0

    # FFT both w and A
    w_f = torch.fft.fft2(w)
    A_f = torch.fft.fft2(A)

    # perform convolution in frequency space
    kap_f = w_f * A_f

    # Convert back to position space
    kap = torch.fft.fftshift(torch.fft.ifft2(kap_f))

    return torch.real(kap) / torch.sum(w).item()

def FFT(A):
    """Fourier transform of array A"""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(A)))

def IFFT(A):
    """Inverse Fourier transform of array A"""
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(A)))

#===========================================================================
# Optimiser
#===========================================================================

def update_with_adam(alpha, beta1, beta2, epsilon, gt, mt, vt, t, params):

    """https://arxiv.org/abs/1412.6980"""
    t = t + 1

    # Compute biased moments
    mtn = beta1*mt + (1-beta1)*gt
    vtn = beta2*vt + (1-beta2)*gt**2

    # Bias corrected moments (required due to initialisation of 0)
    mth = mtn / (1- beta1**t)
    vth = vtn / (1 - beta2**t)

    # Update parameters
    new_params = params - alpha*mth/(torch.sqrt(vth) + epsilon)
    return new_params, mtn, vtn

#===========================================================================
# Animate the density history
#===========================================================================

def animate_dens_history(hist, x, y, fname):
    """Create an animation of the density history"""

    # Setup the plot
    fig, ax = plt.subplots(figsize = (6, 5), dpi = 150)
    ax.set_aspect(1)
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$y$ (nm)")

    # Create list of artists
    artists = []
    for i in range(len(hist)):
        plot = ax.pcolormesh(x, y, hist[i], vmin = 0, vmax = 1,
                             shading = "auto", cmap = "gist_gray")
        txt = ax.text(0.5, 1.02, f"Iteration {i}", ha = "center", va = "bottom",
                      transform = ax.transAxes)
        artists.append([plot, txt])

    # Create colorbar
    fig.colorbar(plot, ax = ax, label = r"Material")
    # Create animation
    animation = ani.ArtistAnimation(fig = fig, artists = artists, repeat = True, interval = 50)
    animation.save(filename = fname, writer = "pillow")
    