import matplotlib.pyplot as plt
import matplotlib.animation as ani

def compare_cost(NN_cost, NN_px_cost, LMpx_costs, px_costs, blur, filename):

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1, 1)

    ax.plot(NN_cost, label = "NN")
    ax.plot(NN_px_cost, label = "NN + px")
    for i in range(len(blur)):
        ax.plot(LMpx_costs[i], label = f"LMpx {blur[i]}")
        ax.plot(px_costs[i], label = f"px {blur[i]}")

    ax.set_title("Cost function evolution")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost function")
    ax.legend()

    plt.savefig(filename)
    plt.close()

    return

def plot_final_design(design, label, filename):

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1, 1)

    ax.pcolormesh(design, vmin = 0, vmax = 1, 
                   shading = "auto", cmap = "gist_gray")
    ax.set_title(label)
    ax.axis("off")

    plt.savefig(filename)
    plt.close()

def compare_performances_dual_pol(NNts, NNtp, NNpxts, NNpxtp, LMpxts, LMpxtp, pxts, pxtp, targets, targetp, angles, blur, filename):

    plt.clf()
    plt.cla()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

    ax1.scatter(angles, targets, color = "black", marker = "X", label = "target")
    ax2.scatter(angles, targetp, color = "black", marker = "X", label = "target")

    ax1.scatter(angles, NNts, label = "NN")
    ax2.scatter(angles, NNtp, label = "NN")

    ax1.scatter(angles, NNpxts, label = "NN + px")
    ax2.scatter(angles, NNpxtp, label = "NN + px")

    for i in range(len(blur)):
        ax1.scatter(angles, LMpxts[i], label = f"LMpx {blur[i]}")
        ax2.scatter(angles, LMpxtp[i], label = f"LMpx {blur[i]}")
        ax1.scatter(angles, pxts[i], label = f"px {blur[i]}")
        ax2.scatter(angles, pxtp[i], label = f"px {blur[i]}")
        

    ax1.set_title("s polarisation")
    ax2.set_title("p polarisation")
    ax1.set_xlabel("Angle of incidence (deg)")
    ax2.set_xlabel("Angle of incidence (deg)")
    ax1.set_ylabel("Amplitude transmission")
    ax1.legend()
    ax2.legend()

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def compare_performances_single_pol(NNt, NNpxt, LMpxt, pxt, target, angles, blur, filename):

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1, 1)

    ax.scatter(angles, target, color = "black", marker = "X", label = "target")

    ax.scatter(angles, NNt, label = "NN")
    ax.scatter(angles, NNpxt, label = "NN + px")
    for i in range(len(blur)):
        ax.scatter(angles, LMpxt[i], label = f"LMpx {blur[i]}")
        ax.scatter(angles, pxt[i], label = f"px {blur[i]}")

    ax.set_xlabel("Angle of incidence (deg)")
    ax.set_ylabel("Amplitude transmission")
    ax.legend()

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()


def compare_performances_dual_pol_spectral(NNts, NNtp, NNpxts, NNpxtp, LMpxts, LMpxtp, pxts, pxtp, targets, targetp, wavelengths, blur, filename):

    plt.clf()
    plt.cla()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

    ax1.scatter(wavelengths, targets, color = "black", marker = "X", label = "target")
    ax2.scatter(wavelengths, targetp, color = "black", marker = "X", label = "target")

    ax1.scatter(wavelengths, NNts, label = "NN")
    ax2.scatter(wavelengths, NNtp, label = "NN")

    ax1.scatter(wavelengths, NNpxts, label = "NN + px")
    ax2.scatter(wavelengths, NNpxtp, label = "NN + px")

    for i in range(len(blur)):
        ax1.scatter(wavelengths, LMpxts[i], label = f"LMpx {blur[i]}")
        ax2.scatter(wavelengths, LMpxtp[i], label = f"LMpx {blur[i]}")
        ax1.scatter(wavelengths, pxts[i], label = f"px {blur[i]}")
        ax2.scatter(wavelengths, pxtp[i], label = f"px {blur[i]}")
        

    ax1.set_title("s polarisation")
    ax2.set_title("p polarisation")
    ax1.set_xlabel("Angle of incidence (deg)")
    ax2.set_xlabel("Angle of incidence (deg)")
    ax1.set_ylabel("Transmittance")
    ax1.legend()
    ax2.legend()

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def compare_performances_single_pol_spectral(NNt, NNpxt, LMpxt, pxt, target, wavelengths, blur, filename):

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1, 1)

    ax.scatter(wavelengths, target, color = "black", marker = "X", label = "target")

    ax.scatter(wavelengths, NNt, label = "NN")
    ax.scatter(wavelengths, NNpxt, label = "NN + px")
    for i in range(len(blur)):
        ax.scatter(wavelengths, LMpxt[i], label = f"LMpx {blur[i]}")
        ax.scatter(wavelengths, pxt[i], label = f"px {blur[i]}")

    ax.set_xlabel("Angle of incidence (deg)")
    ax.set_ylabel("Transmittance")
    ax.legend()

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def animate_history(hist, fname):
    """Create an animation of the density history"""

    # Setup the plot
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(figsize = (6, 5), dpi = 150)
    ax.set_aspect(1)
    ax.set_axis_off()

    # Create list of artists
    artists = []
    for i in range(len(hist)):
        plot = ax.pcolormesh(hist[i], vmin = 0, vmax = 1,
                             shading = "auto", cmap = "gist_gray")
        txt = ax.text(0.5, 1.02, f"Iteration {i}", ha = "center", va = "bottom",
                      transform = ax.transAxes)
        artists.append([plot, txt])

    # Create colorbar
    fig.colorbar(plot, ax = ax, label = r"Material")
    # Create animation
    animation = ani.ArtistAnimation(fig = fig, artists = artists, repeat = True, interval = 50)
    animation.save(filename = fname, writer = "pillow")
    plt.close()