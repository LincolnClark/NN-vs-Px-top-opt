import matplotlib.pyplot as plt
import matplotlib.animation as ani

def compare_cost(NN_cost, pixel_cost, filename):

    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1, 1)

    ax.plot(NN_cost, label = "Neural Network")
    ax.plot(pixel_cost, label = "Pixel")
    ax.set_title("Cost function evolution")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost function")
    ax.legend()

    plt.savefig(filename)

    return

def compare_final_designs(NN_des, px_des, filename):

    plt.clf()
    plt.cla()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.pcolormesh(NN_des, vmin = 0, vmax = 1, 
                   shading = "auto", cmap = "gist_gray")
    ax2.pcolormesh(px_des, vmin = 0, vmax = 1, 
                   shading = "auto", cmap = "gist_gray")
    ax1.set_title("Neural Network")
    ax2.set_title("Pixel")

    plt.savefig(filename)

def compare_performances(NNts, NNtp, pxts, pxtp, targets, targetp, angles, filename):

    plt.clf()
    plt.cla()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

    ax1.scatter(angles, targets, color = "black", marker = "X", label = "target")
    ax2.scatter(angles, targetp, color = "black", marker = "X", label = "target")

    ax1.scatter(angles, NNts, label = "NN")
    ax1.scatter(angles, pxts, label = "pixel")
    ax2.scatter(angles, NNtp, label = "NN")
    ax2.scatter(angles, pxtp, label = "pixel")

    ax1.set_title("s polarisation")
    ax2.set_title("p polarisation")
    ax1.set_xlabel("Angle of incidence (deg)")
    ax2.set_xlabel("Angle of incidence (deg)")
    ax1.set_ylabel("Amplitude transmission")
    ax1.legend()
    ax2.legend()

    fig.tight_layout()
    plt.savefig(filename)

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