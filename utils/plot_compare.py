import matplotlib.pyplot as plt

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