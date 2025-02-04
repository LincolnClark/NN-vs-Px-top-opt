import numpy as np
from imageruler import imageruler

def convergence_time(cost_history, tol = 0.05):
    """
    Calculates the number of iterations to reach tol*100 % of the final cost value
    """
    cost_history = np.array(cost_history)

    final_cost = cost_history[-1]
    conv_cost = final_cost * (tol + 1)
    
    # Find the index of the closest value to conv_cost
    ind = np.argmin(np.absolute(cost_history - conv_cost))

    return ind

def length_scale(design, px_size):
    """
    Calculate the minimum feature size for a particular binary design
    Using this paper: https://doi.org/10.1364/JOSAB.506412
    """

    min_width, min_spacing = imageruler.minimum_length_scale(design, periodic = (True, True), ignore_scheme=imageruler.IgnoreScheme.LARGE_FEATURE_EDGES)
    
    return min_width * px_size, min_spacing * px_size

def binarisation_level(design):
    """
    Calculate the difference between fully binarised design and design
    """

    bin_design = design >= 0.5
    grey_level = bin_design - design

    return np.sum(np.abs(grey_level))/(np.prod(design.shape))