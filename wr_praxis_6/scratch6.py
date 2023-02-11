import os
import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib import animation

from lib import fpoly, dfpoly, fractal_functions, generate_sampling, get_colors, generate_cylinder, load_object, \
    prepare_visualization, update_visualization, calculate_abs_gradient
from main import find_root_bisection, find_root_newton, generate_newton_fractal, surface_area, surface_area_gradient, gradient_descent_step


#find_root_bisection(lambda x: x ** 2 - 2, np.float64(-1.0), np.float64(2.0))
"""
size = 100 # size of the image
max_iterations = 200

for c, el in enumerate(fractal_functions[:]):
    f, df, roots, borders, name = el
    sampling, size_x, size_y = generate_sampling(borders, size)
    res = generate_newton_fractal(f, df, roots, sampling, n_iters_max=max_iterations)
    colors = get_colors(roots)

    # Generate image
    img = np.zeros((sampling.shape[0], sampling.shape[1], 3))
    for i in range(size_y):
        for j in range(size_x):
            if res[i, j][1] <= max_iterations:
                img[i, j] = colors[res[i, j][0]] / max(1.0, res[i, j][1] / 6.0)

    plt.plot
    plt.imsave('data/fractal_' + name + '.png', img)
    #self.assertTrue(np.allclose(self.data["fr_" + str(c)], img))
    # np.savez("data"+name, fr=img)
"""
nc = 32 # number of elements per layer
nz = 12 # number of layers
v, f, c = generate_cylinder(16, 8)
surface_area(v,f)
surface_area_gradient(v,f)