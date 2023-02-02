import os
import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib import animation

from lib import fpoly, dfpoly, fractal_functions, generate_sampling, get_colors, generate_cylinder, load_object, \
    prepare_visualization, update_visualization, calculate_abs_gradient
from main import find_root_bisection, find_root_newton, generate_newton_fractal, surface_area, surface_area_gradient, gradient_descent_step


find_root_bisection(lambda x: x ** 2 - 2, np.float64(-1.0), np.float64(2.0))