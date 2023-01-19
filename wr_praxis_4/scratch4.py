import numpy as np
from main import lagrange_interpolation, hermite_cubic_interpolation, natural_cubic_interpolation, periodic_cubic_interpolation
from lib import plot_function, plot_function_interpolations, plot_spline, animate, linear_animation, cubic_animation, \
    runge_function, pad_coefficients

# 1.1 Lagrange
#x_s, y_s = runge_function(n=10)
#lagrange_interpolation(np.array([1,2]), np.array([1,2]))
#lagrange_interpolation(x_s, y_s)

# tut beispiel
#x = np.array([1,2,4])
#y = np.array([3,4,1])

# tut besp
#x = np.array([1,2,3])
#y = np.array([1,2,2])

# x^2
#x = np.array([1,2,3])
#y = np.array([1,4,9])
#print(x.size, y.size)
#print(x, y)
#print("output:\n",lagrange_interpolation(x,y))



# 1.2 kubisch Hermite
#x = np.array([1,2,3,4])
#degree = 4
#vander = np.vander(x,degree)
#print(vander)
#print(np.invert(vander))

# tut 9
#x = np.array([1,2])
#y = np.array([1,2])
#yp = np.array([0,1])
#print(hermite_cubic_interpolation(x,y,yp))


# spline natural boundary conditions
x, y = runge_function(6)
#natural_cubic_interpolation(x,y)
periodic_cubic_interpolation(x,y)



