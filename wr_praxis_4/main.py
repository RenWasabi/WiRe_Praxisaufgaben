import numpy as np

####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)


    # n points => interpolate with degree n-1 polynom (n base polynomials)
    polynomial = np.poly1d(np.zeros(x.size)) # init with 0 because we'll add to it

    print(polynomial)
    base_functions = []

    # TODO: Generate Lagrange base polynomials and interpolation polynomial

    # IMPORTANT: When they're saying base_functions, what they actually want is the base polynomials,
    # so they shouldn't be multiplied with the function value before adding to the list
    # calculate the lagrange base polynomials and functions
    for i in range(x.size): # loop for every base polynomial l_i(x)
        base_polynomial = np.poly1d([1]) # init with 1 bc we'll multiply, MISSING: what if input is single data point
        for j in range(x.size): # loop over all x values to create multiplicands
            if i == j:
                continue
            multiplicant = np.poly1d([ 1/(x[i]-x[j]), -1*x[j]/(x[i]-x[j]) ] )
            base_polynomial = np.polymul(base_polynomial, multiplicant)
        base_function = np.polymul(base_polynomial, y[i])
        print("base function l_", i, " :\n", base_function)
        base_functions.append(base_polynomial) # base function is base polynomial * function value
        polynomial = np.polyadd(polynomial, base_function)


    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x:  x-values of interpolation points
    y:  y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # TODO compute piecewise interpolating cubic polynomials

    # loop over every function segment (between two x)
    for i in range(x.size-1):
        # set up system of linear equations
        function_equations1 = np.array([[1,x[i],x[i]**2,x[i]**3],
                                         [1,x[i+1],x[i+1]**2,x[i+1]**3],
                                         [0,1,2*x[i],3*x[i]**2],
                                         [0,1,2*x[i+1],3*x[i+1]**2]])
        # different column order
        function_equations2 = np.array([[x[i] ** 3, x[i] ** 2, x[i], 1],
                                       [x[i + 1] ** 3, x[i + 1] ** 2, x[i + 1], 1],
                                       [3 * x[i] ** 2, 2 * x[i], 1, 0],
                                       [3 * x[i + 1] ** 2, 2 * x[i + 1], 1, 0]])
        #print(function_equations1)
        constraints = np.array([y[i],y[i+1],yp[i],yp[i+1]])
        coefficients = np.linalg.solve(function_equations1,constraints)
        coefficients = np.flip(coefficients)
        function = np.poly1d(coefficients)
        spline.append(function)

    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO construct linear system with natural boundary conditions
    A = np.zeros((4 * x.size - 4, 4 * x.size - 4))  # matrix holding the function
    constraint = np.zeros(4 * x.size - 4)  # vector holding the values to be resolved against

    # fill the matrix
    # beginning boundary condition
    A[0,0:4] = [0,0,2,6*x[0]]
    # constraint[0] is already 0

    # regular values
    for i in range(x.size - 2):
        A[4*i+1,4*i:4*i+4] = [1,x[i],x[i]**2,x[i]**3]
        A[4 * i + 2, 4 * i:4 * i + 4] = [1, x[i+1], x[i+1] ** 2, x[i+1] ** 3]
        A[4*i+3,4*i:4*(i+1)+4] = [0,1,2*x[i+1],3*x[i+1]**2, 0,-1,-2*x[i+1],-3*x[i+1]**2]
        A[4*i+4,4*i:4*(i+1)+4] = [0,0,2,6*x[i+1], 0,0,-2,-6*x[i+1]]
        constraint[4*i+1] = y[i]
        constraint[4*i+2] = y[i+1]


    # function values for function n-1
    A[4*(x.size-2)+1,4*(x.size-2):] = [1,x[-2],x[-2]**2,x[-2]**3]
    A[4 * (x.size - 2)+2,4*(x.size-2):] = [1,x[-1],x[-1]**2,x[-1]**3]
    constraint[4*(x.size-2)+1] = y[y.size-2]
    constraint[4 * (x.size - 2) + 2] = y[y.size-1]
    # end boundary condition
    A[4*(x.size-2)+3,4*(x.size-2):] = [0,0,2,6*x[x.size-1]]
    # constraint is already 0

    # TODO solve linear system for the coefficients of the spline
    coefficients = np.linalg.solve(A,constraint)

    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(x.size-1):
        func_coeff = coefficients[4*i:4*i+4]
        function = np.poly1d(np.flip(func_coeff))
        spline.append(function)

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions



    # TODO solve linear system for the coefficients of the spline

    spline = []
    # TODO extract local interpolation coefficients from solution


    return spline


if __name__ == '__main__':

    x = np.array( [1.0, 2.0, 3.0, 4.0])
    y = np.array( [3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation( x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
