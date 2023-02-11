import numpy as np


####################################################################################################
# Exercise 1: Function Roots

def find_root_bisection(f: object, lival: np.floating, rival: np.floating, ival_size: np.floating = -1.0, n_iters_max: int = 256) -> np.floating:
    """
    Find a root of function f(x) in (lival, rival) with bisection method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    lival: initial left boundary of interval containing root
    rival: initial right boundary of interval containing root
    ival_size: minimal size of interval / convergence criterion (optional)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root of the function
    """

    assert (n_iters_max > 0)
    assert (rival > lival)

    # TODO: set meaningful minimal interval size if not given as parameter, e.g. 10 * eps
    info = np.finfo(np.float64)
    if ival_size == -1.0:
        ival_size = 10*info.eps



    # intialize iteration
    fl = f(lival)
    fr = f(rival)

    # make sure the given interval contains a root
    assert (not ((fl > 0.0 and fr > 0.0) or (fl < 0.0 and fr < 0.0)))

    n_iterations = 0
    # TODO: loop until final interval is found, stop if max iterations are reached
    for n_iterations in range(n_iters_max):
        x = (lival+rival)/2
        if (f(x) <= 0 and fl <= 0) or (f(x) >= 0 and fl >= 0):
            lival = x
            fl = f(lival)
        else:
            rival = x
            fr = f(rival)
        # break if interval smaller than machine precision
        if np.abs(lival-rival) < info.eps:
            break


    # TODO: calculate final approximation to root
    root = np.float64(0.0)
    root = np.float64((lival+rival)/2)

    return root


def func_f(x):
    return x**3 - 2*x + 2 # -1.76929235423863

def deri_f(x):
    return 3 * x**2 - 2

def func_g(x):
    return 6*x/(x**2 + 1)

def deri_g(x):
    return 6 * (1 - x**2) / (x**2 + 1)**2

def find_root_newton(f: object, df: object, start: np.inexact, n_iters_max: int = 256) -> (np.inexact, int):
    """
    Find a root of f(x)/f(z) starting from start using Newton's method.

    Arguments:
    f: function object (assumed to be continuous), returns function value if called as f(x)
    df: derivative of function f, also callable
    start: start position, can be either float (for real valued functions) or complex (for complex valued functions)
    n_iters_max: maximum number of iterations (optional)

    Return:
    root: approximate root, should have the same format as the start value start
    n_iterations: number of iterations
    """

    assert(n_iters_max > 0)

    # Initialize root with start value
    root = start

    # TODO: chose meaningful convergence criterion eps, e.g 10 * eps
    precision = np.finfo(root).eps
    criterion = 10*precision

    # Initialize iteration
    fc = f(root)
    dfc = df(root)
    n_iterations = 0

    # TODO: loop until convergence criterion eps is met
    for n_iterations in range(n_iters_max+1):

        # TODO: return root and n_iters_max+1 if abs(derivative) is below f_eps or abs(root) is above 1e5 (to avoid divergence)
        if np.abs(dfc) < precision or np.abs(root) > 1e5:
            return(root, n_iters_max+1)

        # TODO: update root value and function/dfunction values
        old_root = root
        root = root - fc/dfc
        fc = f(root)
        dfc = df(root)

        # TODO: avoid infinite loops and return (root, n_iters_max+1)

        if np.abs(old_root- root) < precision:
            if n_iterations == n_iters_max:
                n_iterations = n_iters_max+1
            break # close enough




    return root, n_iterations

####################################################################################################
# Exercise 2: Newton Fractal


def generate_newton_fractal(f: object, df: object, roots: np.ndarray, sampling: np.ndarray, n_iters_max: int=20) -> np.ndarray:
    """
    Generates a Newton fractal for a given function and sampling data.

    Arguments:
    f: function (handle)
    df: derivative of function (handle)
    roots: array of the roots of the function f
    sampling: sampling of complex plane as 2d array
    n_iters_max: maxium number of iterations the newton method can calculate to find a root

    Return:
    result: 3d array that contains for each sample in sampling the index of the associated root and the number of iterations performed to reach it.
    """

    result = np.zeros((sampling.shape[0], sampling.shape[1], 2), dtype=int)

    # TODO: iterate over sampling grid
    for i in range(sampling.shape[0]):
        for j in range(sampling.shape[1]):

            # TODO: run Newton iteration to find a root and the iterations for the sample (in maximum n_iters_max iterations)
            new_position, iterations_reached = find_root_newton(f,df,sampling[i,j],n_iters_max)

            # TODO: determine the index of the closest root from the roots array. The functions np.argmin and np.tile could be helpful.
            # index = 0
            index = np.argmin(np.abs((np.tile(new_position, roots.shape)-roots)))

            # TODO: write the index and the number of needed iterations to the result
            # result[i, j] = np.array([index, n_iters_max+1])
            result[i, j] = np.array([index, iterations_reached])

    return result


####################################################################################################
# Exercise 3: Minimal Surfaces

def surface_area(v: np.ndarray, f: np.ndarray) -> float:
    """
    Calculate the area of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    area: the total surface area
    """

    # initialize area
    area = 0.0

    # TODO: iterate over all triangles and sum up their area

    # calculates area of a single triangle
    triangle_area = lambda x_i, x_j, x_k : np.linalg.norm(np.cross(x_i-x_j,x_k-x_j))/2
    number_faces  = f.shape[0]
    print(number_faces)
    for i in range(number_faces):
        area += triangle_area(v[f[i,0]],v[f[i,1]],v[f[i,2]])
    return area


def surface_area_gradient(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Calculate the area gradient of the given surface represented as triangles in f.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i

    Return:
    gradient: the surface area gradient of all vertices in v
    """

    # intialize the gradient
    gradient = np.zeros(v.shape)

    # TODO: iterate over all triangles and sum up the vertices gradients
    for i in range(f.shape[0]):
        a = v[f[i,2]]-v[f[i,1]]
        b = v[f[i, 2]] - v[f[i, 0]]
        c = v[f[i, 1]] - v[f[i, 0]]
        #  compute vector normal to triangle face
        n = np.cross(c,b)
        n = n/np.linalg.norm(n)
        gradient[f[i,0]] += np.cross(a,n)
        gradient[f[i, 1]] -= np.cross(b,n)
        gradient[f[i, 2]] += np.cross(c,n)



    return gradient


def gradient_descent_step(v: np.ndarray, f: np.ndarray, c: np.ndarray, epsilon: float=1e-6, ste: float=1.0, fac: float=0.5) -> (bool, float, np.ndarray, np.ndarray):
    """
    Calculate the minimal area surface for the given triangles in v/f and boundary representation in c.

    Arguments:
    v: vertices of the triangles
    f: vertex indices of all triangles. f[i] gives 3 vertex indices for the three corners of the triangle i
    c: list of vertex indices which are fixed and can't be moved
    epsilon: difference tolerance between old area and new area

    Return:
    converged: flag that determines whether the function converged
    area: new surface area after the gradient descent step
    updated_v: vertices with changed positions
    gradient: calculated gradient
    """


    # TODO: calculate gradient and area before changing the surface
    gradient = np.zeros_like(v)
    area = 0.0

    # TODO: calculate indices of vertices whose position can be changed

    # TODO: find suitable step size so that area can be decreased, don't change v yet
    step = ste

    # TODO: now update vertex positions in v

    # TODO: Check if new area differs only epsilon from old area
    # Return (True, area, v, gradient) to show that we converged and otherwise (False, area, v, gradient)

    return False, area, v, gradient


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
