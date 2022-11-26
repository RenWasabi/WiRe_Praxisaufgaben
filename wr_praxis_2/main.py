
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    n, m_a = A.shape
    m_b, = b.shape # without , m_b would be tuple(m_b,)
    if n != m_b:
        raise ValueError('Shapes of matrix {matrix_shape} and vector {vector_shape} not compatible.'\
                         .format(matrix_shape=A.shape, vector_shape=b.shape))
    if n != m_a:
        raise ValueError('Matrix is not square.')

    # TODO: Perform gaussian elimination
    #print(A)
    #print(b)

    # for directly going to the next step of the outer loop in case pivoting is necessary
    # but no non-zero element available
    skip_step = False
    for k in range(n): # gauss elimination has n steps because A has n rows
        # create matrix for elimination in step k
        trans_matrix = np.eye((n))
        # fill matrix with the factors for column k, rows k+1 to n-1
        for i in range(k+1, n, 1):
            # if A[k,k] is zero, pivoting is necessary
            if np.isclose(A[k,k], 0):
                if use_pivoting is False:
                    raise ValueError("The linear system cannot be solved without pivoting.")
                # find the (absolute) largest element in column k, row k+1 to n-1
                # make a slice the column and the desired rows, use argmax
                pivot_row_index = np.argmax(np.absolute(A[k:,k]))
                # special case: diagonal element zero and all elements below it as well
                # => underdetermined system, go to next step k+1 in elimination
                if np.isclose(A[pivot_row_index,k], 0):
                    #print("Hint: The system is underdetermined.")
                    skip_step = True # set flag for skipping rest of iteration of outer loop
                    break
                # exchange row k and row in A and b
                A[[k,pivot_row_index],:] = A[[pivot_row_index,k],:]
                b[[k,pivot_row_index],] = b[[pivot_row_index,k],]

            if skip_step is True:
                continue
            # end pivoting/no pivoting necessary
            trans_factor = -A[i,k] / A[k,k]
            trans_matrix[i,k] = trans_factor
        # transform a and b with the transformation matrix
        A = trans_matrix @ A
        b = trans_matrix @ b

    #print("Nach Transformation:")
    #print(A)
    #print(b)

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    n, m = A.shape
    p, = b.shape
    print(A)
    print(b)
    if n != p:
        raise ValueError('Shapes of matrix {matrix_shape} and vector {vector_shape} not compatible.'\
                         .format(matrix_shape=A.shape, vector_shape=b.shape))


    # TODO: Initialize solution vector with proper size
    x = np.zeros(n)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    # last i needs to be calculated manually because the sum loop would fail otherwise
    if np.isclose(A[n-1, n-1],0):
        if np.isclose(b[n-1],0):
            raise ValueError('The matrix is underdetermined: infinite solutions possible.')
        else:
            raise ValueError('The matrix does not have a solution.')
    x[n-1] = b[n-1]/A[n-1, n-1]
    if n == 1:
        return x
    for i in range(n-2, -1, -1):
        if np.isclose(A[i,i], 0):
            if np.isclose(b[i],0):
                raise ValueError('The matrix is underdetermined: infinite solutions possible.')
            else:
                raise ValueError('The matrix does not have a solution.')
        # calculate the sum of the previous a_i,j x_i products
        sum = 0
        for j in range(i+1, n):
            sum += A[i,j]*x[j]
        x[i] = (b[i]-sum)/A[i,i]
    print(x)
    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape



    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))


    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape


    # TODO Solve the system by forward- and backsubstitution
    x = np.zeros(m)


    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    L = np.zeros((1, 1))
    # TODO: Initialize intensity vector
    g = np.zeros(1)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
    intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)


    return [L, g]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
