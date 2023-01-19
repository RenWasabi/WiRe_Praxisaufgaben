import numpy as np
import tomograph
import main



#A = np.random.randn(5, 5)
#b = np.random.randn(5, )

"""
# example that works without pivoting
A = np.array([[1,2,-1],[1,3,1],[1,2,1]])
b = np.array([3,8,7])
'''expected result:
A
[[ 1.  2. -1.]
 [ 0.  1.  2.]
 [ 0.  0.  2.]]
b
[3. 5. 4.]'''
main.gaussian_elimination(A, b)
"""

"""
# example that needs pivoting
A = np.array([[0,2,1],[1,3,2],[3,3,3]])
b = np.array([1,2,3])
'''expected result:
A
[[3. 3. 3.]
 [0. 2. 1.]
 [0. 0. 0.]]
b
[3. 1. 0.]
'''
main.gaussian_elimination(A, b)
#main.gaussian_elimination(A, b, False)
"""

"""
# back_substitution
# system with 1 solution
A = np.array([[4,7,2,3],[0,1,-6,1],[0,0,5,1],[0,0,0,-2]])
b = np.array([9,7,-4,-2])
'''desired result:
x = [2,0,-1,1]'''
print(main.back_substitution(A,b))
"""
"""
# system with no solution
A = np.array([[1,1],[2,2]])
b = np.array([2,0])
A, b = main.gaussian_elimination(A,b)
print("After Gauss: ", A, b)
print(main.back_substitution(A,b))
# desired result: Value error, no solution
"""
"""
# system with infinite solutions
A = np.array([[1,1],[2,2]])
b = np.array([2,4])
A, b = main.gaussian_elimination(A,b)
print("After Gauss: ", A, b)
print(main.back_substitution(A,b))
"""

# FG-Test matrix
A = np.array([[11,44,1],[0.1,0.4,3],[0,1,-1]])
b = np.array([1,1,1])
print(A,b)
A, b = main.gaussian_elimination(A,b)
print("After Gauss:\n", A, b)
print(main.back_substitution(A,b))

#print(np.isclose(-1.38777878e-17,0))