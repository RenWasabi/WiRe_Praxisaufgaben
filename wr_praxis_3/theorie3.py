import numpy as np

A = np.array([[3,5,8,1],[2,1,7,1],[2,4,8,8],[6,6,4,2]])

eigval, eigv = np.linalg.eig(A)

print(A)
print(eigval)
print(eigv)