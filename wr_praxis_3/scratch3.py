import numpy as np
import lib
import matplotlib as mpl
#import unittest
from main import power_iteration, load_images, setup_data_matrix, calculate_pca, project_faces, accumulated_energy, identify_faces

#class Tests(unittest.TestCase):
#    def my_test_power_iteration(self):

#M = np.array([[1,4],[4,5]])
#print("M:\n", M)
#power_iteration(M)

# for some reason working directory is wr_praxis_1, not three => specify in test file
img_list, dim_x, dim_y = load_images("../wr_praxis_3/data/train")
print("x, y: ",dim_x, dim_y)

data_matrix = setup_data_matrix(img_list)
pcs, svals, mean_data = calculate_pca(data_matrix)
k = accumulated_energy(svals)
pcs = pcs[0:k, :]
coeff_train = project_faces(pcs, img_list, mean_data)
identify_faces(coeff_train, pcs, mean_data, "../wr_praxis_3/data/test" )




