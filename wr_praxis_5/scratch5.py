import os
import numpy as np
import unittest
import time
import matplotlib.pyplot as plt


from lib import idft, dft, ifft, plot_harmonics, read_audio_data, write_audio_data
from main import dft_matrix, is_unitary, create_harmonics, shuffle_bit_reversed_order, fft, \
    generate_tone, low_pass_filter

"""
dft = dft_matrix(2)
print(dft)
x = np.real(dft)
y = np.imag(dft)
print(x)
print(y)
plt.scatter(x,y)
plt.show()
"""

#create_harmonics()

data = np.asarray([1,0,-1,1,2,-1,-2,0])
print(data)
#shuffled = shuffle_bit_reversed_order(data)
#print(shuffled)
print(fft(data))
