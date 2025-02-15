import numpy as np


file_path = "../data/T1/028.npy"
file = np.load(file_path)
print(file.dtype)
print(file.shape)