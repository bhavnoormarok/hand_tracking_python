import numpy as np

def normalize(arr, axis=-1):
    return arr / (np.linalg.norm(arr, axis=-1, keepdims=True))

