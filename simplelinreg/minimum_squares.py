import numpy as np
from numpy.linalg import inv

def LSQ(observed: np.ndarray, independent: np.ndarray):
  y = observed.reshape(-1, 1)
  X = np.c_[
    np.ones(len(independent)),
    independent
  ]
  Xt = X.T
  XtX = Xt @ X
  
  b = inv(XtX) @ Xt @ y
  return b.flatten()