import numpy as np
from functools import partial
from minimum_squares import LSQ

def linear_func(x, a, b):
    return a + b*x

class LinearModel:
  def __init__(self, X: np.ndarray, Y: np.ndarray):
    self.dependent = Y
    self.predictor = X
    self.__ysize = len(Y)
    self.__xsize = len(X)

    if self.__xsize != self.__ysize:
      print("Size mismatch between predictors array and dependent variables array.")
      print(f"Predictors length: {self.__xsize}\nDependent variables length: {self.__ysize}.")
      raise ValueError("Size mismatch between predictors array and dependent variables array.")
  
class SimpleLinearRegression(LinearModel):
  def __init__(self, X: np.ndarray, Y: np.ndarray):
    super().__init__(X, Y)
    self.__alpha = np.float64(1)
    self.__beta = np.float64(0)

  @property
  def parameters(self) -> np.ndarray:
    return np.array((self.__alpha, self.__beta))
  
  def fit(self):
    self.__alpha, self.__beta = LSQ(
      observed = self.dependent,
      independent = self.predictor
    )
  
  def __call__(self, X: np.ndarray) -> np.ndarray:
    x = np.array(X)
    if self.__alpha == 1 and self.__beta == 0:
      print("The model wasn't fitted to the data yet. Call fit() for this purpose.")
    return linear_func(x, self.__alpha, self.__beta)

if __name__ == "__main__":
  slr = SimpleLinearRegression(
    X=np.array([1, 2, 3]),
    Y=np.array([2, 4, 6])
  )
  slr.fit()
  prevision = slr([1, 2, 3])
  print(prevision)
  print(slr.parameters)
  
  


    
