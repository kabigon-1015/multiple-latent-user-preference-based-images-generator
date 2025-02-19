import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from mcmc._PARAMETERS import *

class Metropolis:
  def __init__(self,d,D):
    self.D = D 
    self.d = d
    self.xs = []

  #次の状態を決定
  def nextsample(self, x, y):
    fx = self.d.get_posterior(x)
    fy = self.d.get_posterior(y)
    a = min(1, (fy / fx))
    if np.random.rand() < a:
      return y
    else:
      return x

  #候補点の生成（一様分布）
  def generate_x_new(self,x):
    return x + 0.5 * np.random.normal(size=self.D)

  #サンプリング
  def sampling(self):
    self.xs.append(np.random.normal(size=self.D))
    for i in range(times):
      self.xs.append(self.nextsample(self.xs[i],self.generate_x_new(self.xs[i])))
    sample = np.array(self.xs[burn_in:])
    return sample
