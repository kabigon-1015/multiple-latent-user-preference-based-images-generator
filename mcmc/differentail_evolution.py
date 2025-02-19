import numpy as np
from mcmc._PARAMETERS import *
from mcmc.metropolis import Metropolis

class DE_MC(Metropolis):
  def __init__(self,d,D):
    super().__init__(d,D)
    #各個体の初期値を設定
    self.Population = [np.random.normal(size=D) for i in range(POPULATION)]

  #候補点の生成（差分進化）
  def generate_x_new(self,x):
    c1,c2 = self.selection(x)
    return self.Population[x] + F * (self.Population[c1] - self.Population[c2]) + np.random.uniform(-sigma_e,sigma_e)

  #個体の選択（ランダム）
  def selection(self,x):
    candidate_list = np.arange(POPULATION)
    candidate_list = np.delete(candidate_list,x)
    c1,c2 = np.random.choice(candidate_list,2)
    return c1,c2

  #サンプリング
  def sampling(self):
    self.xs.append(self.Population[snum])
    for i in range(times):
      for n in range(POPULATION):
        x_new = self.generate_x_new(n)
        for a in range(self.D):
          if CR < np.random.rand():
            x_new[a] = self.Population[n][a]
        self.Population[n] = super().nextsample(self.Population[n],x_new)
        if n == snum:
          self.xs.append(self.Population[snum])
    sample = np.array(self.xs[burn_in:])
    return sample
