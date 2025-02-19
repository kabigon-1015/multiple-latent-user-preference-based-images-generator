import numpy as np
from mcmc._PARAMETERS import *
from mcmc.metropolis import Metropolis

class Replica_exchange(Metropolis):
  def __init__(self,d,D):
    super().__init__(d,D)
    #レプリカごとに初期値を設定
    self.Population = [np.random.normal(size=self.D) for i in range(POPULATION)]
    #分布に与えるパラメータ
    self.Beta = np.empty([POPULATION])
    self.Beta[0] = 1.0
    for b in range(POPULATION-1):
      self.Beta[b+1] = self.Beta[b]*(np.exp((np.log(IT))/np.sum(range(POPULATION))*(b+1)))
    #確率密度（パラメータなし）
    self.pdf = np.empty(POPULATION)
    for n in range(POPULATION):
      self.pdf[n] = self.d.get_posterior(self.Population[n])

  #レプリカ交換 
  def exchange(self):
    for p in range(np.random.randint(2),(POPULATION-1),2):
      x1 = self.Population[p]
      x2 = self.Population[p+1]
      beta1 = self.Beta[p]
      beta2 =  self.Beta[p+1]
      pdf1 = self.pdf[p+1]**beta1
      pdf2 = self.pdf[p]**beta2
      r = (pdf1*pdf2)/(self.pdf[p]**beta1 * self.pdf[p+1]**beta2)
      if r > 1.0 or np.random.rand() < r:
        self.Population[p]= x2
        self.Population[p+1] = x1

  #次の状態を決定
  def nextsample(self, x, y, b):
    fx = self.d.get_posterior(x)**b
    fy = self.d.get_posterior(y)**b
    a = min(1, (fy / fx))
    if np.random.rand() < a:
      return y
    else:
      return x

  #サンプリング
  def sampling(self):
    self.xs.append(self.Population[snum])
    for i in range(times):
      for k in range(POPULATION):
        x_new = super().generate_x_new(self.Population[k])
        _new_pdf = self.d.get_posterior(x_new)**self.Beta[k]
        self.Population[k] = self.nextsample(self.Population[k],x_new,self.Beta[k])
        self.pdf[k] = self.d.get_posterior(self.Population[k])
        if k == snum:
          self.xs.append(self.Population[snum])
      if i % frequency_exchange == 0:
        self.exchange()
    sample = np.array(self.xs[burn_in:])
    return sample
