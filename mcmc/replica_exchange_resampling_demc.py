import numpy as np
from mcmc._PARAMETERS import *
from mcmc.differentail_evolution import DE_MC
from mcmc.replica_exchange import Replica_exchange

class reRDEMC(Replica_exchange,DE_MC):
  def __init__(self,d,D):
    super().__init__(d,D)
  
  #個体の選択（重要度を考慮）
  def selection(self,x):
    candidate_list = np.arange(POPULATION)
    candidate_list = np.delete(candidate_list,x)
    probability = (self.pdf ** self.Beta[x]) / (self.pdf ** self.Beta * np.sqrt(self.Beta))
    probability = np.delete(probability,1)
    probability/=probability.sum()
    candidate = np.random.choice(candidate_list,2,p=probability.astype(np.float64),replace=False)
    return candidate[0],candidate[1]

  #サンプリング
  def sampling(self):
    self.xs.append(self.Population[snum])
    for i in range(times):
      for k in range(POPULATION):
        x_new = super().generate_x_new(k)
        _new_pdf = self.d.get_posterior(x_new)**self.Beta[k]
        self.Population[k] = super().nextsample(self.Population[k],x_new,self.Beta[k])
        self.pdf[k] = self.d.get_posterior(self.Population[k])
        if k == snum:
          self.xs.append(self.Population[snum])
      if i % frequency_exchange == 0:
        self.exchange()
    sample = np.array(self.xs[burn_in:])
    return sample