from scipy.stats import multivariate_normal

class distribution:
  def __init__(self,mu,D,p):
    self.mu = mu
    self.D = D
    self.p = p
    self.sigma = []
    self.K = p.shape[0]

  def set_sigma(self, sigma):
    self.sigma = sigma

  def get_posterior(self, x):
    y = 0
    for i in range(self.K):
      y = y + self.p[i] * multivariate_normal.pdf(x, mean=self.mu[i], cov=self.sigma)
    return y

  def get_mu(self):
    return self.mu