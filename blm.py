import emcee
import numpy as np
import patsy
from scipy.stats import norm

class AbstractLinearModel(object):
	pass

class LinearRegression(AbstractLinearModel):
	def __init__(self, X, y):
		self.X_ = X
		self.y_ = y
		self.nobs_ = X.shape[0]
		self.nparams_ = X.shape[1]+1

	def from_formula(formula, data):
		y, X = patsy.dmatrices(formula, data)
		y = y[:,0]
		return LinearRegression(X, y)

	def sample(self, n_sim=1000, burn=0, nwalkers=None):
		if not nwalkers:
			nwalkers = 2*self.nparams_
		p0 = [np.random.uniform(0, 1, self.nparams_) for i in range(nwalkers)]
		sampler = emcee.EnsembleSampler(nwalkers, self.nparams_, self._logprobability)
		sampler.run_mcmc(p0, n_sim)
		samples = sampler.chain[:, burn:, :].reshape(-1, self.nparams_)
		self.samples_coef_ = samples[:,:-1]
		self.samples_sigma_ = samples[:,-1]

	def _logprior(self, v):
		coefs, sigma = v[:-1], v[-1]
		if sigma > 0:
			return np.log(1)
		return -np.inf

	def _loglikelihood(self, v):
		coef, sigma = v[:-1], v[-1]
		y_hat = np.dot(coef, self.X_.T)
		return np.sum(norm.logpdf(self.y_, y_hat, sigma)) 

	def _logprobability(self, v):
		p = self._logprior(v)
		if p == -np.inf:
			return -np.inf
		return self._loglikelihood(v) + p

	def summary(self):
		pass

	def predict_sample(self, data):
		X = patsy.dmatrix(self.X_.design_info, data)
		return np.dot(self.samples_coef_, X.T)
