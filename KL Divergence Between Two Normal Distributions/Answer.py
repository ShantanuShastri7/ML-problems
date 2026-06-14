import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):

	log = np.log(sigma_q/sigma_p)
	middleTerm = (np.power(sigma_p, 2) + np.power((mu_p - 		mu_q), 2))/(2*np.power(sigma_q, 2)) - 0.5
	return log + middleTerm
