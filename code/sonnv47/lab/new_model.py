# import numpy as np
# from scipy.special import digamma
# import time
#
# def dirichlet_expectation(alpha):
#     """
#     For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
#     """
#     if len(alpha.shape) == 1:
#         return digamma(alpha) - digamma(sum(alpha))
#     return (digamma(alpha) - digamma(np.sum(alpha, axis=1))[:, np.newaxis])
#
# def softmax_convert(matrix):
#     matrix_max = np.max(matrix, axis = 1)[:, np.newaxis]
#     temp = np.exp(matrix - matrix_max)
#     matrix_softmax = temp / np.sum(temp, axis=1)[:, np.newaxis]
#     return matrix_softmax
#
# class Stream:
#
#     def __init__(self, num_topics, n_terms, n_infer, shape_alpha, shape_beta, sigma, learning_rate, espi):
#         self.num_topics = num_topics
#         self.n_terms = n_terms
#         self.n_infer = n_infer
#         self.learning_rate = learning_rate
#         self.espi = espi
#         self.shape_alpha = shape_alpha
#         self.shape_beta = shape_beta
#         self.sigma = sigma
#         self.droprate = np.random.beta(self.shape_alpha, self.shape_beta, size=(self.num_topics, self.n_terms))
#         self.pi = np.random.normal(np.ones_like(self.droprate), np.sqrt(self.droprate / (1 - self.droprate)))
#         self.beta_temp = np.random.rand(self.num_topics, self.n_terms)
#         self.beta /= np.sum(self.beta_temp, axis=1)[:, np.newaxis]
#
#
#     def do_e_step(self, batchsize, wordinds, wordcnts):
#
#         gamma = np.random.gamma(100., 1./100, (batchsize, self.num_topics))
#         Elogtheta = dirichlet_expectation(gamma)
#         expElogtheta = np.exp(Elogtheta)
#
#         droprate_t = np.random.beta(2*self.droprate, 2*(1 - self.droprate))
#         pi_t = np.random.normal(np.ones_like(droprate_t), np.sqrt(droprate_t / (1 - droprate_t)))
#         beta_t = np.random.normal(self.beta, self.sigma*np.ones_like(self.beta))
#         # droprate_t = self.droprate
#         # pi_t = self.pi
#         # beta_t = self.beta
#         beta_temp = beta_t * pi_t
#         beta_drop = softmax_convert(beta_temp)
#
#         beta = beta_drop # NOTE
#
#         for d in range(batchsize):
#             inds = wordinds[d]
#             cnts = wordcnts[d]
#             gamma_d = gamma[d, :]
#             Elogtheta_d = Elogtheta[d, :]
#             expElogtheta_d = expElogtheta[d, :]
#             # gamma_d = np.ones(self.K) * self.alpha + float(np.sum(cntx)) / self.K
#             # expElogtheta_d = np.expElogtheta(dirichlet_expectation(gamma_d))
#             beta_d = beta[:, inds]
#             for i in range(self.n_infer):
#                 phi_d = expElogtheta_d * beta_d.transpose()
#                 phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis]
#                 gamma_d = self.alpha + np.dot(cnts, phi_d)
#                 expElogtheta_d = np.exp(dirichlet_expectation(gamma_d))
#
#             gamma[d] = gamma_d
#             expElogtheta[d] = expElogtheta_d
#
#         print ('Updating parameter...')
#         start = time.time()
#         droprate_t_opt, pi_t_opt, beta_t_opt = self.update_gradient(droprate_t, pi_t, beta_t, wordinds, wordcnts, gamma, expElogtheta)
#         end = time.time()
#         print ('Total time update: ', end - start)
#         return (gamma, droprate_t_opt, pi_t_opt, beta_t_opt)
#
#     def update_gradient(self, droprate_t, pi_t, beta_t, wordinds, wordcnts, gamma, expElogtheta):
#         for i in range(50):
#             print("Iter:", i)
#             grad_pi_t, grad_beta_t = self.gradient_ascent(droprate_t, pi_t, beta_t, wordinds, wordcnts, gamma, expElogtheta)
#             pi_t += self.learning_rate*grad_pi_t / batchsize
#             beta_t += self.learning_rate*grad_beta_t / batchsize
#             droprate_t = (pi_t - 1)**2 / (2*2*self.pi - (pi_t - 1)**2 - 3) # choose shape_alpha = 2*pi_(t-1), shape_beta = 2*(1-pi_(t-1))
#         return (droprate_t, pi_t, beta_t)
#
#     def gradient_ascent(self, droprate_t, pi_t, beta_t, wordinds, wordcnts, gamma, expElogtheta):
#         batchsize = len(wordinds)
#
#         sum_phi = np.zeros((self.num_topics, 1)) # k-element is sum(phi_dnk, d=1...D, n=1...Nd)
#         # sum_exp = np.zeros((self.num_topics, 1)) # k-element is sum(exp(beta_ki*pi_ki), i=1...V)
#         sum_iphi = np.zeros((self.num_topics, self.n_terms)) # kj-element is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)
#
#         sum_exp = np.sum(np.exp(beta_t*pi_t), axis=1)
#
#         for d in range(batchsize):
#             inds = wordinds[d]
#             cnts = wordcnts[d]
#             expElogtheta_d = expElogtheta[d]
#             beta_d = beta_t[:, inds]
#
#             phi_d = expElogtheta_d * beta_d.transpose() # infer phi_d one more time
#             phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis] # shape num_term_dxK
#
#             sum_phi += np.sum(phi_d, axis=0)[:, np.newaxis] # sum n=1...Nd for each k --------------------------
#
#             #for n, term in enumerate(inds):
#                 #sum_iphi[:, term] += cnts[n] * phi_d[n] # NOTE -----------------------
# 			sum_iphi[:, inds] += phi_d.T * cnts
#
#         grad_pi_t = (1 - pi_t)*(1 - droprate_t)/droprate_t + beta_t*(sum_iphi - np.exp(beta_t*pi_t)*sum_phi/sum_exp)
#         grad_beta_t = -2*beta_t*(beta_t - self.beta) + pi_t*(sum_phi - np.exp(beta_t*pi_t)*sum_phi/sum_exp)
#
#         return (grad_pi_t, grad_beta_t)
#
#
#     def update_stream(self, batchsize, wordinds, wordcnts):
#         gamma, droprate_t, pi_t, beta_t = self.do_e_step(batchsize, wordinds, wordcnts)
#         self.pi = pi_t
#         self.droprate = droprate_t
#         self.beta = beta_t
#         return gamma
#
#
#
