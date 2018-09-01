import numpy as np
from scipy.special import digamma


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return digamma(alpha) - digamma(sum(alpha))
    return (digamma(alpha) - digamma(np.sum(alpha, axis=1))[:, np.newaxis])


class Stream:

    def __init__(self, alpha, beta, k, n_terms, n_infer):
        self.alpha = alpha
        self.beta = beta
        self.K = k
        self.n_infer = n_infer
        self.n_terms = n_terms

    def do_e_step(self, batchsize, wordinds, wordcnts):
    	"""
        Does infernce for documents in 'w_obs' part.
        Arguments:
            batch_size: number of documents to be infered.
            wordinds: A list whose each element is an array (terms), corresponding to a document.
                 Each element of the array is index of a unique term, which appears in the document,
                 in the vocabulary.
            wordcnts: A list whose each element is an array (frequency), corresponding to a document.
                 Each element of the array says how many time the corresponding term in wordids appears
                 in the document.
        Returns: gamma the variational parameter of topic mixture (theta).
        """
        # Initialize the variational distribution q(theta|gamma) for the mini-batch
        gamma = np.random.gamma(100., 1. / 100, (batchsize, self.K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        # Now, for each document d update that document's gamma and phi
        for d in range(batchsize):
            inds = wordinds[d]
            cnts = wordcnts[d]
            gamma_d = gamma[d, :]
            Elogtheta_d = Elogtheta[d, :]
            expElogtheta_d = expElogtheta[d, :]
            # gamma_d = np.ones(self.K) * self.alpha + float(np.sum(cntx)) / self.K
            # expElogtheta_d = np.expElogtheta(dirichlet_expectation(gamma_d))
            beta_d = self.beta[:, inds]
            for i in range(self.n_infer):
                phi_d = expElogtheta_d * beta_d.transpose()
                phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis]
                gamma_d = self.alpha + np.dot(cnts, phi)
                expElogtheta_d = np.exp(dirichlet_expectation(gamma_d))

            gamma[d] = gamma_d
            # gamma[d] /= sum(gamma[d]) # why normalization

        return gamma

    def compute_doc(self, gamma_d, wordinds, wordcnts):
        """
        Compute log predictive probability for each document in 'w_ho' part.
        """
        ld2 = 0
        frequency = np.sum(wordcnts)
        for i in range(len(wordinds)):
            p = np.dot(gamma_d, self.beta[:, wordinds[i]])
            ld2 += np.log(p)
        if (frequency == 0):
            return ld2
        else:
            return ld2/frequency

    def compute_perplexity(self, wordinds1, wordcnts1, wordinds2, wordcnts2):
        """
        Compute log predictive probability for all documents in 'w_ho' part.
        """
        batchsize = len(wordinds1)
        # e step
        gamma = self.do_e_step(batchsize, wordinds1, wordcnts1)
        # compute perplexity
        LD2 = 0
        for i in range(batchsize):
            LD2 += self.compute_doc(gamma[i], wordinds2[i], wordcnts2[i])
        return LD2/batchsize

