"""
Classes and other stuff
"""

import scipy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import phoeton.functions as pf


class FhoenClassification:
    """ The main FhoenClassification Class

    """

    def __init__(self,
                 data,
                 ddsector=None):
        """ Initialize parmeters which both simple and advanced model use

        Parameters:
        ------------
        data:   pandas.DataFrame
            Must at contain the later predictor 'y'
        ddsector: Array like
            Must contain two values to filter for wind direction
        """

        # Pandas DataFrame
        self.data = data

        # check if ddsector is prescribed
        if ddsector:
            assert 'dd' in data
            # TODO: make ddsector foolproof
            valid = np.where((data['dd'] > ddsector[0]) &
                             (data['dd'] < ddsector[1]))
            self.valid = valid
            self.data = self.data.iloc[valid]
        else:
            self.valid = np.arange(len(self.data))


class SimpleFhoenClassification(FhoenClassification):

    # Fit the logistic model
    def fit(self,
            y,
            omega=None,
            maxiter=10):
        """
        y:       String
            Predictor, must be present in >>data<<
        omega:   String
            Concomitant, must be present in >>data<<
        maxiter:    Integer
            Number of iteration loops
        """

        # predictor
        assert y in self.data
        Y = self.data[y].iloc[self.valid].values

        # Concomitant
        if omega:
            assert omega in self.data
            X = self.data[omega].iloc[self.valid].values

        # This is now just a test implementation
        # The actuall structure must later be more flexible depending on the
        # given Concamitants

        # init latent variable zn
        zn = np.zeros_like(Y)
        zn[Y >= np.median(Y)] = 1

        # init distributional parameters
        theta = np.array([np.quantile(Y, 0.25),
                          np.log(np.std(Y)),
                          np.quantile(Y, 0.75),
                          np.log(np.std(Y))])

        # initial parameters of the Logistical Model
        alpha = sm.Logit(zn, sm.add_constant(X)).fit(disp=0).params

        # Keep stuff for later...
        self.init_parameter = {'theta': theta, 'alpha': alpha}

        # Main Optimization loop
        for i in range(maxiter):
            # E Step
            prob = pf.logitprob(X, alpha)

            dnorm1 = scipy.stats.norm(loc=theta[0],
                                      scale=np.exp(theta[1])).pdf(Y)
            dnorm2 = scipy.stats.norm(loc=theta[2],
                                      scale=np.exp(theta[3])).pdf(Y)
            post = prob * dnorm2 / ((1-prob) * dnorm1 + prob * dnorm2)

            # Emperical update of mu and std
            mu1 = 1 / np.sum(1-post) * np.sum((1-post) * Y)
            mu2 = 1 / np.sum(post) * np.sum(post * Y)

            sd1 = np.sqrt(1 / np.sum(1-post) * np.sum((1-post) * (Y - mu1)**2))
            sd2 = np.sqrt(1 / np.sum(post) * np.sum((post) * (Y - mu2)**2))

            theta = np.array([mu1, np.log(sd1), mu2, np.log(sd2)])

            Q1a = pf.Q1a_fun(theta[0:2], Y, post)
            Q1b = pf.Q1b_fun(theta[2:4], Y, post)
            Q1 = Q1a + Q1b

            Q2 = scipy.optimize.minimize(pf.Q2_fun, alpha,
                                         method='BFGS',
                                         args=(X, post),
                                         jac=pf.Q2_grad,
                                         options={'maxiter': 50})

            alpha = Q2.x

            # log likelihood
            Q = Q1 + Q2.fun

            print('EM step  %2d: loglik = %10.3f' % (i, Q))
            print('    Gauss 1: %12.8f %12.8f' % (theta[0], np.exp(theta[1])))
            print('    Gauss 2: %12.8f %12.8f' % (theta[2], np.exp(theta[3])))
            print('      Alpha: %12.8f %12.8f' % (alpha[0], alpha[1]))
            print('')

        # Store the probability time serie
        # not true jet...
        self.fhoen_probability = Q
