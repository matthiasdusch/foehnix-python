"""
Classes and other stuff
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import logistic

from foehnix.families import Family, GaussianFamily
from foehnix.foehnix_filter import foehnix_filter
from foehnix.iwls_logit import iwls_logit

# logger
log = logging.getLogger(__name__)


class Control:
    """Control Object for Foehnix

    Can be passed to the Foehnix class or will be initialized
    """
    def __init__(self, family, switch, left=float('-Inf'), right=float('Inf'),
                 truncated=False, standardize=True, maxit=100, tol=1e-8,
                 alpha=None, verbose=True):

        # set logging
        if verbose is True:
            logging_level = 'INFO'
        elif verbose is False:
            logging_level = 'CRITICAL'
        elif verbose == 'DEBUG':
            logging_level = 'DEBUG'
        logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=getattr(logging, logging_level))

        # Check limits for censoring/truncation
        # TODO

        # Check if family object is provided or initialize it
        # TODO: Gibts ne schoenere Logik hier?
        if isinstance(family, Family):
            log.info('foehnix.Family object provided.')
        elif np.isinf([left, right]).all():
            if family == 'gaussian':
                log.debug('Initializing Gaussian foehnix mixture model family.')
                family = GaussianFamily
            elif family == 'logistic':
                log.debug('Initializing Logistic foehnix mixture model family.')
                family = LogisticFamily
        elif np.isfinite([left, right]).any():
            if truncated:
                if family == 'gaussian':
                    log.debug('Initializing truncated Gaussian fmmf.')
                    family = TruncatedGaussianFamily
                elif family == 'logistic':
                    log.debug('Initializing truncated Logistic fmmf.')
                    family = TruncatedLogisticFamily
            elif not truncated:
                if family == 'gaussian':
                    log.debug('Initializing censored Gaussian fmmf.')
                    family = CensoredGaussianFamily
                elif family == 'logistic':
                    log.debug('Initializing censored Logistic fmmf.')
                    family = CensoredLogisticFamily

        # Maxit and tol are the maximum number of iterations for the
        # optimization. Need to be numeric. If one value is given it will
        # be used for both, the EM algorithm and the IWLS optimization for
        # the concomitants. If two values are given the first one is used
        # for the EM algorithm, the second for the IWLS solver.
        # TODO hier stimmt eventuell der Ablauf try if else except nicht!!
        try:
            if len(maxit) == 2:
                self.maxit_em = maxit[0]
                self.maxit_iwls = maxit[1]
            else:
                raise RuntimeError('maxit must be integer or list of length 2')
        except TypeError:
            self.maxit_em = maxit
            self.maxit_iwls = maxit
        try:
            if len(tol) == 2:
                self.tol_em = tol[0]
                self.tol_iwls = tol[1]
            else:
                raise RuntimeError('tol must be integer or list of length 2')
        except TypeError:
            self.tol_em = tol
            self.tol_iwls = tol

        self.family = family
        self.switch = switch
        self.left = left
        self.right = right
        self.truncated = truncated
        self.standardize = standardize
        self.maxit = maxit
        self.tol = tol
        self.alpha = alpha


class Foehnix:
    """Foehn Classification Based on a Two-Component Mixture Model

    This is the main method of the foehnix package to estimate two-component
    mixture models for automated foehn classification.
    """
    def __init__(self, predictor, data, concomitant=None, switch=False,
                 filter_method=None, family='gaussian', control=None,
                 **kwargs):
        """ Initialize parmeters which all methods need.

        Parameters
        ----------
        predictor : str
            Name of the main predictor (covariate) variable which is used to
            identify the foehn/no-foehn cluster. Must be present in ``data``.
        data : pandas.DataFrame
            Index must be a time object, rows must contain neccesary data
        concomitant : str
            Name of the covariate for the concomitant model. Must be present in
            ``data``. If None (default), a mixture model without concomitants
            will be initialized.
        switch : bool
            | ``False`` (default) if higher values of covariate ``y`` are
            assumed to be the foehn cluster.
            | ``True`` if lower values are the foehn cluster.
        filter_method : dict, function or None
            Evaluates a filter on the data. E.g. a filter on the wind direction
            data to only use data from within a certain wind sector. See
            :py:class:`foehnix.foehnix_filter` for details on the syntax.
        family : str or foehnix.Family class
            | 'gaussian' (default)
            | 'logistic'
        control : :py:class:`foehnix.Control`
            If None (default) it will be initialized.
        kwargs : kwargs to pass to the control function
        """

        # TODO multiple concomitants? Dann list(str) etc...
        if isinstance(concomitant, str):
            concomitant = [concomitant]
        elif concomitant is None:
            concomitant = []

        # Log execution time of foehnix
        # TODO

        # Check if predictor and concomitant have sensible values
        if predictor not in data:
            raise RuntimeError('Predictor variable not found in data')

        # TODO think
        #if concomitant is not None and concomitant not in data:
        #    raise RuntimeError('Concomitant variable not found in data')

        # Initialize Control
        if not isinstance(control, Control):
            control = Control(family, switch, **kwargs)

        # Create a strictly regular time series with pandas Datetime
        data.index = pd.to_datetime(data.index)
        # check if regular
        if not data.index.is_monotonic_increasing:
            raise RuntimeError('DataFrame index is not monotonic increasing!')
        # force to a strict increasing dataframe with minimal spacing
        mindiff = data.index.to_series().diff().min()
        data = data.asfreq(mindiff)

        # TODO a lot of checks

        # create a subset of the needed data
        columns = concomitant + [predictor]
        subset = data.reindex(columns, axis=1).copy()
        # drop columns full of NaN (this is the case for concomitant==None)
        subset.dropna(axis=1, how='all', inplace=True)

        # create index where predictor or concomitant is NaN
        idx_notnan = subset.dropna().index

        # Apply foehnix filter
        filter_obj = foehnix_filter(data, filter_method=filter_method)

        # Take all elements which are not NaN and which are within
        # filter_obj['good']
        idx_take = idx_notnan[idx_notnan.isin(filter_obj['good'])]

        if len(idx_take) == 0:
            raise RuntimeError('No data left after applying required filters.')

        # and trim data to final size
        y = subset.loc[idx_take, predictor].values.copy()
        y = y.reshape(len(y), 1)

        if len(concomitant) > 0:
            logitX = np.ones([len(y), len(concomitant)+1])
            for nr, conc in enumerate(concomitant):
                logitX[:, nr+1] = subset.loc[idx_take, conc].values.copy()

        # Standardize data
        # TODO

        # call model accordingly
        if len(concomitant) == 0:
            log.info('Calling Foehnix.no_concomitant_fit')
            self.no_concomitant_fit(y, control)
        elif control.alpha is None:
            log.info('Calling Foehnix.unreg_fit')
            self.unreg_fit(y, logitX, control)

        log.info('Estimation finished, create final object.')

        # Final coefficients of the concomitant model have to be destandardized
        # if standardize == TRUE.
        if self.optimizer['ccmodel'] is not None:
            # TODO need to check if ceofficients are standardized
            coef = '2do'
        else:
            coef = None

        # store relevant data within the Foehnix class
        self.data = data
        self.foehnix_filter = foehnix_filter
        self.filter_obj = filter_obj
        self.predictor = predictor
        self.concomitant = concomitant
        self.control = control
        self.switch = switch
        # TODO struktur von coef
        self.coef = self.optimizer['theta']
        self.coef_concomitants = coef

        # TODO some more stuff like weights and estimated coefficients

        # The final result, the foehn probability. Creates an object of the same
        # class as the input "data" (currently only pandas.DataFrame!) with two
        # columns. The first contains the final foehn probability (column name
        # prob), the second column contains a flag. The flag is as follows:
        # - NaN  if not modelled (data for the model not available).
        # - 0    if foehn probability has been modelled, data not left out due
        #        to the filter rules.
        # - 1    if the filter removed the observations/sample, not used for the
        # foehn classification model, but no missing observations.

        # The following procedure is used:
        # - By default, use NaN for both columns.
        # - If probabilities modelled: set first column to the modelled
        #   a-posteriory probability, set the second column to TRUE.
        # - If observations removed due to the filter options: set first column
        #   to 0 (probability for foehn is 0), set the second column to FALSE.

        # Foehn probability (a-posteriori probability)
        tmp = pd.DataFrame([], columns=['prob', 'flag'], index=data.index)
        # Store a-posteriory probability and flag = TRUE
        tmp.loc[idx_take, 'prob'] = self.optimizer['post'].reshape(len(y))
        tmp.loc[idx_take, 'flag'] = 1
        # Store prob = 0 and flag=0 where removed due to filter rule
        tmp.loc[filter_obj['bad']] = 0

        # store in self
        self.prob = tmp.copy()

    def no_concomitant_fit(self, y, control):
        """Fitting foehnix Mixture Model Without Concomitant Model.

        Parameters
        ----------
        """

        # Lists to trace log-likelihood path and the development of
        # the coefficients during EM optimization.
        llpath = []
        coefpath = []

        # Given the initial probabilities: calculate parameters for the two
        # components (mu1, logsd1, mu2, logsd2) given the selected family and
        # calculate the a-posteriori probabilities.
        z = np.zeros_like(y)
        if control.switch:
            z[y <= np.mean(y)] = 1
        else:
            z[y >= np.mean(y)] = 1
        theta = control.family.theta(y, z, init=True)  # M-step

        # Initial probability (fifty fifty) and inital prior probabilites for
        # the component membership.
        prob = np.mean(z)
        post = control.family.posterior(y, prob, theta)

        # EM algorithm: estimate probabilities (prob; E-step), update the model
        # given the new probabilities (M-step). Always with respect to the
        # selected family.
        i = 0  # iteration variable
        delta = 1  # likelihood difference between to iteration: break criteria
        converged = True  # Set to False if we do not converge before maxit

        while delta > control.tol_em:
            # M-step: update probabilites and theta
            prob = np.mean(post)
            # TODO was mach das theta=theta hier?
            # theta = control.family.theta(y, post, theta=theta)
            theta = control.family.theta(y, post)

            # E-step: calculate a-posteriori probability
            post = control.family.posterior(y, np.mean(prob), theta)

            # Store log-likelihood and coefficients of the current iteration.
            llpath.append(control.family.loglik(y, post, prob, theta))
            coefpath.append(theta)
            log.info('EM iteration %d/%d, ll = %10.2f' % (i, control.maxit_em,
                                                          np.sum(llpath[i])))
            if np.isnan(np.sum(llpath[i])):
                log.critical('Likelihood got NA!')
                raise RuntimeError('Likelihood got NA!')

            # update liklihood difference
            if i > 0:
                delta = np.sum(llpath[i]) - np.sum(llpath[i-1])

            # increase iteration variable
            i += 1

            # check if we converged
            if i == control.maxit_em:
                converged = False
                break

        # If converged, remove last likelihood and coefficient entries
        if converged:
            llpath = llpath[:-1]
            coefpath = coefpath[:-1]

        # TODO might have to adjust the content of llpath and coefpath
        colcoef = 1

        # create results dict
        fdict = {'prob': prob,
                 'post': post,
                 'theta': theta,
                 'loglik': np.sum(llpath[-1]),
                 'edf': len(coefpath),
                 'AIC': -2 * np.sum(llpath[-1]) + 2 * colcoef,
                 'BIC': -2 * np.sum(llpath[-1]) + np.log(len(y)) * colcoef,
                 'ccmodel': None,
                 'loglikpath': llpath,
                 'coefpath': coefpath,
                 'converged': converged}

        self.optimizer = fdict

    def unreg_fit(self, y, logitX, control):
        """Fitting foehnix Mixture Model Without Concomitant Model.

        Parameters
        ----------
        """

        # Lists to trace log-likelihood path and the development of
        # the coefficients during EM optimization.
        llpath = []
        coefpath = []

        # Given the initial probabilities: calculate parameters for the two
        # components (mu1, logsd1, mu2, logsd2) given the selected family and
        # calculate the a-posteriori probabilities.
        z = np.zeros_like(y)
        if control.switch:
            z[y <= np.mean(y)] = 1
        else:
            z[y >= np.mean(y)] = 1
        theta = control.family.theta(y, z, init=True)  # M-step

        # Initial probability: fifty/fifty!
        # Force standardize = FALSE. If required logitX has alreday been
        # standardized in the parent function (foehnix)
        ccmodel = iwls_logit(logitX, z, standardize=False,
                             maxit=control.maxit_iwls, tol=control.tol_iwls)

        # Initial probabilities and prior  probabilities
        prob = logistic.cdf(logitX.dot(ccmodel['beta']))
        post = control.family.posterior(y, prob, theta)

        """
        # initial parameters of the Logistical Model
        # alpha = sm.Logit(zn, sm.add_constant(X)).fit(disp=0).params

        # Keep stuff for later...
        # self.init_parameter = {'theta': theta, 'alpha': alpha}
        """

        # EM algorithm: estimate probabilities (prob; E-step), update the model
        # given the new probabilities (M-step). Always with respect to the
        # selected family.
        i = 0  # iteration variable
        delta = 1  # likelihood difference between to iteration: break criteria
        converged = True  # Set to False if we do not converge before maxit

        while delta > control.tol_em:
            # M-step: update probabilites and theta
            ccmodel = iwls_logit(logitX, post, beta=ccmodel['beta'],
                                 standardize=False,
                                 maxit=control.maxit_iwls, tol=control.tol_iwls)
            prob = logistic.cdf(logitX.dot(ccmodel['beta']))
            # TODO was mach das theta=theta hier?
            theta = control.family.theta(y, post)

            # E-step: update expected a-posteriori
            post = control.family.posterior(y, prob, theta)

            # Store log-likelihood and coefficients of the current iteration.
            llpath.append(control.family.loglik(y, post, prob, theta))
            # TODO append theta and ccmodel.beta?
            coefpath.append(theta)

            log.info('EM iteration %d/%d, ll = %10.2f' % (i, control.maxit_em,
                                                          np.sum(llpath[i])))
            # update liklihood difference
            if i > 0:
                delta = np.sum(llpath[i]) - np.sum(llpath[i-1])

            # increase iteration variable
            i += 1

            # check if we converged
            if i == control.maxit_em:
                converged = False
                break

        # If converged, remove last likelihood and coefficient entries
        if converged:
            llpath = llpath[:-1]
            coefpath = coefpath[:-1],

        # TODO hardcoded, muss noch die coefpath-Variable anpassen
        # colcoef = ncol(coefpath)
        colcoef = 6

        # create results dict
        fdict = {'prob': prob,
                 'post': post,
                 'theta': theta,
                 'loglik': np.sum(llpath[-1]),
                 'edf': len(coefpath),
                 'AIC': -2 * np.sum(llpath[-1]) + 2 * colcoef,
                 'BIC': -2 * np.sum(llpath[-1]) + np.log(len(y)) * colcoef,
                 'ccmodel': ccmodel,
                 'loglikpath': llpath,
                 'coefpath': coefpath,
                 'converged': converged,
                 'iter': i-1}

        self.optimizer = fdict

    def summary(self):
        """ print summary
        """

        sum_na = self.prob.isna().sum()['flag']
        sum_0 = (self.prob['flag'] == 0).sum()
        sum_1 = (self.prob['flag'] == 1).sum()

        mean_n = self.prob.notna().sum()['flag']
        mean_occ = 100 * (self.prob['prob'] >= .5).sum() / mean_n

        mean_prob = 100 * self.prob['prob'][self.prob['flag'].notna()].mean()

        # Additional information about the data/model
        nr = len(self.prob)
        print("\nNumber of observations (total) %8d" % nr)
        print("Removed due to missing values  %8d (%3.1f percent)" %
              (sum_na, sum_na / nr * 100))
        print("Outside defined wind sector    %8d (%3.1f percent)" %
              (sum_0, sum_0 / nr * 100))
        print("Used for classification        %8d (%3.1f percent)" %
              (sum_1, sum_1 / nr * 100))
        print("\nClimatological foehn occurance %.2f percent (on n = %d)" %
              (mean_occ, mean_n))
        print("Mean foehn probability %.2f percent (on n = %d)" %
              (mean_prob, mean_n))
        print("\nLog-likelihood: %.1f, %d effective degrees of freedom" %
              (self.optimizer['loglik'], -999))
        print("Corresponding AIC = %.1f, BIC = %.1f\n" %
              (self.optimizer['AIC'], self.optimizer['BIC']))
