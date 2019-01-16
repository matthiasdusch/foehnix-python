import numpy as np
import pandas as pd
import logging
from scipy.stats import logistic
import time

from foehnix.families import *
from foehnix.foehnix_filter import foehnix_filter
from foehnix.iwls_logit import iwls_logit
import foehnix.foehnix_functions as func

# logger
log = logging.getLogger(__name__)


class Control:
    """
    Foehnix Two-Component Mixture-Model Control Object

    Can be passed to the Foehnix class or will be initialized
    """
    def __init__(self, family, switch, left=float('-Inf'), right=float('Inf'),
                 truncated=False, standardize=True, maxit=100, tol=1e-8,
                 force_inflate=False, alpha=None, verbose=True):
        """
        Initialization of the Control object

        Parameters
        ----------
        family : str or :py:class:`foehnix.Family`
            specifying the distribution of the components in the mixture model.

            - 'gaussian'
            - 'logistic'
            - :py:class:`foehnix.Family`
        switch : bool
            whether or not the two components should be switched.

            - ``False`` (default): the component which shows higher values
              within the predictor is assumed to be the foehn cluster.
            - ``True``: lower values are assumed to be the foehn cluster.
        left : float
            left censoring or truncation point. Default `-Inf`
        right : float
            right censoring or truncation point. Default `Inf`
        truncated : bool
            If ``True`` truncation is used instead of censoring. This only
            affects the model if ``left`` and/or ``right`` are specified.
        standardize : bool
            Defines whether or not the model matrix for the concomitant model
            should be standardized for model estimation. Recommended.
        maxit : int or [int, int]
            Maximum number of iterations for the iterative solvers.
            Default is 100. If a vector of length 2 is provided the first value
            is used for the EM algorithm, the second for the IWLS backfitting.
        tol : float or [float, float]
            Tolerance defining when convergence of the iterative solvers is
            reached. Default is 1e-8. If a vector of length 2 is provided the
            first value is used for the EM algorithm, the second for the IWLS
            backfitting.
        force_inflate : bool
            :py:class:`foehnix.Foehnix` will create a strictly regular time
            series by inflating the data to the smallest time intervall in the
            data set. If the inflation rate is larger than 2 the model will
            stop except the user forces inflation by specifying
            ``force_inflate = True``. This can cause a serious runtime
            increase. Default is False.
        alpha : TODO parameter for the penalization of the concomitatnt model
        verbose : bool or str
            Sets the verbose level of the model logging

            - True (default): Information on most tasks will be provided
            - False: Only critical errors and warnings will be provided
            - 'DEBUG': More detailed information will be provided
        """

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
        if np.isfinite([left, right]).any():
            left = np.max([-np.inf, left])
            right = np.min([np.inf, right])
            if left >= right:
                raise ValueError('For censoring and truncation left must be '
                                 'smaller than right.')

        # Check if family object is provided or initialize it
        if isinstance(family, Family):
            log.debug('custom foehnix.Family object provided.')
        else:
            family = initialize_family(familyname=family, left=left,
                                       right=right, truncated=truncated)

        # Maxit and tol are the maximum number of iterations for the
        # optimization. Need to be numeric. If one value is given it will
        # be used for both, the EM algorithm and the IWLS optimization for
        # the concomitants. If two values are given the first one is used
        # for the EM algorithm, the second for the IWLS solver.
        try:
            if len(maxit) == 2:
                self.maxit_em = maxit[0]
                self.maxit_iwls = maxit[1]
            else:
                raise ValueError('maxit must be integer or list of length 2')
        except TypeError:
            self.maxit_em = maxit
            self.maxit_iwls = maxit
        try:
            if len(tol) == 2:
                self.tol_em = tol[0]
                self.tol_iwls = tol[1]
            else:
                raise ValueError('tol must be float or list of length 2')
        except TypeError:
            self.tol_em = tol
            self.tol_iwls = tol

        self.family = family
        self.switch = switch
        self.left = left
        self.right = right
        self.truncated = truncated
        self.standardize = standardize
        self.force_inflate = force_inflate
        self.alpha = alpha


class Foehnix:
    """
    Foehn Classification Based on a Two-Component Mixture Model

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
        data : :py:class:`pandas.DataFrame`
            Index must be a time object, rows must contain neccesary data
        concomitant : str or list of str
            Name(s) of the covariates for the concomitant model. Must be
            present in ``data``. If None (default), a mixture model without
            concomitants will be initialized.
        switch : bool
            - ``False`` (default) if higher values of covariate ``y`` are
              assumed to be the foehn cluster.
            - ``True`` if lower values are the foehn cluster.
        filter_method : dict, function or None
            Evaluates a filter on the data. E.g. a filter on the wind direction
            data to only use data from within a certain wind sector. See
            :py:class:`foehnix.foehnix_filter` for details on the syntax.
        family : str or foehnix.Family class
            - 'gaussian' (default)
            - 'logistic'
        control : :py:class:`foehnix.foehnix.Control`
            If None (default) it will be initialized.
        kwargs : kwargs to pass to the control function
        """

        # Log execution time of foehnix
        start_time = time.time()

        # Initialize Control
        if not isinstance(control, Control):
            control = Control(family, switch, **kwargs)
            log.debug('Foehnix Control object initialized.')

        # Handle multiple concomitants as list of strings:
        if isinstance(concomitant, str):
            concomitant = [concomitant]
        elif concomitant is None:
            concomitant = []

        # Check if predictor and concomitant have sensible values
        if predictor not in data:
            raise ValueError('Predictor variable not found in data')
        for con in concomitant:
            if con not in data:
                raise ValueError('Concomitant "%s" not found in data' % con)

        # Convert index to datetime
        data.index = pd.to_datetime(data.index)
        # check if regular
        if not data.index.is_monotonic_increasing:
            raise RuntimeError('DataFrame index is not monotonic increasing!')

        # calculate minimal difference to make data strictly increasing
        mindiff = data.index.to_series().diff().min()
        inflated = data.asfreq(mindiff).index.size
        lendata = len(data)

        if (inflated/lendata > 2) and (control.force_inflate is False):
            log.critical('You have provided a time series object spanning the '
                         'time period %s to %s \n'
                         'The smallest recorded time interval is %d hours. '
                         'foehnix tries to inflate the time series to create '
                         'a strictly regular time series object which, in '
                         'this case, would yield a data set of dimension '
                         '%d x %d (%d values) which is %.2f times the '
                         'original data set. To avoid running into memory '
                         'issues foehnix stops here! We ask you to check your '
                         'data set.\n'
                         'This condition can be overruled by setting the '
                         'input argument ``force_inflate = True`` if needed. '
                         'For more details please read the foehnix.control '
                         'manual page.' % (data.index[0], data.index[-1],
                                           mindiff.seconds/3600,
                                           inflated, data.shape[1],
                                           inflated*data.shape[1],
                                           inflated/lendata))
            raise RuntimeError('DataFrame gets inflated, see log for details!')

        # Keep the number of observations (rows) added due to inflation.
        N_inflated = inflated - lendata
        # if inflation is ok or forced, create strictly increasing dataframe
        # with minimal spacing
        data = data.asfreq(mindiff)

        # create a subset of the needed data
        columns = concomitant + [predictor]
        subset = data.reindex(columns, axis=1).copy()

        # create index where predictor or concomitant is NaN
        idx_notnan = subset.dropna().index

        # Apply foehnix filter
        filter_obj = foehnix_filter(data, filter_method=filter_method)

        # Take all elements which are not NaN and which are within
        # filter_obj['good']
        idx_take = idx_notnan[idx_notnan.isin(filter_obj['good'])]
        if len(idx_take) == 0:
            raise RuntimeError('No data left after applying required filters.')

        # TODO check with Reto: constant value and truncated check should be
        # TODO   after filtering in my opinion. Else the filtered data might
        # TODO   contain constant values or values outside truncation

        # check if we have columns with constant values.
        # This would lead to a non-identifiable problem
        if (subset.loc[idx_take].nunique() == 1).any():
            raise RuntimeError('Columns with constant values in the data!')

        # and trim data to final size
        y = subset.loc[idx_take, predictor].values.copy()
        y = y.reshape(len(y), 1)

        if len(concomitant) > 0:
            _logitx = np.ones([len(y), len(concomitant)+1])
            for nr, conc in enumerate(concomitant):
                _logitx[:, nr+1] = subset.loc[idx_take, conc].values.copy()

            # store concomitant matrix in a dict
            logitx = {'values': _logitx,
                      'names': ['Intercept'] + concomitant,
                      'scale': np.std(_logitx, axis=0),
                      'center': np.mean(_logitx, axis=0),
                      'is_standardized': False}

            # If std == 0 (e.g. for the Intercept), set center=0 and scale=1
            logitx['center'][logitx['scale'] == 0] = 0
            logitx['scale'][logitx['scale'] == 0] = 1

            # standardize data if control.standardize = True (default)
            if control.standardize is True:
                logitx = func.standardize(logitx)

        # If truncated family is used: y has to lie within the truncation
        # points as density is not defined outside the range ]left, right[.
        if (control.truncated is True) and (
                (y.min() < control.left) or (y.max() > control.right)):
            log.critical('Data %s outside of specified range for truncation '
                         '(left = %.2f, right = %.2f)' % (predictor,
                                                          control.left,
                                                          control.right))
            raise ValueError('Data exceeds truncation range, log for details')

        #
        # - Call the according model
        #
        if len(concomitant) == 0:
            log.info('Calling Foehnix.no_concomitant_fit')
            self.no_concomitant_fit(y, control)
        elif control.alpha is None:
            log.info('Calling Foehnix.unreg_fit')
            self.unreg_fit(y, logitx, control)

        log.info('Estimation finished, create final object.')

        # Final coefficients of the concomitant model have to be destandardized
        if self.optimizer['ccmodel'] is not None:
            if logitx['is_standardized'] is True:
                coef = func.destandardize_coefficients(
                    self.optimizer['ccmodel']['coef'], logitx)
            else:
                coef = self.optimizer['ccmodel']['coef']

        # If there was only one iteration: drop a warning
        if self.optimizer['iter'] == 0:
            log.critical('The EM algorithm stopped after one iteration!\n'
                         'The coefficients returned are the initial '
                         'coefficients. This indicates that the model as '
                         'specified is not suitable for the data. Suggestion: '
                         'check model (e.g, using model.plot() and '
                         'model.summary(detailed = True) and try a different '
                         'model specification (change/add concomitants).')

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
        # Calculate the weighted standard error of the estimated
        # coefficients for the test statistics.
        # 1. calculate weighted sum of squared residuals for both components
        res_c1 = (y - self.coef['mu1']) * (1 - self.optimizer['post'])
        res_c2 = (y - self.coef['mu2']) * self.optimizer['post']
        mu1_se = np.sqrt(np.sum(res_c1**2) /
                         (np.sum((1 - self.optimizer['post'])**2) *
                          (np.sum(1 - self.optimizer['post']) - 1)))
        mu2_se = np.sqrt(np.sum(res_c2**2) /
                         (np.sum(self.optimizer['post']**2) *
                          (np.sum(self.optimizer['post']) - 1)))
        # Standard errors for intercept of mu1(component1) and mu2(component2)
        self.mu_se = {'mu1_se': mu1_se,
                      'mu2_se': mu2_se}

        # The final result, the foehn probability. Creates an object of the
        # same class as the input "data" (currently only pandas.DataFrame!)
        # with two columns. The first contains the final foehn probability
        # (column name prob), the second column contains a flag. The flag is as
        # follows:
        # - NaN  if not modelled (data for the model not available).
        # - 0    if foehn probability has been modelled, data not left out due
        #        to the filter rules.
        # - 1    if the filter removed the observations/sample, not used for
        #        the foehn classification model, but no missing observations.

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

        # Store execution time in seconds
        self.time = time.time() - start_time

    def no_concomitant_fit(self, y, control):
        """Fitting foehnix Mixture Model Without Concomitant Model.

        Parameters
        ----------
        y : :py:class:`numpy.ndarray`
            Covariate for the components of the mixture model
        control : :py:class:`foehnix.foehnix.Control`
            Foehnix control object
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
                log.critical('Likelihood got NaN!')
                raise RuntimeError('Likelihood got NaN!')

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
        ll = np.sum(llpath[-1])

        # TODO might have to adjust the content of llpath and coefpath
        colcoef = 1

        # create results dict
        fdict = {'prob': prob,
                 'post': post,
                 'theta': theta,
                 'loglik': ll,
                 'edf': len(coefpath),
                 'AIC': -2 * ll + 2 * colcoef,
                 'BIC': -2 * ll + np.log(len(y)) * colcoef,
                 'ccmodel': None,
                 'loglikpath': llpath,
                 'coefpath': coefpath,
                 'converged': converged}

        self.optimizer = fdict

    def unreg_fit(self, y, logitx, control):
        """Fitting foehnix Mixture Model Without Concomitant Model.

        Parameters
        ----------
        y : :py:class:`numpy.ndarray`
            Covariate for the components of the mixture model
        logitx : dict
            Covariats for the concomitant model
            Must contain:

            - ``'values'`` : :py:class:`numpy.ndarray` the model matrix
            - ``'center'`` : list, mean of each model matrix row
            - ``'scale'`` : list, standard deviation of matrix rows
            - ``'name'`` : list, names of the rows
            - ``'is_standardized'``: boolean if matrix is standardized
        control : :py:class:`foehnix.foehnix.Control`
            Foehnix control object
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
        ccmodel = iwls_logit(logitx, z, standardize=False,
                             maxit=control.maxit_iwls, tol=control.tol_iwls)

        # Initial probabilities and prior  probabilities
        prob = logistic.cdf(logitx['values'].dot(ccmodel['beta']))
        post = control.family.posterior(y, prob, theta)

        # EM algorithm: estimate probabilities (prob; E-step), update the model
        # given the new probabilities (M-step). Always with respect to the
        # selected family.
        i = 0  # iteration variable
        delta = 1  # likelihood difference between to iteration: break criteria
        converged = True  # Set to False if we do not converge before maxit

        while delta > control.tol_em:
            # M-step: update probabilites and theta
            ccmodel = iwls_logit(logitx, post, beta=ccmodel['beta'],
                                 standardize=False,
                                 maxit=control.maxit_iwls, tol=control.tol_iwls)
            prob = logistic.cdf(logitx['values'].dot(ccmodel['beta']))
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

        ll = np.sum(llpath[-1])
        # TODO im em algorithmus:
        # TODO - beta (Intercept und ff) zu coefpath dazu schreiben
        # TODO - coefpath etwas anderes format wählen
        # TODO

        import pdb
        pdb.set_trace()

        # TODO hardcoded, muss noch die coefpath-Variable anpassen
        # colcoef = ncol(coefpath)
        colcoef = 6

        # create results dict
        fdict = {'prob': prob,
                 'post': post,
                 'theta': theta,
                 'loglik': ll,
                 'edf': len(coefpath),
                 'AIC': -2 * ll + 2 * colcoef,
                 'BIC': -2 * llpath + np.log(len(y)) * colcoef,
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
