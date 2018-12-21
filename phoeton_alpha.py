import numpy as np
import statsmodels.api as sm
import scipy


def Q1_fun(par, y, post):
    dnorm1 = scipy.stats.norm(loc=par[0], scale=np.exp(par[1])).logpdf(y)
    dnorm2 = scipy.stats.norm(loc=par[2], scale=np.exp(par[3])).logpdf(y)
    return -np.sum((1-post) * dnorm1 + post * dnorm2)


def Q2_fun(par, omega, post):
    prob = logitprob(omega, par)
    return -np.sum((1-post) * np.log(1-prob) + post * np.log(prob))


def logitprob(x, alpha):
    x = alpha[0] + x * alpha[1]
    return np.exp(x) / (1 + np.exp(x))

# synthetic data points
N = 1000
# wind speed
mu1 = 10
mu2 = 25
sd1 = 4
sd2 = 7
ff = np.random.normal(loc=mu1, scale=sd1, size=int(N/2))
ff = np.append(ff, np.random.normal(loc=mu2, scale=sd2, size=int(N/2)))
# relative humidity
rh = 70 - ff + np.random.normal(loc=0, scale=5, size=N)

# prepare data
y = ff.copy()
x = np.nan
omega = rh.copy()

# init latent variable zn
zn = np.zeros_like(ff)
zn[ff >= np.median(ff)] = 1

# init distributional parameters
theta = np.array([np.quantile(ff, 0.25),
                  np.log(np.std(ff)),
                  np.quantile(ff, 0.75),
                  np.log(np.std(ff))])

# init parameter for logit model
glm = sm.GLM(zn, sm.add_constant(rh),
             family=sm.families.Binomial(link=sm.families.links.logit))
alpha = glm.fit().params

# Keep stuff for later
init = {'theta': theta, 'alpha': alpha}

# optimization
for i in range(10):
    # E Step
    prob = logitprob(omega, alpha)
    # cdf oder pdf?
    dnorm1 = scipy.stats.norm(loc=theta[0], scale=np.exp(theta[1])).pdf(y)
    dnorm2 = scipy.stats.norm(loc=theta[2], scale=np.exp(theta[3])).pdf(y)
    post = prob * dnorm2 / ((1-prob) * dnorm1 + prob * dnorm2)

    # M Step
    Q1 = scipy.optimize.minimize(Q1_fun, theta,
                                 method='Nelder-Mead',
                                 args=(y, post),
                                 options={'maxiter': 50})
    # print('niter Q1: %3d' % Q1.nit)

    Q2 = scipy.optimize.minimize(Q2_fun, alpha,
                                 method='Nelder-Mead',
                                 args=(omega, post),
                                 options={'maxiter': 50})
    # print('niter Q2: %3d' % Q2.nit)

    # update
    Q = Q1.fun + Q2.fun
    theta = Q1.x
    alpha = Q2.x

    print('EM step %2d: loglik = %10.3f' % (i, Q))


print('alpha: {}'.format(alpha))
#theta[1] = np.exp(theta[1])
#theta[3] = np.exp(theta[3])
print('theta: {}'.format(theta))
print('init: {}'.format(init))

