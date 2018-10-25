import numpy as np
import scipy


def Q1a_fun(par, y, post):
    dnorm1 = scipy.stats.norm(loc=par[0], scale=np.exp(par[1])).logpdf(y)
    return -np.sum((1-post) * dnorm1)


def Q1a_grad(par, y, post):
    mu1 = par[0]
    si1 = np.exp(par[1])
    gmu1 = (1 - post) * (y - mu1) / si1**2
    gsi1 = (1 - post) * ((y - mu1)**2 - si1**2) / si1**3 * si1
    return -np.array([gmu1.sum(), gsi1.sum()])


def Q1b_fun(par, y, post):
    dnorm2 = scipy.stats.norm(loc=par[0], scale=np.exp(par[1])).logpdf(y)
    return -np.sum(post * dnorm2)


def Q1b_grad(par, y, post):
    mu2 = par[0]
    si2 = np.exp(par[1])
    gmu2 = post * (y - mu2) / si2**2
    gsi2 = post * ((y - mu2)**2 - si2**2) / si2**3 * si2
    return -np.array([gmu2.sum(), gsi2.sum()])


def Q2_fun(par, omega, post):
    prob = logitprob(omega, par)
    return -np.sum((1-post) * np.log(1-prob) + post * np.log(prob))


def Q2_grad(par, omega, post):
    x = par[0] + omega * par[1]
    prob = np.exp(x) / (1 + np.exp(x))
    gr = (post / prob - (1-post) / (1-prob)) * np.exp(x) / (1 + np.exp(x))**2
    return -np.array([np.sum(gr), np.sum(gr*omega)])


def logitprob(x, alpha):
    x = alpha[0] + x * alpha[1]
    return np.exp(x) / (1 + np.exp(x))
