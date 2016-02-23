#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
import numdifftools as nd

from scipy.optimize import minimize
from scipy.stats import rv_discrete

#class background_gen(rv_discrete):
#    def _pmf(self, a1, a2):
#        return 0.5 + a1*x + 0.5*a2*(3*x**2 - 1)

def scale_data(x, xlow, xhigh, invert=False):
    if not invert:
        return 2*(x - xlow)/(xhigh - xlow) - 1
    else:
        return 0.5*(x + 1)*(xhigh - xlow) + xlow

def gaussian(x, mu=0, sigma=1):
    return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*(x - mu)**2/sigma**2)

def legendre_polynomial(x, a1=0.1, a2=0.1):
    ''' Legendre polynomial with Michael's reparameterization and normalized to unity'''
    return 0.5 + a1*x + 0.5*a2*(3*x**2 - 1)

def combined_model(x, params):
    pdf_bg  = legendre_polynomial(x, a1=params[3], a2=params[4])
    pdf_sig = gaussian(x, mu=params[1], sigma=params[2])
    return params[0]*pdf_bg + (1 - params[0])*pdf_sig

def bg_objective(params, X, data=None, eps=None):
    pdf = legendre_polynomial(X, a1 = params[0], a2 = params[1])
    ll  = -np.sum(np.log(pdf))
    return ll

def bg_sig_objective(params, X, data=None, eps=None):
    pdf_sig = gaussian(X, mu = params[1], sigma = params[2])
    pdf_bg  = legendre_polynomial(X, a1 = params[3], a2 = params[4])
    pdf     = params[0]*pdf_bg + (1 - params[0])*pdf_sig
    ll      = -np.sum(np.log(pdf))
    return ll

def regularization(params, X, objective, lambda1 = 1., lambda2 = 1.):
    return objective(params, X) + lambda1 * np.sum(np.abs(params)) + lambda2 * np.sum(params**2)

def fit_plot(pdf, data, params):
    x = np.linspace(-1, 1, num=1000)
    plt.plot(x, pdf(x, params))
    h = plt.hist(data, bins=29, range=[-1., 1.], normed=True, histtype='step')
    plt.show()


if __name__ == '__main__':

    # get data and convert variables to be on the range [-1, 1]
    data = pd.read_csv('data/dimuon_mass_1b1f.txt', header=None)[0].values
    data_scaled = np.apply_along_axis(scale_data, 0, data, xlow=12, xhigh=70)

    # Initialize model parameters
    bg_params   = {'a0': 1., 'a1':1., 'a2':1.}
    sig_params  = {'mu': 0.01, 'mean':30., 'sigma':1.}

    # fit background only model
    a1 = 0.5
    a2 = 0.5
    bnds = [(0., 2.), (0., 0.5)] # a1, a2
    bg_result = minimize(regularization, 
                         [0.5, 0.05], 
                         method = 'SLSQP', 
                         bounds = bnds,
                         args   = (data_scaled, bg_objective))

    # fit signal+background model
    bnds = [(0., 1.), # A
            (-0.8, -0.2), (0., 0.4), # mean, sigma
            (0., 2.), (0., 0.5)] # a1, a2
    result = minimize(regularization, 
                      [0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1]], 
                      method = 'SLSQP',
                      #jac    = True,
                      args   = (data_scaled, bg_sig_objective),
                      bounds = bnds)

    print result
    qtest = np.sqrt(2*np.abs(bg_sig_objective(result.x, data_scaled) - bg_objective(bg_result.x, data_scaled)))
    print 'q = {0}'.format(qtest)

    '''
    ### test space ###
    data = rng.normal(loc=50., scale=5., size=10000)
    x = rng.rand(10000)*100.
    y = gaussian(x, 50., 5.)

    #cons    = ({'type': 'ineq', 'fun': lambda p: p[0] > 12.},
    #           {'type': 'ineq', 'fun': lambda p: p[0] < 70.},
    #           {'type': 'ineq', 'fun': lambda p: p[0] < 70.})
    #bnds    = ((0., 100.), (0., 10.))
    result = minimize(test_objective, [35., 4.], method='nelder-mead', args=(data,))
    #result = minimize(test_objective, [35., 4.], method='Newton-CG', args=(data,), bounds=bnds)
    '''
