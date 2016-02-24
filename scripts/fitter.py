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

def scale_data(x, xlow=12., xhigh=70., invert=False):
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
    x = np.linspace(-1, 1, num=10000)
    y = (166.*2./29.)*pdf(x, params) 
    x = scale_data(x, invert=True)

    h = plt.hist(data, bins=29, range=[12., 70.], normed=False, histtype='step')
    bincenters  = (h[1][1:] + h[1][:-1])/2.
    binerrs     = np.sqrt(h[0]) 

    plt.clf()
    plt.errorbar(bincenters, h[0], yerr=binerrs, fmt='o')
    plt.plot(x, y)
    plt.title('mumu + 1 b jet + 1 forward jet')
    plt.xlabel('M_mumu [GeV]')
    plt.ylabel('entries / 2 GeV')
    plt.savefig('figures/dimuon_mass_fit.pdf')

    #plt.rc('text', usetex=True)
    #fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    #ax1 = axes[0]
    #ax1.errorbar(bincenters, h[0], yerr=binerrs, fmt='o')
    #ax1.plot(x, y)
    #ax1.set_xlabel('M_{\mu\mu} [GeV]')
    #ax1.set_ylabel('entries/2 GeV')
    #fig.show()

def get_corr(f_obj, params, data):
    hcalc   = nd.Hessian(f_obj, step=0.01, method='central', full_output=True) 
    hobj    = hcalc(params, data)[0]
    hinv    = np.linalg.inv(hobj)

    # get uncertainties on parameters
    sig     = np.sqrt(hinv.diagonal())

    # calculate correlation matrix
    mcorr   = hinv/np.outer(sig, sig)

    return sig, mcorr

if __name__ == '__main__':

    # get data and convert variables to be on the range [-1, 1]
    data = pd.read_csv('data/dimuon_mass_1b1f.txt', header=None)[0].values
    data_scaled = np.apply_along_axis(scale_data, 0, data, xlow=12, xhigh=70)

    # fit background only model
    a1 = 0.5
    a2 = 0.5
    bnds = [(0., 2.), (0., 0.5)] # a1, a2
    bg_result = minimize(regularization, 
                         [0.5, 0.05], 
                         method = 'SLSQP', 
                         bounds = bnds,
                         args   = (data_scaled, bg_objective))
    bg_sigma, bg_corr = get_corr(bg_objective, bg_result.x, data_scaled)   

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
    comb_sigma, comb_corr = get_corr(bg_sig_objective, result.x, data_scaled)   
    qtest = np.sqrt(2*np.abs(bg_sig_objective(result.x, data_scaled) - bg_objective(bg_result.x, data_scaled)))

    pct_sigma = np.abs(comb_sigma/result.x)
    mu      = scale_data(result.x[1], invert=True) 
    sig_mu  = mu*pct_sigma[1]
    width   = result.x[2]*(70. - 12.)/2. 
    sig_wid = width*pct_sigma[2]

    np.set_printoptions(precision=3.)
    print '\n'
    print 'RESULTS'
    print '-------'
    print 'A        = {0:.3f} +/- {1:.3f}'.format(result.x[0], comb_sigma[0])
    print 'mu       = {0:.3f} +/- {1:.3f}'.format(mu, sig_mu)
    print 'width    = {0:.3f} +/- {1:.3f}'.format(width, sig_wid)
    print 'a0       = {0:.3f} +/- {1:.3f}'.format(result.x[3], comb_sigma[3])
    print 'a1       = {0:.3f} +/- {1:.3f}'.format(result.x[4], comb_sigma[4])
    print'\n'
    print 'correlation matrix:'
    print comb_corr
    print'\n'
    print 'q = {0}'.format(qtest)

    fit_plot(combined_model, data, result.x)

