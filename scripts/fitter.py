#!/usr/bin/env python

import pickle
from timeit import default_timer as timer
from multiprocessing import Process

import pandas as pd
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import numdifftools as nd

from scipy.optimize import minimize
from scipy.stats import rv_continuous

from numba import jit

# global options
np.set_printoptions(precision=3.)

### Data manipulation ###
def scale_data(x, xlow=12., xhigh=70., invert=False):
    if not invert:
        return 2*(x - xlow)/(xhigh - xlow) - 1
    else:
        return 0.5*(x + 1)*(xhigh - xlow) + xlow
### PDF definitions ###
def gaussian(x, a):
    return 1./(np.sqrt(2*np.pi)*a[1])*np.exp(-0.5*(x - a[0])**2/a[1]**2)

def legendre_polynomial(x, a):
    ''' Second order Legendre polynomial with Michael's reparameterization and normalized to unity'''
    return 0.5 + a[0]*x + 0.5*a[1]*(3*x**2 - 1)

def combined_model(x, a):
    pdf_bg  = legendre_polynomial(x, (a[3], a[4]))
    pdf_sig = gaussian(x, (a[1], a[2]))
    return a[0]*pdf_bg + (1 - a[0])*pdf_sig

### Fitting tools ###
def bg_objective(a, X):
    pdf = legendre_polynomial(X, (a[0], a[1]))
    ll  = -np.sum(np.log(pdf))
    return ll

def bg_sig_objective(a, X):
    pdf_sig = gaussian(X, (a[1], a[2]))
    pdf_bg  = legendre_polynomial(X, (a[3], a[4]))
    pdf     = a[0]*pdf_bg + (1 - a[0])*pdf_sig
    ll      = -np.sum(np.log(pdf))
    return ll

def regularization(params, X, objective, lambda1 = 1., lambda2 = 1.):
    return objective(params, X) + lambda1 * np.sum(np.abs(params)) + lambda2 * np.sum(params**2)

def get_corr(f_obj, params, data):
    hcalc   = nd.Hessian(f_obj, step=0.01, method='central', full_output=True) 
    hobj    = hcalc(params, data)[0]
    hinv    = np.linalg.inv(hobj)

    # get uncertainties on parameters
    sig     = np.sqrt(hinv.diagonal())

    # calculate correlation matrix
    mcorr   = hinv/np.outer(sig, sig)

    return sig, mcorr

### For monte carlo studies ###
def mc_generator(pdf, samp_per_toy=1000, ntoys=1, domain=(-1.,1.)):
    '''Rejection sampling with broadcasting'''
    rnums = rng.rand(2, 2*ntoys*samp_per_toy) 
    x = rnums[0]
    x = (domain[1] - domain[0])*x + domain[0]
    keep = pdf(x) > rnums[1]
    x = x[keep]
    x = x[:-(x.size%samp_per_toy)]
    x = x.reshape(x.size/samp_per_toy, samp_per_toy)
    return x

def local_sig(p_bg, p_sigbg, ntoys, samp_per_toy):
    bg_pdf = lambda x: 0.5 + p_bg[0]*x + 0.5*p_bg[1]*(3*x**2 -1)
    sim = mc_generator(bg_pdf, samp_per_toy, ntoys)
    results = []
    nlls = []
    bnds = [(0., 1.05), # A
            2*(p_sigbg[1], ), 2*(p_sigbg[2], ), # mean, sigma
            2*(p_sigbg[3], ), 2*(p_sigbg[4], )] # a1, a2
    for toy in sim:
        res = minimize(regularization, 
                       [0.01, -0.3, 0.1, p_bg[0], p_bg[1]], 
                       method = 'SLSQP',
                       #jac    = True,
                       args   = (toy, bg_sig_objective, 0., 0.),
                       bounds = bnds)
        if res.success:
            results.append(res.x) 
            nlls.append(bg_sig_objective(res.x, toy))

    return sim, np.array(results), nlls

### Plotting scripts ###

def fit_plot(pdf, data, params):
    x = np.linspace(-1, 1, num=10000)
    y = (166.*2./29.)*pdf(x, params) 
    x = scale_data(x, invert=True)

    h = plt.hist(data, bins=29, range=[12., 70.], normed=False, histtype='step')
    bincenters  = (h[1][1:] + h[1][:-1])/2.
    binerrs     = np.sqrt(h[0]) 

    plt.clf()
    plt.errorbar(bincenters, h[0], yerr=binerrs, fmt='o')
    plt.plot(x, y, linewidth=2.)
    plt.title('mumu + 1 b jet + 1 forward jet')
    plt.xlabel('M_mumu [GeV]')
    plt.ylabel('entries / 2 GeV')
    plt.xlim([12., 70.])
    plt.ylim([0., 20.])
    plt.savefig('figures/dimuon_mass_fit.pdf')

    #plt.rc('text', usetex=True)
    #fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    #ax1 = axes[0]
    #ax1.errorbar(bincenters, h[0], yerr=binerrs, fmt='o')
    #ax1.plot(x, y)
    #ax1.set_xlabel('M_{\mu\mu} [GeV]')
    #ax1.set_ylabel('entries/2 GeV')
    #fig.show()

if __name__ == '__main__':
    # Start the timer
    start = timer()

    pout = False
    # get data and convert variables to be on the range [-1, 1]
    print 'Getting data and scaling to lie in range [-1, 1].'
    ntuple      = pd.read_csv('data/ntuple_1b1f.csv')
    data        = ntuple['dimuon_mass'].values
    data_scaled = np.apply_along_axis(scale_data, 0, data, xlow=12, xhigh=70)

    # fit background only model
    print 'Performing background only fit with second order Legendre polynomial normalized to unity.'
    a1 = 0.5
    a2 = 0.5
    bnds = [(0., 2.), (0., 0.5)] # a1, a2
    bg_result = minimize(regularization, 
                         [0.5, 0.05], 
                         method = 'SLSQP', 
                         bounds = bnds,
                         args   = (data_scaled, bg_objective))
    bg_sigma, bg_corr = get_corr(bg_objective, bg_result.x, data_scaled)   

    if pout:
        print '\n'
        print 'RESULTS'
        print '-------'
        print 'a0       = {0:.3f} +/- {1:.3f}'.format(bg_result.x[0], bg_sigma[0])
        print 'a1       = {0:.3f} +/- {1:.3f}'.format(bg_result.x[1], bg_sigma[1])
        print'\n'
        print 'correlation matrix:'
        print bg_corr
        print'\n'

    # fit signal+background model
    print 'Performing background plus signal fit with second order Legendre polynomial normalized to unity plus a Gaussian kernel.'
    bnds = [(0., 1.), # A
            (-0.8, -0.2), (0., 0.4), # mean, sigma
            (0., 2.), (0., 0.5)] # a1, a2
    result = minimize(regularization, 
                      [0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1]], 
                      method = 'SLSQP',
                      #jac    = True,
                      args   = (data_scaled, bg_sig_objective, 1., 1.),
                      bounds = bnds)
    comb_sigma, comb_corr = get_corr(bg_sig_objective, result.x, data_scaled)   
    qtest = 2*np.abs(bg_sig_objective(result.x, data_scaled) - bg_objective(bg_result.x, data_scaled))

    # Convert back to measured mass values
    pct_sigma   = np.abs(comb_sigma/result.x)
    mu          = scale_data(result.x[1], invert=True) 
    sig_mu      = mu*pct_sigma[1]
    width       = result.x[2]*(70. - 12.)/2. 
    sig_width   = width*pct_sigma[2]

    if pout:
        print '\n'
        print 'RESULTS'
        print '-------'
        print 'A        = {0:.3f} +/- {1:.3f}'.format(result.x[0], comb_sigma[0])
        print 'mu       = {0:.3f} +/- {1:.3f}'.format(mu, sig_mu)
        print 'width    = {0:.3f} +/- {1:.3f}'.format(width, sig_width)
        print 'a0       = {0:.3f} +/- {1:.3f}'.format(result.x[3], comb_sigma[3])
        print 'a1       = {0:.3f} +/- {1:.3f}'.format(result.x[4], comb_sigma[4])
        print'\n'
        print 'correlation matrix:'
        print comb_corr
        print'\n'
        print 'q = {0:.3f}'.format(qtest)

    fit_plot(combined_model, data, result.x)

    ### TOY MC ###
    proc = Process(name='test_process', target=local_sig, args=(bg_result.x, result.x, 10000))
    mc_toys, mc_results, mc_lls = local_sig(result.x[3:], result.x, ntoys=10000, samp_per_toy=166)
    plt.clf()
    #plt.yscale('log')
    plt.hist(1-mc_results[:,0], bins=50, range=[-0.2, 0.2], histtype='stepfilled')
    plt.show()

    toy_file = open('data/toy_ouput.pkl', 'wb')
    pickle.dump(mc_toys, toy_file)
    pickle.dump(mc_results, toy_file)
    toy_file.close()
    print timer() - start
