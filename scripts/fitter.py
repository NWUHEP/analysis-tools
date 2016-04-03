#!/usr/bin/env python

import pickle
from timeit import default_timer as timer
from multiprocessing import Process

import pandas as pd
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import numdifftools as nd

from scipy.stats import chi2, norm 
from scipy import integrate
from scipy.optimize import minimize

from numba import jit

# global options
np.set_printoptions(precision=3.)

### Data manipulation ###
def scale_data(x, xlow=12., xhigh=70., invert=False):
    if not invert:
        return 2*(x - xlow)/(xhigh - xlow) - 1
    else:
        return 0.5*(x + 1)*(xhigh - xlow) + xlow

def get_data(filename, varname, xlim):
    ntuple  = pd.read_csv(filename)
    data    = ntuple[varname].values
    data    = np.apply_along_axis(scale_data, 0, data, xlow=xlim[0], xhigh=xlim[1])
    n_total = data.size

    return data, n_total

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
    sig = np.sqrt(hinv.diagonal())

    # calculate correlation matrix
    mcorr = hinv/np.outer(sig, sig)

    return sig, mcorr

### toy MC p-value calculator ###
def calc_local_pvalue(N_bg, N_sig, var_bg, ntoys=1e7):
    print ''
    print 'Calculating local p-value and significance based on {0}...'.format(ntoys)
    toys    = rng.normal(N_bg, var_bg, int(ntoys))
    pvars   = rng.poisson(toys)
    pval    = pvars[pvars > N_bg + N_sig].size/ntoys
    print 'local p-value = {0}'.format(pval)
    print 'local significance = {0:.2f}'.format(np.abs(norm.ppf(pval)))

    return pval

### Plotter ###
def fit_plot(data, sig_pdf, params, bg_pdf, bg_params, suffix, path='figures'):
    N       = data.size
    nbins   = 29.
    binning = 2.

    x       = np.linspace(-1, 1, num=10000)
    y_sig   = (N*binning/nbins)*sig_pdf(x, params) 
    y_bg1   = (params[0]*N*binning/nbins)*bg_pdf(x, params[-2:]) 
    y_bg2   = (N*binning/nbins)*bg_pdf(x, bg_params) 
    x       = scale_data(x, invert=True)

    h = plt.hist(data, bins=nbins, range=[12., 70.], normed=False, histtype='step')
    bincenters  = (h[1][1:] + h[1][:-1])/2.
    binerrs     = np.sqrt(h[0]) 
    plt.close()

    fig, ax = plt.subplots()
    ax.errorbar(bincenters, h[0], yerr=binerrs, fmt='ko')
    ax.plot(x, y_sig, 'b-', linewidth=2.)
    ax.plot(x, y_bg1, 'b--', linewidth=2.) 
    ax.plot(x, y_bg2, 'r-.', linewidth=2.) 

    if suffix[:4] == '1b1f':
        ax.set_title('mumu + 1 b jet + 1 forward jet')
        ax.set_ylim([0., 25.])
    elif suffix[:4] == '1b1c':
        ax.set_title('mumu + 1 b jet + 1 central jet + MET < 40 + deltaPhi(mumu,bj)')
        ax.set_ylim([0., 50.])
    ax.set_xlabel('M_mumu [GeV]')
    ax.set_ylabel('entries / 2 GeV')
    ax.set_xlim([12., 70.])

    #plt.rc('text', usetex=True)
    #fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    #ax1 = axes[0]
    #ax1.errorbar(bincenters, h[0], yerr=binerrs, fmt='o')
    #ax1.plot(x, y)
    #ax1.set_xlabel('M_{\mu\mu} [GeV]')
    #ax1.set_ylabel('entries/2 GeV')
    #fig.show()

    fig.savefig('{0}/dimuon_mass_fit_{1}.pdf'.format(path, suffix))
    fig.savefig('{0}/dimuon_mass_fit_{1}.png'.format(path, suffix))
    plt.close()

if __name__ == '__main__':
    # Start the timer
    start = timer()
    pout  = True

    # get data and convert variables to be on the range [-1, 1]
    print 'Getting data and scaling to lie in range [-1, 1].'
    minalgo     = 'SLSQP'
    channel     = '1b1c'
    xlimits     = (12, 70)

    data, n_total = get_data('data/ntuple_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    print 'Analyzing {0} events...'.format(n_total)

    # fit background only model
    print 'Performing background only fit with second order Legendre polynomial normalized to unity.'
    bnds = [(0., 1.), (0., 1.)] # a1, a2
    bg_result = minimize(regularization, 
                         (0.5, 0.05), 
                         method = minalgo, 
                         jac    = False,
                         bounds = bnds,
                         args   = (data, bg_objective)
                         )
    bg_sigma, bg_corr = get_corr(bg_objective, bg_result.x, data)   

    if pout:
        print '\n'
        print 'RESULTS'
        print '-------'
        print 'a1       = {0:.3f} +/- {1:.3f}'.format(bg_result.x[0], bg_sigma[0])
        print 'a2       = {0:.3f} +/- {1:.3f}'.format(bg_result.x[1], bg_sigma[1])
        print'\n'
        print 'correlation matrix:'
        print bg_corr
        print'\n'

    # fit signal+background model
    print 'Performing background plus signal fit with second order Legendre polynomial normalized to unity plus a Gaussian kernel.'
    bnds = [(0., 1.05), # A
            (-0.8, -0.2), (0., 0.4), # mean, sigma
            (0., 2.), (0., 0.5)] # a1, a2
    result = minimize(regularization, 
                      (0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1]), 
                      method = minalgo,
                      jac    = False,
                      args   = (data, bg_sig_objective),
                      bounds = bnds
                      )
    comb_sigma, comb_corr = get_corr(bg_sig_objective, result.x, data)   
    qtest = np.sqrt(2*np.abs(bg_sig_objective(result.x, data) - bg_objective(bg_result.x, data)))

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
        print 'a1       = {0:.3f} +/- {1:.3f}'.format(result.x[3], comb_sigma[3])
        print 'a2       = {0:.3f} +/- {1:.3f}'.format(result.x[4], comb_sigma[4])
        print'\n'
        print 'Correlation matrix:'
        print comb_corr
        print'\n'

    #=======================#
    ### Caluculate yields ###
    #=======================#
    # integrate over background only function in the range (mu - 2*sigma, mu +
    # 2*sigma) to determine background yields.  Signal yields come from
    # N*(1-A).
    f_bg    = lambda x: legendre_polynomial(x, (result.x[3], result.x[4]))
    xlim    = (result.x[1] - 2*result.x[2], result.x[1] + 2*result.x[2])
    N_b     = result.x[0]*n_total*integrate.quad(f_bg, xlim[0], xlim[1])[0]
    sig_b   = N_b/np.sqrt(n_total*result.x[0])
    N_s     = n_total*(1 - result.x[0]) 
    sig_s   = n_total*comb_sigma[0]

    print 'N_b = {0:.2f} +\- {1:.2f}'.format(N_b, sig_b)
    print 'N_s = {0:.2f} +\- {1:.2f}'.format(N_s, sig_s)
    print 'q = {0:.3f}'.format(qtest)

    ### Simple local p-value ###
    calc_local_pvalue(N_b, N_s, sig_b, 1e8)

    ### Make plots ###
    fit_plot(scale_data(data, invert=True), combined_model, result.x, legendre_polynomial, bg_result.x, channel)

    print ''
    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
