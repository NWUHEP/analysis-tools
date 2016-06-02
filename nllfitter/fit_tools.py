#!/usr/bin/env python

import pickle
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy.random as rng
import numdifftools as nd

#from scipy import stats
from scipy.stats import chi2, norm 
from scipy import integrate
from scipy.optimize import minimize


# global options
np.set_printoptions(precision=3.)

### Data manipulation ###
def scale_data(x, xlow=12., xhigh=70., invert=False):
    if not invert:
        return 2*(x - xlow)/(xhigh - xlow) - 1
    else:
        return 0.5*(x + 1)*(xhigh - xlow) + xlow

def get_data(filename, varname, xlim):
    '''
    Get data from file and convert to lie in the range [-1, 1]
    '''
    ntuple  = pd.read_csv(filename)
    data    = ntuple[varname].values
    data    = data[np.all([(data > xlim[0]), (data < xlim[1])], axis=0)]
    data    = np.apply_along_axis(scale_data, 0, data, xlow=xlim[0], xhigh=xlim[1])
    n_total = data.size

    return data, n_total

### PDF definitions ###
def bg_pdf(x, a): 
    '''
    Second order Legendre Polynomial with constant term set to 0.5.

    Parameters:
    ===========
    x: data
    a: model parameters (a1 and a2)
    '''
    return 0.5 + a[0]*x + 0.5*a[1]*(3*x**2 - 1)

def sig_pdf(x, a):
    '''
    Second order Legendre Polynomial (normalized to unity) plus a Gaussian.

    Parameters:
    ===========
    x: data
    a: model parameters (a1, a2, mu, and sigma)
    '''
    return (1 - a[0])*bg_pdf(x, a[3:5]) + a[0]*norm.pdf(x, a[1], a[2])

### toy MC p-value calculator ###
def calc_local_pvalue(N_bg, var_bg, N_sig, var_sig, ntoys=1e7):
    print ''
    print 'Calculating local p-value and significance based on {0} toys...'.format(int(ntoys))
    toys    = rng.normal(N_bg, var_bg, int(ntoys))
    pvars   = rng.poisson(toys)
    pval    = pvars[pvars > N_bg + N_sig].size/pvars.size
    print 'local p-value = {0}'.format(pval)
    print 'local significance = {0:.2f}'.format(np.abs(norm.ppf(pval)))

    return pval

### Plotter ###
def fit_plot(data, xlim, sig_pdf, params, bg_pdf, bg_params, suffix, path='plots'):
    N       = data.size
    binning = 2.
    nbins   = int((xlim[1] - xlim[0])/binning)

    # Scale pdfs and data from [-1, 1] back to the original values
    x       = np.linspace(-1, 1, num=10000)
    y_sig   = (N*binning/nbins)*sig_pdf(x, params) 
    y_bg1   = (1 - params[0]) * N * binning/nbins * bg_pdf(x, params[3:]) 
    y_bg2   = (N*binning/nbins)*bg_pdf(x, bg_params) 
    x       = scale_data(x, xlow=xlim[0], xhigh=xlim[1],invert=True)

    # Get histogram of data points
    h = plt.hist(data, bins=nbins, range=xlim, normed=False, histtype='step')
    bincenters  = (h[1][1:] + h[1][:-1])/2.
    binerrs     = np.sqrt(h[0]) 
    plt.close()

    fig, ax = plt.subplots()
    ax.errorbar(bincenters, h[0], yerr=binerrs, fmt='ko')
    ax.plot(x, y_sig, 'b-', linewidth=2.)
    ax.plot(x, y_bg1, 'b--', linewidth=2.) 
    ax.plot(x, y_bg2, 'r-.', linewidth=2.) 

    if suffix[:4] == '1b1f':
        ax.set_title(r'$\mu\mu$ + 1 b jet + 1 forward jet')
        ax.set_ylim([0., 25.])
    elif suffix[:4] == '1b1c':
        ax.set_title(r'$\mu\mu$ + 1 b jet + 1 central jet + MET < 40 + $\Delta\phi (\mu\mu ,bj)$')
        ax.set_ylim([0., 50.])
    ax.set_xlabel(r'$m_{\mu\mu}$ [GeV]')
    ax.set_ylabel('entries / 2 GeV')
    ax.set_xlim(xlim)

    fig.savefig('{0}/dimuon_mass_fit_{1}.pdf'.format(path, suffix))
    fig.savefig('{0}/dimuon_mass_fit_{1}.png'.format(path, suffix))
    plt.close()


