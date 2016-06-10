#!/usr/bin/env python

import pickle
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
import numdifftools as nd
import lmfit
#from scipy import stats
from scipy.stats import chi2, norm 
from scipy import integrate
from scipy.optimize import minimize

# global options
np.set_printoptions(precision=3.)

### Data manipulation ###
def scale_data(x, xmin=12., xmax=70., invert=False):
    if not invert:
        return 2*(x - xmin)/(xmax - xmin) - 1
    else:
        return 0.5*(x + 1)*(xmax - xmin) + xmin

def get_data(filename, varname, xlim):
    '''
    Get data from file and convert to lie in the range [-1, 1]
    '''
    ntuple  = pd.read_csv(filename)
    data    = ntuple[varname].values
    data    = data[np.all([(data > xlim[0]), (data < xlim[1])], axis=0)]
    data    = np.apply_along_axis(scale_data, 0, data, xmin=xlim[0], xmax=xlim[1])
    n_total = data.size

    return data, n_total
  
def get_corr(func, a):
    '''
    Given a function func and parameters a, this will numerically estimate the
    covariance of the parameters about the given values
    
    Parameters:
    ===========
    func : function used to estimate covariance of parameters
    a    : parameters
    '''

    hcalc   = nd.Hessian(func, step=0.01, method='central', full_output=True) 
    hobj    = hcalc(a)[0]
    hinv    = np.linalg.inv(hobj)

    # get uncertainties on parameters
    sig = np.sqrt(hinv.diagonal())
    
    # calculate correlation matrix
    mcorr = hinv/np.outer(sig, sig)

    return sig, mcorr

def kolmogorov_smirinov(data, model_pdf, xlim=(-1, 1), npoints=10000):
    
    xvals = np.linspace(xlim[0], xlim[1], npoints)

    #Calculate CDFs 
    data_cdf = np.array([data[data < x].size for x in xvals]).astype(float)
    data_cdf = data_cdf/data.size

    model_pdf = model_pdf(xvals)
    model_cdf = np.cumsum(model_pdf)*(xlim[1] - xlim[0])/npoints

    return np.abs(model_cdf - data_cdf)

def calculate_likelihood_ratio(bg_model, s_model, data):
    '''
    '''
    bg_nll  = bg_model.nll(data)
    s_nll   = s_model.nll(data)

    return 2*(bg_nll - s_nll)


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
def fit_plot(data, xlim, sig_model, bg_model, suffix, path='plots'):
#def fit_plot(data, xlim, sig_pdf, params, bg_pdf, bg_params, suffix, path='plots'):
    N       = data.size
    binning = 2.
    nbins   = int((xlim[1] - xlim[0])/binning)

    # Scale pdfs and data from [-1, 1] back to the original values
    params = sig_model.get_parameters()
    x       = np.linspace(-1, 1, num=10000)
    y_sig   = (N*binning/nbins)*sig_model.pdf(x) 
    y_bg1   = (1 - params['A']) * N * binning/nbins * bg_model.pdf(x, params) 
    y_bg2   = (N*binning/nbins)*bg_model.pdf(x)
    x       = scale_data(x, xmin=xlim[0], xmax=xlim[1],invert=True)
    data    = scale_data(data, xmin=xlim[0], xmax=xlim[1],invert=True)

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
        ax.set_xlabel(r'$m_{\mu\mu}$ [GeV]')
        ax.set_ylabel('entries / 2 GeV')
    elif suffix[:4] == '1b1c':
        ax.set_title(r'$\mu\mu$ + 1 b jet + 1 central jet + MET < 40 + $\Delta\phi (\mu\mu ,bj)$')
        ax.set_ylim([0., 50.])
        ax.set_xlabel(r'$m_{\mu\mu}$ [GeV]')
        ax.set_ylabel('entries / 2 GeV')
    elif suffix[:4] == 'hgg':
        ax.set_title(r'$h(125)\rightarrow \gamma\gamma$')
        #ax.set_ylim([0., 50.])
        ax.set_xlabel(r'$m_{\gamma\gamma}$ [GeV]')
        ax.set_ylabel('entries / 2 GeV')

    ax.set_xlim(xlim)

    fig.savefig('{0}/dimuon_mass_fit_{1}.pdf'.format(path, suffix))
    fig.savefig('{0}/dimuon_mass_fit_{1}.png'.format(path, suffix))
    plt.close()


