'''
    Miscellaneous tools and helper functions for the nllfitting framework.
'''

#!/usr/bin/env python

from __future__ import division
import pickle
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
import numdifftools as nd
import emcee as mc
import lmfit
#from scipy import stats
from scipy.stats import chi2, norm 
from scipy import integrate
from scipy.optimize import minimize
from scipy.special import wofz

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
  
def kolmogorov_smirinov(data, model_pdf, xlim=(-1, 1), npoints=10000):
    
    xvals = np.linspace(xlim[0], xlim[1], npoints)

    #Calculate CDFs 
    data_cdf = np.array([data[data < x].size for x in xvals]).astype(float)
    data_cdf = data_cdf/data.size

    model_pdf = model_pdf(xvals)
    model_cdf = np.cumsum(model_pdf)*(xlim[1] - xlim[0])/npoints

    return np.abs(model_cdf - data_cdf)

### PDF definitions (maybe put these in a separate file)
def lorentzian(x, a):
    '''
    Lorentzian line shape

    Parameters:
    ===========
    x: data
    a: model parameters (mean and HWHM)
    '''
    return a[1]/(np.pi*((x-a[0])**2 + a[1]**2))

def voigt(x, a):
    '''
    Voigt profile

    Parameters:
    ===========
    x: data
    a: model paramters (mean, gamma, and sigma)
    '''
    z = ((x - a[0]) + 1j*a[1])/(a[2]*np.sqrt(2))

    v = np.real(wofz(z))/(a[2]*np.sqrt(2*np.pi))
    return v


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

def sig_pdf_alt(x, a):
    '''
    Second order Legendre Polynomial (normalized to unity) plus a Voigt
    profile. N.B. The width of the convolutional Gaussian is set to 0.17 which
    corresponds to a dimuon mass resolution 0.5 GeV.

    Parameters:
    ===========
    x: data
    a: model parameters (a1, a2, mu, and gamma)
    '''
    return (1 - a[0])*bg_pdf(x, a[3:5]) + a[0]*voigt(x, [a[1], a[2], 0.0155])


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
def fit_plot(data, xlim, sig_model, bg_model, suffix, path='plots', show=False):
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
    ax.plot(x, y_sig, 'b-', linewidth=2.)
    ax.plot(x, y_bg1, 'b--', linewidth=2.) 
    ax.plot(x, y_bg2, 'r-.', linewidth=2.) 
    ax.errorbar(bincenters, h[0], yerr=binerrs, fmt='ko')
    ax.legend(['bg+sig.', 'bg', 'bg only', 'data']) 

    if suffix[:4] == '1b1f':
        ax.set_title(r'$\mu\mu$ + 1 b jet + 1 forward jet')
        ax.set_ylim([0., 1.5*np.max(h[0])])
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

    if show:
        plt.show()
    else:
        fig.savefig('{0}/dimuon_mass_fit_{1}.pdf'.format(path, suffix))
        fig.savefig('{0}/dimuon_mass_fit_{1}.png'.format(path, suffix))
        plt.close()


### Monte Carlo simulations ###
def lnprob(x, pdf, bounds):
    if np.any(x < bounds[0]) or np.any(x > bounds[1]):
        return -np.inf
    else:
        return np.log(pdf(x))

def generator_emcee(pdf, samples_per_toy=100, ntoys=100, bounds=(-1, 1)):
    '''
    Wrapper for emcee the MCMC hammer (only does 1D distributions for now...)

    Parameters
    ==========
    pdf             : distribution to be sampled
    samples_per_toy : number of draws to be assigned to each pseudo-experiment
    ntoys           : number of toy models to produce
    bounds          : (xmin, xmax) for values of X
    '''
    ndim = 1
    sampler = mc.EnsembleSampler(samples_per_toy, ndim, lnprob, args=[pdf, bounds])

    p0 = [np.random.rand(1) for i in xrange(samples_per_toy)]
    pos, prob, state = sampler.run_mcmc(p0, 1000) # Let walkers settle in
    sampler.reset()
    sampler.run_mcmc(pos, ntoys, rstate0=state)

    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
    print("Autocorrelation time:", sampler.get_autocorr_time())

    return sampler.flatchain[:, 0].reshape(ntoys, samples_per_toy)

def generator(pdf, samples_per_toy=100, ntoys=1, bounds=(-1.,1.)):
    '''
    Rejection sampling with broadcasting gives approximately the requested
    number of toys.  This works okay for simple pdfs.

    Parameters:
    ===========
    pdf             : the pdf that will be sampled to produce the synthetic data
    samples_per_toy : number of datapoint per toy dataset
    ntoys           : number of synthetic datasets to be produced
    bounds          : specify (lower, upper) bounds for the toy data
    '''

    # Generate random numbers and map into domain defined by bounds.  Generate
    # twice the number of requested events in expectation of ~50% efficiency.
    # This will not be the case for more complicated pdfs presumably
    rnums = rng.rand(2, 2*ntoys*samples_per_toy) 
    x = rnums[0]
    x = (bounds[1] - bounds[0])*x + bounds[0]

    # Carry out rejection sampling
    keep = pdf(x) > rnums[1]
    x    = x[keep]
    
    # Remove excess events and shape to samples_per_toy.
    x = x[:-(x.size%samples_per_toy)]
    x = x.reshape(x.size/samples_per_toy, samples_per_toy)

    # if the exact number of toy datasets are not generated either trim or
    # produce more.
    ndata = x.shape[0]
    if ndata < ntoys:
        xplus = generator(pdf, samples_per_toy, (ntoys-ndata), bounds)
        x = np.concatenate((x, xplus))
    elif ndata > ntoys:
        x = x[:ntoys,]

    return x

def calculate_CI(bg_fitter, sig_fitter):

    ### Calculate confidence interval on the likelihood ratio at the +/- 1, 2
    ### sigma levels
    nsims = 1000

    print 'Generating {0} pseudo-datasets from bg+signal fit and determining distribution of q'.format(nsims)
    sims = ft.generator(sig_model.pdf, n_total, ntoys=nsims)
    q_sim = []
    for sim in sims:
        bg_result = bg_fitter.fit(sim)
        sig_result = sig_fitter.fit(sim) 
        if bg_result.status == 0 and sig_result.status == 0:
            nll_bg = bg_model.calc_nll(sim)
            nll_sig = sig_model.calc_nll(sim)
            q = 2*(nll_bg - nll_sig)
            q_sim.append(q)
        else:
            print bg_result.status, sig_result.status

    q_sim = np.array(q_sim)
    q_sim.sort()
    q_upper = q_sim[q_sim > q_max]
    q_lower = q_sim[q_sim < q_max]

    n_upper = q_upper.size
    n_lower = q_lower.size

    one_sigma_up   = q_upper[int(0.34*n_upper)]
    two_sigma_up   = q_upper[int(0.475*n_upper)]
    one_sigma_down = q_lower[int(-0.34*n_lower)]
    two_sigma_down = q_lower[int(-0.475*n_lower)]

    print '{0}: q = {1:.2f}'.format(channel, q_max)
    print '1 sigma c.i.: {0:.2f} -- {1:.2f}'.format(one_sigma_down, one_sigma_up)
    print '2 sigma c.i.: {0:.2f} -- {1:.2f}'.format(two_sigma_down, two_sigma_up)


