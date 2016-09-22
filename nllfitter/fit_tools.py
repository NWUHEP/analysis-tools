#!/usr/bin/env python

from __future__ import division
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import chi2, norm 
from scipy import integrate
from scipy.optimize import minimize
from scipy.special import wofz
#import ROOT as r # Need this for validating scipy Faddeeva function implementation

import emcee as mc
import numdifftools as nd
import lmfit

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
  
def ks_test(data, model_pdf, xlim=(-1, 1), make_plots=False, suffix=None):
    '''
    Kolmogorov-Smirnov test.  Returns the residuals of |CDF_model - CDF_data|.
    '''
    
    n_points = 1e5
    x = np.linspace(xlim[0], xlim[1], n_points)
    pdf = model_pdf(x)
    cdf = np.cumsum(pdf)*(xlim[1] - xlim[0])/n_points

    data.sort()
    x_i = np.array([np.abs(d - x).argmin() for d in data])
    cdf_i = np.linspace(1, data.size, data.size)/data.size

    ks_residuals = np.abs(cdf[x_i] - cdf_i)

    if make_plots:
        plt.hist(ks_residuals, bins=25, histtype='step')
        plt.ylabel('Entries')
        plt.xlabel(r'$|\rm CDF_{model} - CDF_{data}|$')
        plt.savefig('plots/ks_residuals_{0}.pdf'.format(suffix))
        plt.close()

        plt.plot(x, cdf)
        plt.plot(data, cdf_i)
        plt.ylabel('CDF(x)')
        plt.xlabel('x')
        plt.title(suffix)
        plt.legend(['model', 'data'])
        plt.savefig('plots/ks_cdf_overlay_{0}.pdf'.format(suffix))
        plt.close()

    return ks_residuals

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
    mu = a[0]
    gamma = a[1]
    sigma = a[2]

    if gamma == 0:
        return norm.pdf(x, [mu, sigma])
    elif sigma == 0:
        return lorentzian(x, [mu, gamma])
    else:
        z = ((x - mu) + 1j*gamma)/(sigma*np.sqrt(2))
        y = np.real(wofz(z))/(sigma*np.sqrt(2*np.pi))
        return y


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

    bg = bg_pdf(x, a[3:5])
    sig = norm.pdf(x, a[1], a[2]) 
    sig_norm = integrate.quad(lambda z: norm.pdf(z, a[1], a[2]), -1, 1)[0]

    return (1 - a[0])*bg + a[0]*sig/sig_norm

def sig_pdf_alt(x, a):
    '''
    Second order Legendre Polynomial (normalized to unity) plus a Voigt
    profile. N.B. The width of the convolutional Gaussian is set to 0.17 which
    corresponds to a dimuon mass resolution 0.5 GeV.

    Parameters:
    ===========
    x: data
    a: model parameters (A, a1, a2, mu, and gamma)
    '''

    bg = bg_pdf(x, a[3:5])
    sig = voigt(x, [a[1], a[2], 0.0155]) 
    sig_norm = integrate.quad(lambda z: voigt(z, [a[1], a[2], 0.0155]), -1, 1)[0]

    return (1 - a[0])*bg + a[0]*sig/sig_norm

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
    x = x.reshape(int(x.size/samples_per_toy), samples_per_toy)

    # if the exact number of toy datasets are not generated either trim or
    # produce more.
    ndata = x.shape[0]
    if ndata < ntoys:
        xplus = generator(pdf, samples_per_toy, (ntoys-ndata), bounds)
        x = np.concatenate((x, xplus))
    elif ndata > ntoys:
        x = x[:int(ntoys),]

    return x


######################
### plotting tools ###
######################

def plot_pvalue_scan_1D(qscan, x, path):
    '''
    Helper function for plotting 1D pvalue scans.
    '''
    
    p_val = np.array(0.5*chi2.sf(qscan, 1) + 0.25*chi2.sf(qscan, 2))
    plt.plot(x, p_val)

    # Draw significance lines
    ones = np.ones(x.size)
    plt.plot(x, norm.sf(1)*ones, 'r--')
    for i in xrange(2, 7):
        if norm.sf(i) < p_val.min: 
            break
        plt.plot(x, norm.sf(i)*ones, 'r--')
        plt.text(60, norm.sf(i)*1.25, r'${0} \sigma$'.format(i), color='red')
    
    plt.yscale('log')
    plt.title(r'')
    plt.ylim([np.min(0.5*np.min(p_val), 0.5*norm.sf(3)), 1.])
    plt.xlim([x[0], x[-1]])
    plt.xlabel(r'$m_{\mu\mu}$ [GeV]')
    plt.ylabel(r'$p_{local}$')
    plt.savefig(path)
    plt.close()

def plot_pvalue_scan_2D(qscan, x, y, path, nchannels=1):
    '''
    Helper function for plotting 1D pvalue scans.
    '''
    if nchannels == 1:
        p_val = np.array(0.5*chi2.sf(qscan, 1))
    elif nchannels == 2:
        p_val = np.array(0.5*chi2.sf(qscan, 1) + 0.25*chi2.sf(qscan, 2))

    p_val = p_val.reshape(x.size, y.size).transpose()
    z_val = -norm.ppf(p_val)

    ### draw the p values as a colormesh
    plt.pcolormesh(x, y, p_val[:-1, :-1], cmap='viridis_r', norm=LogNorm(vmin=0.25*p_val.min(), vmax=p_val.max()), linewidth=0, rasterized=True)
    cbar = plt.colorbar()
    cbar.set_label(r'$p_{local}$')

    ### draw the z scores as contours 
    vmap = plt.get_cmap('gray_r')
    vcol = [vmap(0.95) if i >= 2 else vmap(0.05) for i in range(5)]
    cs = plt.contour(x, y, z_val, [1, 2, 3, 4, 5], colors=vcol)
    plt.clabel(cs, inline=1, fontsize=10, fmt='%d')

    plt.xlabel(r'$m_{\mu\mu}$ [GeV]')
    plt.ylabel(r'$\sigma$ [GeV]')
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.savefig(path)
    plt.close()


def fit_plot(data, xlim, sig_model, bg_model, suffix, path='plots'):
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
        ax.set_ylim([0., 2*np.max(h[0])])
        ax.set_xlabel(r'$m_{\mu\mu}$ [GeV]')
        ax.set_ylabel('entries / 2 GeV')
    elif suffix[:4] == '1b1c':
        ax.set_title(r'$\mu\mu$ + 1 b jet + 1 central jet + MET < 40 + $\Delta\phi (\mu\mu ,bj)$')
        ax.set_ylim([0., 1.5*np.max(h[0])])
        ax.set_xlabel(r'$m_{\mu\mu}$ [GeV]')
        ax.set_ylabel('entries / 2 GeV')
    elif suffix[:8] == 'combined':
        ax.set_title(r'$\mu\mu$ + 1 b jet + 1 jet')
        ax.set_ylim([0., 1.5*np.max(h[0])])
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

