#!/usr/bin/env python

import sys
import pickle
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.stats import chi2, norm
from scipy.ndimage.morphology import *
from scipy.ndimage import *
from scipy.special import gamma
from scipy.misc import comb, factorial

def calculate_euler_characteristic(a):
   '''Calculate the Euler characteristic for level set a'''
   face_filter=np.zeros((2,2))+1
   right_edge_filter = np.array([[1,1]])
   bottom_edge_filter = right_edge_filter.T
   
   n_faces = np.sum(convolve(a,face_filter,mode='constant')>3)
   n_edges = np.sum(convolve(a,right_edge_filter,mode='constant')>1)
   n_edges += np.sum(convolve(a,bottom_edge_filter,mode='constant')>1)
   n_vertices = np.sum(a>0)
   
   EC = n_vertices-n_edges+n_faces
   #print '%d-%d+%d=%d' %(n_vertices,n_edges,n_faces,EulerCharacteristic) 
   
   return EC

def rho_g(u, j=1, k=1):
    '''
    From theorem 15.10.1 from Random Fields and Geometry (Adler and Taylor)

    Parameters
    ----------
    j: number of nuisance parameters (search dimensions)
    k: d.o.f. of chi2 random field
    u: threshold for excursions in the field
    '''

    coeff_num       = u**((k - j)/2.) * np.exp(-u/2.) 
    coeff_den       = (2.*np.pi)**(j/2.) * gamma(k/2.) * 2**((k-2.)/2.)
    indicate        = lambda m,l: float(k >= j - m - 2.*l)
    sum_fraction    = lambda m,l: ((-1.)**(j - 1. + m + l) * factorial(j - 1)) / (factorial(m)*factorial(l)*2.**l)
    m_terms         = lambda l: np.array([indicate(m,l) * comb(k-l, j-1.-m-2.*l) * sum_fraction(m,l) * u**(m+l) 
                                        for m in np.arange(0, 1 + int(j-1.-2.*l))])
    m_sum           = lambda l: np.sum(m_terms(l), axis=0)
    l_sum           = np.sum([m_sum(l) for l in np.arange(0., 1 + np.floor((j-1)/2))], axis=0) 

    return (coeff_num/coeff_den)*l_sum

def exp_phi_u(u, n_j, k=1):
    '''
    1 or 2 dimensional expressions for chi2 random field EC expectation
    
    Parameters
    ----------
    u: array of scan thresholds
    n_j: array of coefficients
    k: nDOF of chi2 field
    '''
    return chi2.sf(u,k) + np.sum([n*rho_g(u, j+1, k) for j,n in enumerate(n_j)], axis=0)

def lee_objective(a, Y, dY, X, k0):
    '''
    Defines the objective function for regressing the <EC> of our chi2 field.
    The minimization should be done on the quadratic cost weighted by the
    inverse of the variance on the measurement.  There is an additional term
    which will enforce preference for the fit result being greater than the
    data point.  The reasoning is that we would like to have an upper bound on
    our tails (that is, we are being conservative here).

    Parameters
    ----------
    a: list of parameters
    Y: target data
    dY: variance on the data
    X: independent variable values corresponding to values of Y
    '''

    ephi    = exp_phi_u(X, a[1:], k = a[0])
    qcost   = np.sum((Y - ephi)**2/dY)
    ubound  = np.sum(ephi < Y)/Y.size 
    L1_reg  = np.sum(np.abs(a)) 
    L2_reg  = np.sum(a**2)

    return qcost + (a[0] - k0)**2 + 0.5*ubound

def lee_nD(max_local_sig, u, phiscan, j=1, k=1, do_fit=True):
    '''
    Carries GV style look elsewhere corrections with a twist.  Allows for an
    arbitrary number of search dimensions/nuisance parameters and allows the
    number of degrees of freedom of the chi2 random field to be a parameter of
    the model.  Cool shit.

    Parameters
    ----------
    max_local_sig: observed local significance (assumes sqrt(-2*nllr))
    u: array of scan thresholds
    phiscan: scan of EC for values in u
    j = numbers of search dimensions to calculate
    k = assumed numbers of degrees of freedom of chi2 field. If not specified
        it will be a floating parmeter in the LEE estimation (recommended)
    '''
    exp_phi = phiscan.mean(axis=0)
    var_phi = phiscan.var(axis=0)

    ### Remove points where exp_phi > 0 ###
    phimask = (exp_phi > 0.)
    exp_phi = exp_phi[phimask]
    var_phi = var_phi[phimask]
    u       = u[phimask]

    ### if variance on phi is 0, use the poisson error on dY ###
    var_phi[var_phi==0] = 1./np.sqrt(phiscan.shape[0])
    
    if do_fit:
        print 'd.o.f. not specified => fit the EC with scan free parameters N_j and k...'
        bnds   = [(1, None)] + j*[(0., None)]
        p_init = [1.] + j*[1.,]
        result = minimize(lee_objective,
                          p_init,
                          method = 'Nelder-Mead',
                          args   = (exp_phi, var_phi, u, k),
                          #bounds = bnds
                          )
        k = result.x[0]
        n = result.x[1:]
    else:
        print 'd.o.f. specified => fit the EC scan with free parameters N_j and k={0}...'.format(k)
        mask  = np.arange(1, 1 + j)*100
        xvals = u[mask]
        ephis = exp_phi[mask]
        eq    = lambda n: [ephi - exp_phi_u(x, n, k=k) for ephi,x in zip(ephis, xvals)]
        n     = fsolve(eq, j*(1,))

    p_global = exp_phi_u(max_local_sig**2, n, k)

    return k, n, p_global

def validation_plots(u, phiscan, qmax, Nvals, kvals, channel):
    '''Check that the GV tails look okay'''

    ### Get the mean and variance from the phi scan ###
    phiscan = np.array(phiscan)
    exp_phi = np.mean(phiscan, axis=0)
    var_phi = np.var(phiscan, axis=0)
    qmax    = np.array(qmax)


    ### Construct the survival function spectrum from maximum q of each scan ###
    hval, hbins, _ = plt.hist(qmax, bins=30, range=(0.,30.), cumulative=True)
    hval = hval.max() - hval
    herr = np.sqrt(hval)
    pval = hval/hval.max()
    perr = pval*(herr/hval)
    pval = np.concatenate(([1], pval))
    perr = np.concatenate(([0], perr))
    plt.close()

    ### Remove points where values are 0 ###
    pmask = pval > 0.
    emask = exp_phi > 0.

    ### Make the plots ###
    fig, ax = plt.subplots()
    ax.plot(hbins[pmask], pval[pmask], 'm-', linewidth=2.)
    ax.plot(u[emask], exp_phi[emask], 'k-', linewidth=2.)
    ax.fill_between(hbins, pval-perr, pval+perr, color='m', alpha=0.25, interpolate=True)
    for N ,k in zip(Nvals, kvals):
        ax.plot(u, exp_phi_u(u, N, k), '--', linewidth=2.)


    ### Stylize ###
    ax.legend([r'$1 -  \mathrm{CDF}(q(\theta))$', r'$\overline{\phi}_{\mathrm{sim.}}$'] 
            + [r'$\overline{{\phi}}_{{ \mathrm{{th.}} }}; k={0}$'.format(k) if type(k) == int 
                else r'$\overline{{\phi}}_{{ \mathrm{{th.}} }}; k={0:.2f}$'.format(k) for k in kvals])

    ax.set_yscale('log')
    ax.set_ylim(1e-4, 5*np.max(phiscan))
    ax.set_ylabel(r'$\mathbb{\mathrm{P}}[q_{\mathrm{max}} > u]$')
    ax.set_xlabel(r'$u$')
    fig.savefig('plots/GV_validate_{0}.png'.format(channel))
    fig.savefig('plots/GV_validate_{0}.pdf'.format(channel))
    plt.close()

def excursion_plot_1d(x, qscan, u1, suffix, path):
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$M_{\mu\mu}$ [GeV]')
    ax.set_ylabel('q')
    ax.set_xlim([12., 70.])
    ax.set_ylim([0., 25.])
    ax.plot(x, qscan, 'r-', linewidth=2.)
    ax.plot([12., 70.], [u1, u1], 'k-', linewidth=2.)

    fig.savefig('{0}/excursion_1D_{1}.pdf'.format(path, suffix))
    fig.savefig('{0}/excursion_1D_{1}.png'.format(path, suffix))
    plt.close()


