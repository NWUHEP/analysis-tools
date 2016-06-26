#!/usr/bin/env python

from __future__ import division

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
   '''
   Calculate the Euler characteristic for level set a.
   Taken from https://github.com/cranmer/look-elsewhere-2d/blob/master/lee2d.py#L87
   '''
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
    u: threshold for excursions in the field. Can be 
    j: number of nuisance parameters (search dimensions)
    k: d.o.f. of chi2 random field
    '''

    coeff_num       = np.power(u, (k - j)/2.) * np.exp(-u/2.) 
    coeff_den       = np.power(2.*np.pi, j/2.) * gamma(k/2.) * np.power(2, (k - 2.)/2.)

    sum_term = 0.
    for l in xrange(1 + np.max(0, int(((j-1)/2)))):
        for m in xrange(j - 2*l):
            choose = comb(k-1, j-1-m-2*l)
            indicate = float(k >= j-m-2*l)
            sum_num = ((-1)**(j-1+m+l) * factorial(j-1)) 
            sum_den = (factorial(m)*factorial(l)*(2**l))

            sum_term += indicate*choose*(sum_num/sum_den)*np.power(u, m+l)

    return (coeff_num/coeff_den)*sum_term

def exp_phi_u(u, n_j, k=1):
    '''
    Returns the Gross-Vittels expansion of the expectation of the E.C. of a
    chi2 random field with k d.o.f. and give expansion coefficients n_j 

    Parameters
    ----------
    u   : array of scan thresholds
    n_j : array of coefficients
    k   : nDOF of chi2 field
    '''
    
    return chi2.sf(u,k) + np.sum([n*rho_g(u, j+1, k) for j,n in enumerate(n_j)], axis=0)

def lee_objective(a, Y, dY, X, k0, fix_dof=False):
    '''
    Defines the objective function for regressing the <EC> of our chi2 field.
    The minimization should be done on the quadratic cost weighted by the
    inverse of the variance on the measurement.  There is an additional term
    which will enforce preference for the fit result being greater than the
    data point.  The reasoning is that we would like to have an upper bound on
    our tails (that is, we are being conservative here).

    Parameters
    ----------
    a  : list of parameters (a[0] is d.o.f., a[1:] are the expansion coefficients)
    Y  : target data
    dY : variance on the data
    X  : independent variable values corresponding to values of Y
    '''

    if fix_dof:
        ephi = exp_phi_u(X, a[1:], k = k0)
    else:
        ephi = exp_phi_u(X, a[1:], k = a[0])

    quadratic_cost = np.sum((Y - ephi)**2/dY)

    ### upper bound
    ubound = np.sum(ephi < Y)/Y.size 

    ### regularization of expansion coefficients
    L1_reg  = np.sum(np.abs(a[1:])) 
    L2_reg  = np.sum(a[1:]**2)

    ### require expansion coefficients to be positive
    pCoeff1 = 0.
    if np.any(a[1:] < 0):
        pCoeff1 = np.exp(-10*np.sum(a[a<0.]))

    ### require k >= 1
    pCoeff2 = 0.
    if a[0] < 1 and not fix_dof:
        pCoeff2 = np.inf

    objective = quadratic_cost # essential
    #objective += ubound # ad-hoc
    #objective += L1_reg  
    #objective += L2_reg  
    #objective += pCoeff1
    #objective += pCoeff2

    return objective

def lee_nD(max_local_sig, u, phiscan, j=1, k=1, do_fit=True, fix_dof=False):
    '''
    Carries GV style look elsewhere corrections with a twist.  Allows for an
    arbitrary number of search dimensions/nuisance parameters and allows the
    number of degrees of freedom of the chi2 random field to be a parameter of
    the model.  Cool shit.

    Parameters
    ----------
    max_local_sig : observed local significance (assumes sqrt(-2*q))
    u             : array of scan thresholds
    phiscan       : scan of EC 
    j             : numbers of search dimensions to calculate
    k             : assumed numbers of degrees of freedom of chi2 field. If not
                    specified it will be a floating parmeter in the LEE estimation.
    do_fit        : by default the d.o.f. and the amplitude of the LEE
                    expansion terms will be allowed to vary; if False the d.o.f. will be fixed
                    to the input value and the expansion coefficients will be determined by
                    picking a few point from the data.
    '''
    exp_phi = phiscan.mean(axis=0)
    var_phi = phiscan.var(axis=0)

    ### Remove points where exp_phi < 0 ###
    phimask = (exp_phi > 0.)
    exp_phi = exp_phi[phimask]
    var_phi = var_phi[phimask]
    u       = u[phimask]

    ### if variance on phi is 0, use the poisson error on dY ###
    var_phi[var_phi==0] = 1./np.sqrt(phiscan.shape[0])
    
    if do_fit:
        print 'fit the EC with scan free parameters N_j and k...'
        if fix_dof:
            k_bnds = [(k, k)]
        else:
            k_bnds = [(1., np.inf)]

        bnds   = k_bnds + j*[(0., np.inf)]
        p_init = [e] + j*[1.,]
        result = minimize(lee_objective,
                          p_init,
                          method = 'SLSQP',
                          args   = (exp_phi, var_phi, u, k, fix_dof),
                          bounds = bnds
                          )
        k = result.x[0] if result.x[0] >= 1. else 1.
        n = result.x[1:]
    else:
        print 'fit the EC scan with free parameters N_j and k={0}...'.format(k)
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


