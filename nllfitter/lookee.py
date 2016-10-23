#!/usr/bin/env python

from __future__ import division

import sys
import pickle
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
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


def lee_objective(a, Y, dY, X, ndim, kvals, scales):
    '''
    Defines the objective function for regressing the <EC> of the chi-squared field.
    The minimization should be done on the quadratic cost weighted by the
    inverse of the variance on the measurement.  

    Parameters
    ----------
    a      : parameters (list of GV expansion coefficients)
    Y      : expectation of the E.C. from data
    dY     : variance on the E.C. from data
    X      : excursion level of the E.C., u.
    ndim   : number of dimensions scanned over; also number of parameters per k
    kvals  : list of d.o.f. of chi-squared field
    scales : contribution from each chi-squared component for fixed excursions
    '''

    ephi = np.zeros(X.size)
    for i, (s, k) in enumerate(zip(scales, kvals)):
        ephi += s*exp_phi_u(X, a[i*ndim:(i+1)*ndim], k) 

    objective = np.sum((Y - ephi)**2/dY)
    return objective

def get_GV_coefficients(u, phiscan, p_init, p_bnds, kvals, scales):
    '''
    Carries GV style look elsewhere corrections with a twist.  Allows for an
    the model.  Cool shit.
    number of degrees of freedom of the chi2 random field to be a parameter of
    the model.

    Parameters
    ----------
    u       : array of scan thresholds
    phiscan : scans of the E.C. of multiple likelihood scans 
    j       : numbers of search dimensions in the likelihood ratio scan
    kvals   : assumed numbers of degrees of freedom of chi2 field. If not
              specified it will be a floating parmeter in the LEE estimation.
    scales  : contribution from each d.o.f. component for the 0th order E.C.
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
    
    ndim = int(len(p_init)/len(kvals))
    result = minimize(lee_objective,
                      p_init,
                      method = 'SLSQP',
                      args   = (exp_phi, var_phi, u, ndim, kvals, scales),
                      bounds = p_bnds
                      )
    return result.x
    #return np.reshape(result.x, (len(kvals), ndim))

def get_p_global(qmax, kvals, nvals, scales): 
    '''
    Calculate the global p value and z scores.

    Parameters:
    ===========
    qmax: observed excursion of the likelihood ratio
    k: d.o.f. of the chi-squared field
    nvals: array of GV coefficients
    '''
    
    p_global = 0
    for k, n, scale in zip(kvals, nvals, scales):
        p_global += scale*exp_phi_u(qmax, n, k)
    return p_global

def gv_validation_plot(u, phiscan, qmax, nvals, kvals, scales, channel):
    '''
    Overlays expectation of the E.C., the SF of the likelihood scan, and the GV prediction. 

    Parameters
    ==========
    u       : excursion levels of the likelihood ration
    phiscan : E.C. for the q scans
    qmax    : maximum of q for each of the scans
    nvals   : coefficients of the GV prediction
    k       : d.o.f. of the chi-squared field
    scale   : contribution from different d.o.f. components of the chi-squared field
    channel : name of channel under consideration
    '''

    ### Construct the survival function spectrum from maximum q of each scan ###
    hval, hbins = np.histogram(qmax, bins=30, range=(0.,30.))
    hval = np.cumsum(hval)
    hval = hval.max() - hval
    herr = np.sqrt(hval)
    pval = hval/hval.max()
    perr = pval*(herr/hval)
    pval = np.concatenate(([1], pval))
    perr = np.concatenate(([0], perr))
    plt.close()

    ### Get the mean and variance from the phi scan ###
    exp_phi = np.mean(phiscan, axis=0)
    var_phi = np.var(phiscan, axis=0)
    exp_phi_total = np.zeros(u.size)
    for k, n, scale in zip(kvals, nvals, scales):
        exp_phi_total += scale*exp_phi_u(u, n, k)

    ### Remove points where values are 0 ###
    pmask = pval > 0.
    emask = exp_phi > 0.

    ### Plot SF(u) from data ###
    fig, ax = plt.subplots()
    ax.plot(hbins[pmask], pval[pmask], 'm-', linewidth=2)
    ax.fill_between(hbins, pval-perr, pval+perr, color='m', alpha=0.25, interpolate=True)

    ### Plot the E[phi(u)] from the data and the predicted excursions ###
    ax.plot(u[emask], exp_phi[emask], 'k-', linewidth=2.5)
    ax.plot(u, exp_phi_total, 'b--', linewidth=2.5)

    ### Stylize ###
    legend_text = [r'$\sf SF(q(\theta))$', 
                   r'$\sf \overline{\phi}_{sim.}$', 
                   r'$\sf \overline{\phi}_{th.}$'
                  ]

    ax.legend(legend_text)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 5*np.max(phiscan))
    ax.set_ylabel(r'$\sf \mathbb{P}[q_{max} > u]$')
    ax.set_xlim(0, 30)
    ax.set_xlabel(r'u')
    #ax.set_title(channel.replace('_', ' '))
    ax.grid()

    if channel == None:
        plt.show()
    else:
        fig.savefig('plots/fits/GV_validate_{0}.png'.format(channel))
        fig.savefig('plots/fits/GV_validate_{0}.pdf'.format(channel))
        plt.close()


