#!/usr/bin/env python

import pickle
import os,sys

import numpy as np
from scipy.stats import chi2, norm
from scipy.optimize import minimize

import nllfitter.lookee as lee

def func_GV(a, u, N_1b1f, N_1b1c):
    '''
    Sloppy function to fit the k = 2 part of the simultaneous LEE
    '''

    prob_1b1f = 0.25 * lee.exp_phi_u(u, N_1b1f, k=1)
    prob_1b1c = 0.25 * lee.exp_phi_u(u, N_1b1c, k=1)
    prob_comb = 0.25 * lee.exp_phi_u(u, a, k=2)

    return prob_1b1f + prob_1b1c + prob_comb


if __name__ == '__main__':

    if len(sys.argv) > 1:
        ndim    = int(sys.argv[1])
    else:
        ndim    = 1
                   
    path        = 'data/batch_combination_{0}D/'.format(ndim)
    filenames   = [path + f for f in os.listdir(path) if os.path.isfile(path + f)]
    print 'Getting data from {0}...'.format(path)

    qmaxscan    = []
    phiscan     = []
    paramscan   = []
    u_0         = np.linspace(0., 20., 1000)
    for name in filenames:
        f = open(name, 'r')
        u_0 = pickle.load(f)
        qmaxscan.append(pickle.load(f))
        phiscan.append(pickle.load(f))
        paramscan.append(pickle.load(f))
        f.close()

    qmaxscan = np.array([q for scan in qmaxscan for q in scan])
    phiscan = np.concatenate(phiscan, axis=0)
    paramscan = np.concatenate(paramscan, axis=0)

    qmax = 27.57
    k0 = 2 
    p_local = 0.5*chi2.sf(qmax, 1) + 0.25*chi2.sf(qmax, 2) # according to Chernoff 
    z_local = -norm.ppf(p_local)

    ### Doing the LEE ###
    exp_phi = phiscan.mean(axis=0)
    var_phi = phiscan.var(axis=0)

    ### Remove points where exp_phi < 0 ###
    phimask = (exp_phi > 0.)
    exp_phi = exp_phi[phimask]
    var_phi = var_phi[phimask]
    u_0     = u_0[phimask]

    ### if variance on phi is 0, use the poisson error on dY ###
    var_phi[var_phi==0] = 1./np.sqrt(phiscan.shape[0])
    ###

    if ndim == 1:
        N_1b1f = [11.16, 0.]
        N_1b1c = [11.52, 0.]
    elif ndim == 2:
        N_1b1f = [11.23, 16.61]
        N_1b1c = [10.83, 18.69]

    objective = lambda a: np.sum((exp_phi - func_GV(a, u_0, N_1b1f, N_1b1c))**2/var_phi) + np.abs(np.sum(a)) 
    result = minimize(objective,
                      [1., 1.],
                      method = 'SLSQP',
                      bounds = [(0., np.inf), (0., np.inf)]
                     )

    #lee.validation_plots(u_0, phiscan, qmaxscan, [nvals], [k], '{0}_{1}D'.format(channel, ndim))


