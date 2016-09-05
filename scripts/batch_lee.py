#!/usr/bin/env python

import pickle
import os,sys

import numpy as np
from scipy.stats import chi2, norm
from scipy.optimize import minimize

import nllfitter.lookee as lee

if __name__ == '__main__':

    if len(sys.argv) > 2:
        channel = str(sys.argv[1])
        ndim    = int(sys.argv[2])
    else:
        channel = '1b1f'
        ndim    = 1
                   
    path        = 'data/batch_{0}_{1}D/'.format(channel, ndim)
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

    if ndim == 0:
        
        pdf = lambda x, a: chi2.pdf(x, a[0])
        obj = lambda a, x: -np.sum(np.log(pdf(x, a)))
        result = minimize(obj, [1.], method='SLSQP', bounds=[(1., None)], args=(qmaxscan))

    elif channel != 'combination':
        ### Calculate LEE correction ###
        if channel == '1b1f':
            qmax = 18.31
            k0 = 1 
            z_local = np.sqrt(qmax)
            p_local = norm.sf(z_local)
        elif channel == '1b1c':
            qmax = 9.8
            k0 = 1 
            z_local = np.sqrt(qmax)
            p_local = norm.sf(z_local)
        elif channel == 'combined':
            qmax = 24.43
            k0 = 1 
            z_local = np.sqrt(qmax)
            p_local = norm.sf(z_local)

        k, nvals, p_global    = lee.lee_nD(z_local, u_0, phiscan, j=ndim, k=k0, fix_dof=True)
        lee.validation_plots(u_0, phiscan, qmaxscan, [nvals], [k], '{0}_{1}D'.format(channel, ndim))
        print 'k = {0:.2f}'.format(k)
        for i,n in enumerate(nvals):
            print 'N{0} = {1:.2f}'.format(i+1, n)
        print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(p_local, z_local)
        print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))

    elif channel == 'combination': 
        qmax = 27.57
        k0 = 2 
        p_local = 0.5*chi2.sf(qmax, 1) + 0.25*chi2.sf(qmax, 2) # according to Chernoff 
        z_local = -norm.ppf(p_local)

        k, nvals, p_global    = lee.lee_nD(z_local, u_0, phiscan, j=ndim, k=k0, fix_dof=False)
        lee.validation_plots(u_0, phiscan, qmaxscan, [nvals], [k], '{0}_{1}D'.format(channel, ndim))
        print 'k = {0:.2f}'.format(k)
        for i,n in enumerate(nvals):
            print 'N{0} = {1:.2f}'.format(i+1, n)
        print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(p_local, z_local)
        print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))


