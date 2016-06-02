#!/usr/bin/env python

import pickle
import os
import numpy as np
import lookee as lee

from scipy.stats import chi2, norm

if __name__ == '__main__':

    path    = 'data/batch_combination_1D/'
    files   = [open(path + f, 'r') for f in os.listdir(path) if os.path.isfile(path + f)]
    print 'Getting data from {0}...'.format(path)

    qmaxscan    = []
    phiscan     = []
    paramscan   = []
    u_0         = np.linspace(0., 20., 1000)
    for f in files:
        #u_0 = pickle.load(f)
        qmaxscan.append(pickle.load(f))
        phiscan.append(pickle.load(f))
        paramscan.append(pickle.load(f))
        f.close()

    qmaxscan    = np.array([q for scan in qmaxscan for q in scan])
    phiscan     = np.concatenate(phiscan, axis=0)

    ### Calculate LEE correction ###
    qmax    = 26.89
    ndim    = 1
    k1, nvals1, p_global    = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=1)
    k2, nvals2, p_global    = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=2)
    k, nvals, p_global      = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim)
    lee.validation_plots(u_0, phiscan, qmaxscan, [nvals1, nvals2, nvals], [k1, k2, k], 'combination_1D')

    print 'k = {0:.2f}'.format(k)
    for i,n in enumerate(nvals):
        print 'N{0} = {1:.2f}'.format(i, n)
    print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(norm.cdf(-np.sqrt(qmax)), np.sqrt(qmax))
    print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))


