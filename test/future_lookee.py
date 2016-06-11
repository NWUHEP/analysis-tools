#!/usr/bin/env python

import sys, pickle
from itertools import product
from timeit import default_timer as timer

import numpy as np
from scipy.stats import norm

from nllfitter import Parameters, Model, NLLFitter
import nllfitter.fit_tools as ft
from nllfitter.emcee import generator

class ScanParameters:
    '''
    Class for defining parameters for scanning over fit parameters.
    Parameters
    ==========
    names: name of parameters to scan over
    bounds: values to scan between (should be an array with 2 values)
    nscans: number of scan points to consider
    '''
    def __init__(self, names, bounds, nscans, fixed=False):
        self.names  = names
        self.bounds = bounds
        self.nscans = nscans
        self.init_scan_params()

    def init_scan_params(self):
        scans = []
        div   = []
        for n, b, ns in zip(self.names, self.bounds, self.nscans):
            scans.append(np.linspace(b[0], b[1], ns))
            div.append(np.abs(b[1] - b[0])/ns)
        self.div   = div
        self.scans = scans

    def get_scan_vals(self, ):
        '''
        Return an array of tuples to be scanned over.
        '''
        return list(product(*self.scans)), self.div

if __name__ == '__main__':

    start = timer()

    ### Get command line arguments
    if len(sys.argv) > 2:
        channel = str(sys.argv[1])
        nsims   = int(sys.argv[2])
        ndim    = int(sys.argv[3])
    else:
        channel = '1b1f'
        nsims   = 100
        ndim    = 2

    #####################
    ### Configuration ###
    #####################

    minalgo    = 'SLSQP'
    xlimits    = (12., 70.)
    make_plots = False

    ########################
    ### Define fit model ###
    ########################

    data, n_total = ft.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None)
                      )

    bg_model  = Model(ft.bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model)
    bg_result = bg_fitter.fit(data)

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    sig_params.add_many(
                        ('A'     , 0.01 , True , 0.01 , 1.   , None),
                        ('mu'    , -0.5 , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.01 , True , 0.02 , 1.   , None)
                       )
    sig_params += bg_params.copy()
    sig_model  = Model(ft.sig_pdf, sig_params)
    sig_fitter = NLLFitter(sig_model)
    sig_result = sig_fitter.fit(data)

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))

    ### Generate toy MC ###
    sims = generator(bg_model.pdf, n_total, ntoys=nsims)

    #######################################################
    ### Scan over search dimensions/nuisance parameters ###
    #######################################################

    ### Define scan values here ### 
    scan_params = ScanParameters(names = ['mu', 'sigma'],
                                 bounds = [(-0.8, 0.8), (-0.1, 0.02)],
                                 nscans = [25, 25]
                                )

    '''
    paramscan = []
    phiscan = []
    qmaxscan = []
    for i, sim in enumerate(sims):
        if i%10 == 0: print 'Carrying out scan {0}...'.format(i+1)

        params, phis, qmax = q_scan(bg_fitter, sig_fitter, scan_params, sim) 

        if make_plots and i < 9:
            of.fit_plot(sim, xlimits, sig_model, bg_model,
                        '{0}_{1}'.format(channel,i+1), path='figures/scan_fits')

    phiscan     = np.array(phiscan)
    paramscan   = np.array(paramscan)
    qmaxscan    = np.array(qmaxscan)

    ################################
    ### Calculate LEE correction ###
    ################################

    k1, nvals1, p_global = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=1)
    k2, nvals2, p_global = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=2)
    k, nvals, p_global   = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim)
    lee.validation_plots(u_0, phiscan, qmaxscan, 
                         #[nvals1, nvals2, nvals], [k1, k2, k], 
                         [nvals], [k], 
                         'combined_{1}D'.format(channel, ndim))

    print 'k = {0:.2f}'.format(k)
    for i,n in enumerate(nvals):
        print 'N{0} = {1:.2f}'.format(i, n)
    print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(norm.cdf(-np.sqrt(qmax)), np.sqrt(qmax))
    print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))

    # Save scan data
    outfile = open('data/lee_scan_{0}_{1}.pkl'.format('combined', nsims), 'w')
    pickle.dump(u_0, outfile)
    pickle.dump(qmaxscan, outfile)
    pickle.dump(phiscan, outfile)
    pickle.dump(paramscan, outfile)
    outfile.close()
    '''

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
