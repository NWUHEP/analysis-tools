#!/usr/bin/env python
import sys, pickle

from itertools import product
from timeit import default_timer as timer
from collections import OrderedDict

import numpy as np
from scipy.stats import norm

#import pandas as pd
#import numpy.random as rng
#import matplotlib.pyplot as plt

import nllfitter.fitter as of
import nllfitter.future_fitter as ff
import nllfitter.lookee as lee
import nllfitter.toy_MC as mc


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

def q_scan(bg_fitter, sig_fitter, 
           scan_params, sims, 
           u=np.linspace(0.01, 25., 1250.), make_plots=False):

    print 'Scanning ll ratio over {0} pseudo-datasets...'.format(nsims)

    scan_vals, scan_div = scan_params.get_scan_vals()
    paramscan, phiscan, qmaxscan  = [], [], []
    for i, sim in enumerate(sims):
        if not i%10: 
            print 'Carrying out scan {0}...'.format(i+1)
            
        ### Use simulated data for fits (of course) ###
        bg_fitter.data  = sim
        sig_fitter.data = sim

        ### Fit background model ###
        bg_result = bg_fitter.fit([0.5, 0.05], calculate_corr=False)
        bg_nll    = bg_fitter.model.nll(sim, bg_result.x)

        qscan       = []
        params_best = []
        qmaxscan.append(0)
        for scan in scan_vals:
            if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
                qscan.append(0.)
                continue

            ### Set scan values and fit signal model ###
            sig_fitter.model.bounds[1] = (scan[0], scan[0]+scan_div[0])
            sig_fitter.model.bounds[2] = (scan[1], scan[1]+scan_div[1])
            sig_result = sig_fitter.fit((0.01, scan[0], scan[1], bg_result.x[0], bg_result.x[1]),
                                         calculate_corr=False)
            sig_nll    = sig_fitter.model.nll(sim, sig_result.x)

            qtest = np.max(2*(bg_nll - sig_nll), 0)
            qscan.append(qtest)
            if qtest > qmaxscan[-1]: 
                params_best = sig_result.x
                qmaxscan[-1] = qtest

        if make_plots and i < 9:
            sim = of.scale_data(sim, invert=True)
            of.fit_plot(sim,
                        sig_pdf, params_best,    
                        bg_pdf, bg_model.params,
                        '{0}_{1}'.format(channel,i+1), path='figures/scan_fits')

        ### Doing calculations
        qscan = np.array(qscan).reshape(scan_params.nscans)
        phiscan.append([lee.calculate_euler_characteristic((qscan > u) + 0.) for u in u_0])

    qmaxscan    = np.array(qmaxscan)
    phiscan     = np.array(phiscan)
    paramscan   = np.array(paramscan)
    return u, qmaxscan, phiscan, paramscan

        
if __name__ == '__main__':

    start = timer()

    ### Get command line arguments
    if len(sys.argv) > 2:
        channel = str(sys.argv[1])
        nsims   = int(sys.argv[2])
        ndim    = int(sys.argv[3])
    else:
        channel = '1b1f'
        nsims   = 10
        ndim    = 1

    #####################
    ### Configuration ###
    #####################

    minalgo    = 'SLSQP'
    channels   = ['1b1f', '1b1c']
    xlimits    = (12., 70.)
    make_plots = True

    bg_pdf  = lambda x, a:  0.5 + a[0]*x + 0.5*a[1]*(3*x**2 - 1)
    sig_pdf = lambda x, a: (1 - a[0])*bg_pdf(x, a[3:5]) + a[0]*norm.pdf(x, a[1], a[2])

    #########################
    ### Define fit models ###
    #########################

    sims       = OrderedDict()
    bg_models  = OrderedDict() 
    sig_models = OrderedDict() 
    for channel in channels:

        data, n_total = of.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)

        bg_model = ff.Model(bg_pdf, ['a1', 'a2'])
        bg_model.set_bounds([(-1., 1.), (-1., 1.)])
        bg_models[channel] = bg_model

        sig_model = ff.Model(sig_pdf, ['A', 'mu', 'sigma', 'a1', 'a2'])
        sig_model.set_bounds([(0., .5), 
                              (-0.8, -0.2), (0., 0.5),
                              (-1., 1.), (-1., 1.)])
        sig_models[channel] = sig_model

        sims[channel] = mc.mc_generator(bg_model.pdf, n_total, nsims)

    bg_models['1b1f'].parnames  = ['a1', 'a2']
    bg_models['1b1c'].parnames  = ['b1', 'b2']
    combined_bg_model   = ff.CombinedModel([bg_models[ch] for ch in channels])
    combination_bg_fitter = ff.NLLFitter(combined_bg_model, None)

    sig_models['1b1f'].parnames = ['A1', 'mu', 'sigma', 'a1', 'a2']
    sig_models['1b1c'].parnames = ['A2', 'mu', 'sigma', 'b1', 'b2']
    combined_sig_model = ff.CombinedModel([sig_models[ch] for ch in channels])
    combination_sig_fitter = ff.NLLFitter(combined_sig_model, None)

    #######################################################
    ### Scan over search dimensions/nuisance parameters ###
    #######################################################

    ### Define scan values here ### 
    scan_params = ScanParameters(names = ['mu', 'sigma'],
                                 bounds = [(-0.9, 0.9), (0.05, 0.05)],
                                 nscans = [50, 30]
                                )
    scan_vals, scan_div = scan_params.get_scan_vals()

    paramscan   = []
    phiscan     = []
    qmaxscan    = []
    u_0         = np.linspace(0.01, 25., 1250.)
    for i, sim in enumerate(sims):
        if not i%10: 
            print 'Carrying out scan {0}...'.format(i+1)
            
        ### Use simulated data for fits (of course) ###
        combination_bg_fitter.set_data(sim)
        combination_sig_fitter.set_data(sim)

        ### Fit background model ###
        bg_result = combination_bg_fitter.fit([0.5, 0.05, 0.5, 0.05])
        #bg_result = bg_fitter.fit([0.5, 0.05], calculate_corr=False)
        bg_nll    = bg_model.nll(sim, bg_result.x)

        qscan       = []
        params_best = []
        qmaxscan.append(0)
        for scan in scan_vals:
            if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
                qscan.append(0.)
                continue

            ### Set scan values and fit signal model ###
            sig_model.bounds[1] = (scan[0], scan[0]+scan_div[0])
            sig_model.bounds[2] = (scan[1], scan[1]+scan_div[1])
            param_init = combined_sig_model.get_params().values()
            sig_result = combination_sig_fitter.fit(param_init)
            sig_result = sig_fitter.fit((0.01, scan[0], scan[1], bg_result.x[0], bg_result.x[1]), calculate_corr=False)
            sig_nll    = sig_model.nll(sim, sig_result.x)

            qtest = np.max(2*(bg_nll - sig_nll), 0)
            qscan.append(qtest)
            if qtest > qmaxscan[-1]: 
                params_best = sig_result.x
                qmaxscan[-1] = qtest

        if make_plots and i < 9:
            sim = of.scale_data(sim, invert=True)
            of.fit_plot(sim, xlimits,
                        sig_pdf, params_best,    
                        bg_pdf, bg_model.params,
                        '{0}_{1}'.format(channel,i+1), path='figures/scan_fits')

        ### Doing calculations
        qscan = np.array(qscan).reshape(scan_params.nscans)
        phiscan.append([lee.calculate_euler_characteristic((qscan > u) + 0.) for u in u_0])

    qmaxscan    = np.array(qmaxscan)
    phiscan     = np.array(phiscan)
    paramscan   = np.array(paramscan)

    ################################
    ### Calculate LEE correction ###
    ################################

    qmax = 18.3
    k1, nvals1, p_global = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=1)
    k2, nvals2, p_global = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=2)
    k, nvals, p_global   = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim)
    lee.validation_plots(u_0, phiscan, qmaxscan, 
                         [nvals1, nvals2, nvals], [k1, k2, k], 
                         '{0}_{1}D'.format(channel, ndim))

    print 'k = {0:.2f}'.format(k)
    for i,n in enumerate(nvals):
        print 'N{0} = {1:.2f}'.format(i, n)
    print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(norm.cdf(-np.sqrt(qmax)), np.sqrt(qmax))
    print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))

    # Save scan data
    outfile = open('data/lee_scan_{0}_{1}.pkl'.format(channel, nsims), 'w')
    pickle.dump(u_0, outfile)
    pickle.dump(qmaxscan, outfile)
    pickle.dump(phiscan, outfile)
    pickle.dump(paramscan, outfile)
    outfile.close()

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
