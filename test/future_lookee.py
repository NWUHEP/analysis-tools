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

#class ParameterScanner:
#    '''
#    Class for carrying out scans over parameters.  Each scan point will fit the
#    provided signal and background models to data.
#
#    Parameters
#    ==========
#    '''
#    def __init__(self, scan_params, 

def q_scan(bg_fitter, sig_fitter, scan_params, data): 

    qscan       = []
    params_best = []
    qmax = 0

    bg_fitter.set_data(data)
    init_params = bg_fitter.model.get_params().values()
    bg_result = bg_fitter.fit(init_params, calculate_corr=False)

    scan_vals, scan_div = scan_params.get_scan_vals()
    for scan in scan_vals:
        #if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
        #    qscan.append(0.)
        #    continue

        ### Set scan values and fit signal model ###
        sig_model.bounds[1] = (scan[0], scan[0]+scan_div[0])
        sig_model.bounds[2] = (scan[1], scan[1]+scan_div[1])

        param_init = sig_model.get_params().values()
        sig_result = sig_fitter.fit(param_init, calculate_corr=False)
        sig_nll    = combined_sig_model.nll(sim, sig_result.x)

        qtest = ff.calculate_likelihood_ratio(combined_bg_model, combined_sig_model, sim) 
        qscan.append(qtest)
        if qtest > qmaxscan[-1]: 
            params_best = sig_result.x
            qmaxscan[-1] = qtest

    return params_best, phiscan, qmaxscan


    '''
    # Initialization
    bg_model = bg_fitter.model
    sig_model = sig_fitter.model

    scan_vals, scan_div = scan_params.get_scan_vals()
        if i%10 == 0: 
            print 'Carrying out scan {0}...'.format(i+1)
            
        ### Use simulated data for fits (of course) ###
        bg_fitter.set_data(sim)
        sig_fitter.set_data(sim)

        ### Fit background model ###
        bg_result = bg_fitter.fit([0.5, 0.05], calculate_corr=False)

        qscan       = []
        params_best = []
        qmaxscan.append(0)
        

        ### Doing calculations
        qscan = np.array(qscan).reshape(scan_params.nscans)
        phiscan.append([lee.calculate_euler_characteristic((qscan > u) + 0.) for u in u_0])
        '''


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

    data, n_total = of.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)

    bg_model = ff.Model(bg_pdf, ['a1', 'a2'])
    bg_model.set_bounds([(-1., 1.), (-1., 1.)])
    bg_fitter = ff.NLLFitter(bg_model, data, verbose=False)
    bg_result = bg_fitter.fit([0.5, 0.05])

    sig_model = ff.Model(sig_pdf, ['A', 'mu', 'sigma', 'a1', 'a2'])
    sig_model.set_bounds([(0., .5), 
                          (-0.8, -0.2), (0.04, 0.1),
                          (-1., 1.), (-1., 1.)])
    sig_fitter = ff.NLLFitter(sig_model, data, verbose=False)
    sig_result = sig_fitter.fit((0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1]))

    pdf = bg_model.pdf
    sims = mc.mc_generator(pdf, n_total, ntoys=nsims)

    #########################
    ### Define fit models ###
    #########################

    #channels    = ['1b1f', '1b1c']
    #datas      = OrderedDict()
    #sims       = OrderedDict()
    #bg_models  = OrderedDict() 
    #sig_models = OrderedDict() 
    #for channel in channels:

    #    data, n_total = of.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    #    datas[channel] = data

    #    bg_model = ff.Model(bg_pdf, ['a1', 'a2'])
    #    bg_model.set_bounds([(-1., 1.), (-1., 1.)])
    #    bg_models[channel] = bg_model
    #    bg_fitter = ff.NLLFitter(bg_model, data, verbose=False)
    #    bg_result = bg_fitter.fit([0.5, 0.05])

    #    sig_model = ff.Model(sig_pdf, ['A', 'mu', 'sigma', 'a1', 'a2'])
    #    sig_model.set_bounds([(0., .5), 
    #                          (-0.8, -0.2), (0.04, 0.1),
    #                          (-1., 1.), (-1., 1.)])
    #    sig_models[channel] = sig_model
    #    sig_fitter = ff.NLLFitter(sig_model, data, verbose=False)
    #    _ = sig_fitter.fit((0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1]))

    #    pdf = bg_model.pdf
    #    sims[channel] = mc.mc_generator(pdf, n_total, ntoys=nsims)

    #bg_models['1b1f'].parnames = ['a1', 'a2']
    #bg_models['1b1c'].parnames = ['b1', 'b2']
    #combined_bg_model          = ff.CombinedModel([bg_models[ch] for ch in channels])
    #combination_bg_fitter      = ff.NLLFitter(combined_bg_model, [datas[ch] for ch in channels], verbose = False)
    #bg_result = combination_bg_fitter.fit([0.5, 0.05, 0.5, 0.05])

    #sig_models['1b1f'].parnames = ['A1', 'mu', 'sigma', 'a1', 'a2']
    #sig_models['1b1c'].parnames = ['A2', 'mu', 'sigma', 'b1', 'b2']
    #combined_sig_model          = ff.CombinedModel([sig_models[ch] for ch in channels])
    #combination_sig_fitter      = ff.NLLFitter(combined_sig_model, [datas[ch] for ch in channels], verbose = False)
    #param_init = combined_sig_model.get_params().values()
    #sig_result = combination_sig_fitter.fit(param_init)

    #qmax = ff.calculate_likelihood_ratio(combined_bg_model, combined_sig_model, datas.values()) 

    #######################################################
    ### Scan over search dimensions/nuisance parameters ###
    #######################################################

    ### Define scan values here ### 
    scan_params = ScanParameters(names = ['mu', 'sigma'],
                                 bounds = [(-0.85, 0.85), (0.04, 1.)],
                                 nscans = [25, 20]
                                )
    #scan_params = ScanParameters(names = ['mu1', 'mu2'],
    #                             bounds = [(-0.8, 0.8), (-0.8, 0.8)],
    #                             nscans = [25, 25]
    #                            )

    paramscan = []
    phiscan = []
    qmaxscan = []
    for i, sim in enumerate(sims):
        if i%10 == 0: print 'Carrying out scan {0}...'.format(i+1)

        params, phis, qmax = q_scan(bg_fitter, sig_fitter, scan_params, sims) 

        if make_plots and i < 9:
            of.fit_plot(of.scale_data(sim[0], invert=True), xlimits,
                        sig_pdf, params_best,    
                        bg_pdf, bg_model.params,
                        '{0}_{1}'.format('1b1f',i+1), path='figures/scan_fits')
            of.fit_plot(of.scale_data(sim[1], invert=True), xlimits,
                        sig_pdf, params_best,    
                        bg_pdf, bg_model.params,
                        '{0}_{1}'.format('1b1c',i+1), path='figures/scan_fits')

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

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
