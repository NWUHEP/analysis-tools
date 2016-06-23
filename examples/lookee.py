#!/usr/bin/env python

import sys, pickle
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from nllfitter import Parameters, ScanParameters, Model, NLLFitter
import nllfitter.fit_tools as ft
import nllfitter.lookee as lee

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
        ndim    = 2

    #####################
    ### Configuration ###
    #####################

    minalgo    = 'SLSQP'
    xlimits    = (12., 70.)
    make_plots = True
    save_data  = False
    is_batch   = False

    ########################
    ### Define fit model ###
    ########################

    #data, n_total = ft.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    data_1b1f, n_1b1f = ft.get_data('data/events_pf_1b1f.csv', 'dimuon_mass', xlimits)
    data_1b1c, n_1b1c = ft.get_data('data/events_pf_1b1c.csv', 'dimuon_mass', xlimits)
    data = np.concatenate((data_1b1f, data_1b1c))
    n_total = n_1b1f + n_1b1c

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None)
                      )

    bg_model  = Model(ft.bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model, verbose=False)
    bg_result = bg_fitter.fit(data)

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    sig_params.add_many(
                        ('A'     , 0.01  , True , 0.0 , 1.   , None),
                        ('mu'    , -0.5 , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.03 , True , 0.02 , 1.   , None)
                       )
    sig_params += bg_params.copy()

    sig_model  = Model(ft.sig_pdf, sig_params)
    sig_fitter = NLLFitter(sig_model, verbose=False)
    sig_result = sig_fitter.fit(data)

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    qmax = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))

    ### Generate toy MC ###
    print 'Generating pseudodata for likelihood scans...'
    sims = ft.generator(bg_model.pdf, n_total, ntoys=nsims)

    #######################################################
    ### Scan over search dimensions/nuisance parameters ###
    #######################################################

    ### Define scan values here ### 
    print 'Preparing scan parameters...'
    if ndim == 1:
        scan_params = ScanParameters(names = ['mu', 'sigma'],
                                     bounds = [(-0.7, 0.7), (0.04,0.04)],
                                     nscans = [25, 1]
                                    )
    elif ndim == 2:
        scan_params = ScanParameters(names = ['mu', 'sigma'],
                                     bounds = [(-0.7, 0.7), (0.02,0.1)],
                                     nscans = [25, 25]
                                    )

    paramscan = []
    phiscan   = []
    qmaxscan  = []
    u_0       = np.linspace(0.01, 25., 1250.)
    for i, sim in enumerate(sims):
        if i%10 == 0: print 'Carrying out scan {0}...'.format(i+1)

        # fit background model
        bg_result = bg_fitter.fit(sim)
        if bg_result.status == 0:
            nll_bg = bg_model.calc_nll(sim)
        else:
            continue

        # scan over signal parameters
        nllscan, params = sig_fitter.scan(scan_params, sim) 
        qscan = -2*(nllscan - nll_bg)
        paramscan.append(params)
        qmaxscan.append(np.max(qscan))

        ### Calculate E.C. of the random field
        qscan = np.array(qscan).reshape(scan_params.nscans)
        phiscan.append([lee.calculate_euler_characteristic((qscan > u) + 0.) for u in u_0])

        if make_plots and i < 50:
            sig_model.update_parameters(params)
            bg_model.update_parameters(bg_result.x)
            ft.fit_plot(sim, xlimits, sig_model, bg_model,
                        '{0}_{1}'.format(channel,i+1), path='plots/scan_fits')
            if ndim == 2:
                cmap = plt.imshow(qscan.transpose(), cmap='viridis', vmin=0., vmax=10.) 
                plt.colorbar()
                plt.savefig('plots/scan_fits/qscan_{0}_{1}.png'.format(channel, i))
                plt.savefig('plots/scan_fits/qscan_{0}_{1}.pdf'.format(channel, i))
                plt.close()

    phiscan     = np.array(phiscan)
    paramscan   = np.array(paramscan)
    qmaxscan    = np.array(qmaxscan)

    ################################
    ### Calculate LEE correction ###
    ################################

    if not is_batch:
        k1, nvals1, p_global = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=1, do_fit=False)
        k2, nvals2, p_global = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=2, do_fit=False)
        k, nvals, p_global   = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim)

        if make_plots:
            lee.validation_plots(u_0, phiscan, qmaxscan, 
                                 [nvals1, nvals2, nvals], [k1, k2, k], 
                                 #[nvals], [k], 
                                 '{0}_{1}D'.format(channel, ndim))

        print 'k = {0:.2f}'.format(k)
        for i,n in enumerate(nvals):
            print 'N{0} = {1:.2f}'.format(i, n)
        print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(norm.cdf(-np.sqrt(qmax)), np.sqrt(qmax))
        print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))

    # Save scan data
    if save_data or is_batch:
        outfile = open('data/lee_scan_{0}_{1}.pkl'.format('combined', nsims), 'w')
        pickle.dump(u_0, outfile)
        pickle.dump(qmaxscan, outfile)
        pickle.dump(phiscan, outfile)
        pickle.dump(paramscan, outfile)
        outfile.close()

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
