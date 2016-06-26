#!/usr/bin/env python

from __future__ import division
import sys, pickle
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from nllfitter import Parameters, ScanParameters, Model, CombinedModel, NLLFitter
import nllfitter.fit_tools as ft
import nllfitter.lookee as lee

if __name__ == '__main__':
    start = timer()

    ### Get command line arguments
    if len(sys.argv) > 2:
        nsims   = int(sys.argv[1])
        ndim    = int(sys.argv[2])
    else:
        nsims   = 100
        ndim    = 2

    ### Config 
    minalgo    = 'SLSQP'
    xlimits    = (12., 70.)
    nscan      = (50, 30)
    channels   = ['1b1f', '1b1c']
    make_plots = False
    save_data  = False
    is_batch   = True

    ########################
    ### Define fit model ###
    ########################

    datasets  = []
    for channel in channels:
        data, n_total  = ft.get_data('data/events_pf_{0}.csv'.format(channel), 
                                  'dimuon_mass', xlimits)
        datasets.append(data)

    ### Fit single models to initialize parameters ###
    ### Define bg model and carry out fit ###
    bg1_params = Parameters()
    bg1_params.add_many(
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None)
                      )
    bg1_model = Model(ft.bg_pdf, bg1_params)
    bg_fitter = NLLFitter(bg1_model, verbose=False)
    bg_result = bg_fitter.fit(datasets[0])

    bg2_params = Parameters()
    bg2_params.add_many(
                       ('b1', 0., True, None, None, None),
                       ('b2', 0., True, None, None, None)
                      )
    bg2_model = Model(ft.bg_pdf, bg2_params)
    bg_fitter = NLLFitter(bg2_model, verbose=False)
    bg_result = bg_fitter.fit(datasets[1])

    ### Carry out combined background fit ###
    bg_model = CombinedModel([bg1_model, bg2_model])
    bg_fitter = NLLFitter(bg_model, verbose=False)
    bg_result = bg_fitter.fit(datasets)

    ### Define bg+sig model and carry out fit ###
    sig1_params = Parameters()
    sig1_params.add_many(
                        ('A1'    , 0.01 , True , 0.01 , 1.   , None),
                        ('mu'    , -0.5 , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.01 , True , 0.02 , 1.   , None)
                       )
    sig1_params += bg1_params.copy()
    sig1_model = Model(ft.sig_pdf, sig1_params)
    sig_fitter = NLLFitter(sig1_model, verbose=False)
    sig_result = sig_fitter.fit(datasets[0])

    sig2_params = Parameters()
    sig2_params.add_many(
                        ('A2'    , 0.01 , True , 0.01 , 1.   , None),
                        ('mu'    , -0.5 , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.01 , True , 0.02 , 1.   , None)
                       )
    sig2_params += bg2_params.copy()
    sig2_model = Model(ft.sig_pdf, sig2_params)
    sig_fitter = NLLFitter(sig2_model, verbose=False)
    sig_result = sig_fitter.fit(datasets[1])

    ### Carry out combined signal+background fit ###
    sig_model  = CombinedModel([sig1_model, sig2_model])
    sig_fitter = NLLFitter(sig_model, verbose=False)
    sig_result = sig_fitter.fit(datasets)


    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    qmax = 2*(bg_model.calc_nll(datasets) - sig_model.calc_nll(datasets))

    ### Generate toy MC ###
    print 'Generating {0} pseudodatasets for likelihood scans...'.format(nsims)
    sims = []
    for model, dataset in zip(bg_model.models, datasets):
        simdata = ft.generator(model.pdf, dataset.size, ntoys=nsims)
        sims.append(simdata)
    sims = zip(sims[0], sims[1])

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

        sim = list(sim) # fitter doesn't like tuples for some reason... 
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
            #for j, channel in enumerate(channels):
            #    ft.fit_plot(sim[j], xlimits, sig_model.models[j], bg_model.models[j],
            #               'combined_{0}_{1}'.format(channel,i+1), path='plots/scan_fits')
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

    if is_batch:
        # Save scan data
        outfile = open('data/lee_scan_{0}_{1}_{2}.pkl'.format('combined', nsims, ndim), 'w')
        pickle.dump(u_0, outfile)
        pickle.dump(qmaxscan, outfile)
        pickle.dump(phiscan, outfile)
        pickle.dump(paramscan, outfile)
        outfile.close()

    elif not is_batch:
        k1, nvals1, p_global = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=1, do_fit=False)
        k2, nvals2, p_global = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=2, do_fit=False)
        k, nvals, p_global   = lee.lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim)

        if make_plots:
            lee.validation_plots(u_0, phiscan, qmaxscan, 
                                 [nvals1, nvals2, nvals], [k1, k2, k], 
                                 #[nvals], [k], 
                                 '{0}_{1}D'.format('combined', ndim))

        print 'k = {0:.2f}'.format(k)
        for i,n in enumerate(nvals):
            print 'N{0} = {1:.2f}'.format(i, n)
        print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(norm.cdf(-np.sqrt(qmax)), np.sqrt(qmax))
        print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))


    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
