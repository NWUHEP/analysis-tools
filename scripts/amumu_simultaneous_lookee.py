#!/usr/bin/env python

from __future__ import division
import sys, pickle
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, chi2

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
    make_plots = True
    is_batch   = False

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
                        ('A1'    , 0.01 , True , 0.   , 1.   , None),
                        ('mu'    , -0.5 , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.01 , True , 0.02 , 1.   , None)
                       )
    sig1_params += bg1_params.copy()
    sig1_model = Model(ft.sig_pdf, sig1_params)
    sig_fitter = NLLFitter(sig1_model, verbose=False)
    sig_result = sig_fitter.fit(datasets[0])

    sig2_params = Parameters()
    sig2_params.add_many(
                        ('A2'    , 0.01 , True , 0.   , 1.   , None),
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

    sig_params = sig_model.get_parameters()
    mu_max    = sig_params['mu'].value
    sigma_max = sig_params['sigma'].value
    if ndim == 1:
        nscans = [50, 1]
        bnds   = [(-0.8, 0.8), (sigma_max, sigma_max)]
        scan_params = ScanParameters(names  = ['mu', 'sigma'],
                                     bounds = bnds,
                                     nscans = nscans
                                    )
    elif ndim == 2:
        nscans = [50, 50]
        bnds   = [(-0.8, 0.8), (0.02, 0.15)]
        scan_params = ScanParameters(names = ['mu', 'sigma'],
                                     bounds = bnds,
                                     nscans = nscans 
                                    )

    paramscan = []
    phiscan   = []
    qmaxscan  = []
    u_0       = np.linspace(0.01, 30., 300)
    mu        = np.linspace(xlimits[0]+0.1*(xlimits[1] - xlimits[0]), 
                            xlimits[1]-0.1*(xlimits[1] - xlimits[0]), nscans[0]) 
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
        nllscan, params, dof = sig_fitter.scan(scan_params, sim, amps=[0, 5]) 
        qscan = -2*(nllscan - nll_bg)
        paramscan.append(params)
        qmaxscan.append(np.max(qscan))

        ### Calculate E.C. of the random field
        if qscan.size != np.prod(scan_params.nscans): 
            print 'The scan must have failed :('
            continue

        qscan   = np.array(qscan).reshape(nscans)
        phiscan.append([lee.calculate_euler_characteristic((qscan > u) + 0.) for u in u_0])

        if make_plots and i < 50:
            #for j, channel in enumerate(channels):
            #    ft.fit_plot(sim[j], xlimits, sig_model.models[j], bg_model.models[j],
            #               'simultaneous_{0}_{1}'.format(channel,i+1), path='plots/scan_fits')

            if ndim == 1:
                ft.plot_pvalue_scan_1D(qscan.flatten(), mu, 'simultaneous_{0}'.format(i+1))
            elif ndim == 2:
                ft.plot_pvalue_scan_2D(qscan.flatten(), mu, sigma, 'simultaneous_{0}'.format(i+1))

    phiscan     = np.array(phiscan)
    paramscan   = np.array(paramscan)
    qmaxscan    = np.array(qmaxscan)

    if is_batch:
        # Save scan data
        outfile = open('data/lee_scan_simultaneous_{0}_{1}.pkl'.format(nsims, ndim), 'w')
        pickle.dump(u_0, outfile)
        pickle.dump(qmaxscan, outfile)
        pickle.dump(phiscan, outfile)
        pickle.dump(paramscan, outfile)
        outfile.close()
    else:
        ################################
        ### Calculate LEE correction ###
        ################################

        p_local = 0.5*chi2.sf(qmax, 1) + 0.25*chi2.sf(qmax, 2) # according to Chernoff 
        z_local = -norm.ppf(p_local)

        ### Before carrying this out, the E.C. coefficients should be determined for
        ### each of the channels for the separate channels, i.e., when k=1.  The
        ### construction of Gross-Vitells in this case is informed by the construction of
        ### the local p value from Chernoff

        k, nvals = lee.get_GV_coefficients(u_0, phiscan, j=ndim, k=2, scale=1)
        p_global = lee.get_p_global(qmax, k, nvals, scale=1)
        z_global = -norm.ppf(p_global)

        if make_plots:
            lee.gv_validation_plot(u_0, phiscan, qmaxscan, 
                                   nvals, k, scale=1, 
                                   channel='simultaneous_{0}D'.format(ndim))

        print 'k = {0:.2f}'.format(k)
        for i,n in enumerate(nvals):
            print 'N{0} = {1:.2f}'.format(i, n)
        print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(p_local, z_local)
        print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, z_global)


    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
