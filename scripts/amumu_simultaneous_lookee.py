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
        ndim    = 1

    ### Config 
    minalgo    = 'SLSQP'
    xlimits    = (12., 70.)
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
        nscans = (50, 1)
        bnds   = [(-0.8, 0.8), (sigma_max, sigma_max)]
        scan_params = ScanParameters(names  = ['mu', 'sigma'],
                                     bounds = bnds,
                                     nscans = nscans
                                    )
    elif ndim == 2:
        nscans = (30, 20)
        bnds   = [(-0.8, 0.8), (0.02, 0.07)]
        scan_params = ScanParameters(names = ['mu', 'sigma'],
                                     bounds = bnds,
                                     nscans = nscans 
                                    )

    paramscan = []
    phiscan   = []
    dofs      = []
    qmaxscan  = []
    u_0       = np.linspace(0.01, 30., 300)
    mu        = np.linspace(xlimits[0]+0.1*(xlimits[1] - xlimits[0]), 
                            xlimits[1]-0.1*(xlimits[1] - xlimits[0]), nscans[0]) 
    sigma     = np.linspace((xlimits[1]-xlimits[0])/2*0.02, (xlimits[1]-xlimits[0])/2*0.15, nscans[1])
    for i, sim in enumerate(sims):
        if i%10 == 0: 
            print 'Carrying out scan {0}...'.format(i+1)

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

        ### Calculate E.C. of the random field
        if qscan.size != np.prod(scan_params.nscans): 
            print 'The scan must have failed :('
            continue

        qscan = np.array(qscan).reshape(nscans)
        phiscan.append(np.array([lee.calculate_euler_characteristic((qscan > u) + 0.) for u in u_0]))
        paramscan.append(params)
        qmaxscan.append(np.max(qscan))
        dofs.append(dof)

        if make_plots and i < 50:
            #for j, channel in enumerate(channels):
            #    ft.fit_plot(sim[j], xlimits, sig_model.models[j], bg_model.models[j],
            #               'simultaneous_{0}_{1}'.format(channel,i+1), path='plots/scan_fits')

            if ndim == 1:
                ft.plot_pvalue_scan_1D(qscan.flatten(), mu, 
                                       path ='plots/scan_fits/pvalue_scans_{0}_{1}_1D.png'.format(channel, i+1))
            if ndim == 2:
                ft.plot_pvalue_scan_2D(qscan.flatten(), mu, sigma,
                                       path ='plots/scan_fits/pvalue_scans_{0}_{1}_2D.png'.format(channel, i+1))

    phiscan   = np.array(phiscan)
    paramscan = np.array(paramscan)
    qmaxscan  = np.array(qmaxscan)
    dofs      = np.array(dofs)

    if is_batch:
        # Save scan data
        outfile = open('data/lee_scan_simultaneous_{0}_{1}.pkl'.format(nsims, ndim), 'w')
        pickle.dump(u_0, outfile)
        pickle.dump(qmaxscan, outfile)
        pickle.dump(phiscan, outfile)
        pickle.dump(paramscan, outfile)
        pickle.dump(dofs, outfile)
        outfile.close()
    else:
        #################################
        ### Calculate GV coefficients ###
        #################################



        if ndim == 1:
            param_init = [24.47, 30.41, 1.]
            param_bnds = [(24.47, 24.47), (30.09,30.09), (0., np.inf)]
            scales = [0.25, 0.25, 0.25]
            kvals  = [1, 1, 2]
        elif ndim == 2:
            param_init = [26.97, 47.97, 28.27, 44.96, 1., 1.]
            param_bnds = [(26.97, 26.97), (47.97, 47.97), (28.27, 28.27), (44.96, 44.96), (0., np.inf), (0., np.inf)]
            scales = [0.25, 0.25, 0.25]
            kvals  = [1, 1, 2]
        else:
            exit()
        nvals  = lee.get_GV_coefficients(u_0, phiscan, param_init, param_bnds, kvals, scales)

        ### Calculate statistics ###
        p_local  = 0.5*chi2.sf(qmax, 1) + 0.25*chi2.sf(qmax, 2) # according to Chernoff 
        z_local  = -norm.ppf(p_local)
        p_global = lee.get_p_global(qmax, kvals, nvals, scales)
        z_global = -norm.ppf(p_global)

        for i, n in enumerate(nvals.flatten()): 
            print 'N{0} = {1:.2f}'.format(i, n)

        print 'local p_value       = {0:.3e}'.format(p_local)
        print 'local significance  = {0:.2f}'.format(z_local)
        print 'global p_value      = {0:.3e}'.format(p_global)
        print 'global significance = {0:.2f}'.format(z_global)
        print 'trial factor        = {0:.2f}'.format(p_global/p_local)

        lee.gv_validation_plot(u_0, phiscan, qmaxscan, 
                               nvals, kvals, scales, 
                               channel='simultaneous_{0}D'.format(ndim))


    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
