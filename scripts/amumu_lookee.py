#!/usr/bin/env python

import sys, pickle
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from tqdm import tqdm

from nllfitter import Parameters, ScanParameters, Model, NLLFitter
import nllfitter.fit_tools as ft
import nllfitter.lookee as lee


def sig_constraint(sig_pdf, a):
    '''
    Constraint for preventing signal pdf from going negative.  Evaluates
    sig_pdf at the mean value of the signal Gaussian.

    Parameters:
    ===========
    sig_pdf: polynomial + Gaussian model
    x: data
    a: model parameters (A, mu, sigma, a1, a2)
    '''

    fmin = sig_pdf(a[1], a)
    if fmin <= 0.0001:
        return np.inf
    else:
        return 0

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
        ndim    = 1

    #####################
    ### Configuration ###
    #####################

    xlimits    = (12., 70.)
    make_plots = True
    is_batch   = False
    model      = 'Gaussian'

    ########################
    ### Define fit model ###
    ########################

    if channel == 'combined':
        data_1b1f, n_1b1f = ft.get_data('data/fit/events_pf_1b1f.csv', 'dimuon_mass')
        data_1b1c, n_1b1c = ft.get_data('data/fit/events_pf_1b1c.csv', 'dimuon_mass')
        data = np.concatenate((data_1b1f, data_1b1c))
        n_total = n_1b1f + n_1b1c
    else:
        data, n_total = ft.get_data('data/fit/events_pf_{0}.csv'.format(channel), 'dimuon_mass')

    data = data[data<70]
    n_total = data.size

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
    if model == 'Gaussian':
        sig_params.add_many(
                            ('A'     , 0.01 , True , 0.0  , 1.  , None) ,
                            ('mu'    , 30.  , True , 16.  , 66. , None) ,
                            ('sigma' , 1.   , True , 0.45 , 3.  , None)
                           )
        sig_params += bg_params.copy()
        sig_model  = Model(ft.sig_pdf, sig_params)
    elif model == 'Voigt':
        sig_params.add_many(
                            ('A'     , 0.01 , True , 0.0 , 1.  , None) ,
                            ('mu'    , 30.  , True , 16. , 66. , None) ,
                            ('gamma' , 1.9  , True , 0.1 , 2.5 , None)
                           )
        sig_params += bg_params.copy()
        sig_model  = Model(ft.sig_pdf_alt, sig_params)
    sig_fitter = NLLFitter(sig_model, verbose=False)#, fcons=sig_constraint)
    sig_result = sig_fitter.fit(data)

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    qmax = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))

    ### Generate toy MC ###
    sims = ft.generator(bg_model.pdf, xlimits, n_total, ntoys=nsims)

    #######################################################
    ### Scan over search dimensions/nuisance parameters ###
    #######################################################

    ### Define scan values here ### 
    mu_max    = sig_params['mu'].value
    sigma_max = sig_params['sigma'].value
    if ndim == 1:
        nscans = [30, 1]
        bnds   = [(16., 66.), (sigma_max, sigma_max)]
        scan_params = ScanParameters(names  = ['mu', 'sigma'],
                                     bounds = bnds,
                                     nscans = nscans
                                    )
    elif ndim == 2:
        nscans = [30, 20]
        bnds   = [(16., 66.), (0.45, 2.)]
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
    sigma     = np.linspace((xlimits[1]-xlimits[0])/2*0.02, (xlimits[1]-xlimits[0])/2*0.15, nscans[1])
    for i, sim in tqdm(enumerate(sims), 
                       desc       = 'Scanning simulation',
                       unit_scale = True,
                       ncols      = 75,
                       total      = len(sims)
                      ):

        # fit background model
        bg_result = bg_fitter.fit(sim)
        if bg_result.status == 0:
            nll_bg = bg_model.calc_nll(sim)
        else:
            continue

        # scan over signal parameters
        nllscan, params, dof = sig_fitter.scan(scan_params, sim) 
        qscan = -2*(nllscan - nll_bg)
        paramscan.append(params)
        qmaxscan.append(np.max(qscan))

        ### Calculate E.C. of the random field
        if qscan.size != np.prod(scan_params.nscans): 
            print 'The scan must have failed :('
            continue

        qscan = np.array(qscan).reshape(scan_params.nscans)
        phiscan.append([lee.calculate_euler_characteristic((qscan > u) + 0.) for u in u_0])

        if make_plots and i < 50:
            sig_model.update_parameters(params)
            bg_model.update_parameters(bg_result.x)
            ft.fit_plot_1D(sim, xlimits, sig_model, bg_model,
                        '{0}_{1}'.format(channel,i+1), path='plots/scan_fits')
            if ndim == 1:
                ft.plot_pvalue_scan_1D(qscan.flatten(), mu, 
                                       path ='plots/scan_fits/pvalue_scans_{0}_{1}_1D.png'.format(channel, i+1))
            if ndim == 2:
                ft.plot_pvalue_scan_2D(qscan.flatten(), mu, sigma,
                                       path ='plots/scan_fits/pvalue_scans_{0}_{1}_2D.png'.format(channel, i+1))

    phiscan     = np.array(phiscan)
    paramscan   = np.array(paramscan)
    qmaxscan    = np.array(qmaxscan)

    # Save scan data
    if is_batch:
        outfile = open('data/lee_scan_{0}_{1}_{2}.pkl'.format(channel, nsims, ndim), 'w')
        pickle.dump(u_0, outfile)
        pickle.dump(qmaxscan, outfile)
        pickle.dump(phiscan, outfile)
        pickle.dump(paramscan, outfile)
        outfile.close()
    else:
        #################################
        ### Calculate GV coefficients ###
        #################################

        param_init = ndim*[1.,]
        param_bnds = ndim*[(0., np.inf), ]
        kvals      = [1]
        scales     = [0.5]
        nvals      = lee.get_GV_coefficients(u_0, phiscan, param_init, param_bnds, kvals, scales)
        nvals      = np.reshape(nvals, (1, ndim))

        ### Calculate statistics ###
        p_local  = 0.5*chi2.sf(qmax, 1)
        z_local  = -norm.ppf(p_local)
        p_global = lee.get_p_global(qmax, [1], nvals, [0.5])
        z_global = -norm.ppf(p_global)

        if make_plots:
            lee.gv_validation_plot(u_0, phiscan, qmaxscan, 
                                   nvals, [1], [0.5],
                                   '{0}_{1}D'.format(channel, ndim))

        print ''
        for i, n in enumerate(nvals.flatten()): 
            print 'N{0} = {1:.2f}'.format(i+1, n)

        print 'local p value       = {0:.3e}'.format(p_local)
        print 'local significance  = {0:.2f}'.format(z_local)
        print 'global p value      = {0:.3e}'.format(p_global)
        print 'global significance = {0:.2f}'.format(z_global)
        print 'trial factor        = {0:.2f}'.format(p_global/p_local)

    print ''
    print 'Runtime = {0:.2f} s'.format((timer() - start))
