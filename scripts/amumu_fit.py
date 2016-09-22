from __future__ import division

import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from scipy import integrate
from lmfit import Parameter, Parameters

from nllfitter import Model, NLLFitter
import nllfitter.fit_tools as ft

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
    if fmin <= 0:
        return np.inf
    else:
        return 0

if __name__ == '__main__':

    ### Start the timer
    start = timer()

    ### Configuration
    verbose = True
    doToys  = False
    doKS    = False
    model   = 'Voigt'
    nsims   = 50000

    if len(sys.argv) > 2:
        channel = str(sys.argv[1])
        period  = int(sys.argv[2])
    else:
        channel = '1b1f'
        period  = 2012

    print 'Getting data and scaling to lie in range [-1, 1].'
    xlimits = (12., 70.)
    if period == 2012:
        if channel == 'combined':
            #data_1b1f, n_1b1f = ft.get_data('data/muon_2012_1b1f.csv', 'dimuon_mass', xlimits)
            #data_1b1c, n_1b1c = ft.get_data('data/muon_2012_1b1c.csv', 'dimuon_mass', xlimits)
            data_1b1f, n_1b1f = ft.get_data('data/events_pf_1b1f.csv', 'dimuon_mass', xlimits)
            data_1b1c, n_1b1c = ft.get_data('data/events_pf_1b1c.csv', 'dimuon_mass', xlimits)
            data = np.concatenate((data_1b1f, data_1b1c))
            n_total = n_1b1f + n_1b1c
        else:
            #data, n_total = ft.get_data('data/muon_2012_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
            #data, n_total = ft.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
            #data, n_total = ft.get_data('data/emu_2012_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
            data, n_total = ft.get_data('data/dimuon_mass_{0}_cut2_2012.csv'.format(channel), 'dimuon_mass', xlimits)
    elif period == 2016:
        if channel == 'combined':
            data_1b1f, n_1b1f = ft.get_data('data/muon_2016_1b1f.csv', 'dimuon_mass', xlimits)
            data_1b1c, n_1b1c = ft.get_data('data/muon_2016_1b1c.csv', 'dimuon_mass', xlimits)
            data = np.concatenate((data_1b1f, data_1b1c))
            n_total = n_1b1f + n_1b1c
        else:
            data, n_total = ft.get_data('data/muon_2016_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    elif period == 0:
        if channel == 'combined':
            data_1b1f_2016, n_1b1f_2016 = ft.get_data('data/muon_2016_1b1f.csv', 'dimuon_mass', xlimits)
            data_1b1c_2016, n_1b1c_2016 = ft.get_data('data/muon_2016_1b1c.csv', 'dimuon_mass', xlimits)
            data_1b1f_2012, n_1b1f_2012 = ft.get_data('data/muon_2012_1b1f.csv', 'dimuon_mass', xlimits)
            data_1b1c_2012, n_1b1c_2012 = ft.get_data('data/muon_2012_1b1c.csv', 'dimuon_mass', xlimits)
            data = np.concatenate((data_1b1f_2016, data_1b1c_2016, data_1b1f_2012, data_1b1c_2012))
            n_total = n_1b1f_2012 + n_1b1c_2012 + n_1b1f_2016 + n_1b1c_2016

    print 'Analyzing {0} events...\n'.format(n_total)

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
    if model == 'Gaussian':
        sig_params.add_many(
                            ('A'     , 0.01  , True , 0.0   , 1.   , None),
                            ('mu'    , -0.43 , True , -0.1  , -0.7 , None),
                            ('sigma' , 0.04  , True , 0.015 , 0.2  , None)
                           )
        sig_model  = Model(ft.sig_pdf, sig_params)
    elif model == 'Voigt':
        sig_params.add_many(
                            ('A'     , 0.01   , True , 0.0    , 1.     , None),
                            ('mu'    , -0.43  , True , -0.45  , -0.4   , None),
                            ('gamma' , 0.033  , True , 0.02   , 0.04   , None),
                           )
        sig_model  = Model(ft.sig_pdf_alt, sig_params)
    sig_params += bg_params.copy()

    sig_fitter = NLLFitter(sig_model)
    sig_result = sig_fitter.fit(data)

    ### Plots!!! ###
    print 'Making plot of fit results...'
    ft.fit_plot(data, xlimits, sig_model, bg_model, '{0}_{1}_{2}'.format(channel, period, model))

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q_max = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))
    p_value = 0.5*chi2.sf(q_max, 1)
    print 'q = {0:.3f}'.format(q_max)
    print 'p_local = {0:.3e}'.format(p_value)
    print 'z_local = {0}'.format(-norm.ppf(p_value))

    ### Calculate the number of events in around the peak
    f_bg    = lambda x: ft.bg_pdf(x, (sig_result.x[3], sig_result.x[4]))
    xlim    = (sig_result.x[1] - 2*sig_result.x[2], sig_result.x[1] + 2*sig_result.x[2])
    N_b     = (1 - sig_result.x[0])*n_total*integrate.quad(f_bg, xlim[0], xlim[1])[0]
    sig_b   = N_b/np.sqrt(n_total*sig_result.x[0])
    N_s     = n_total*sig_result.x[0]
    sig_s   = np.sqrt(N_s)
	#sig_s   = n_total*comb_sigma[0]

    print 'N_b = {0:.2f} +\- {1:.2f}'.format(N_b, sig_b)
    print 'N_s = {0:.2f} +\- {1:.2f}'.format(N_s, sig_s)
    print ''


    ### Turn off fit verbosity for further tests
    bg_fitter.verbose  = False
    sig_fitter.verbose = False

    if doKS:
        ks_res = ft.ks_test(data, sig_model.pdf, make_plots=True, suffix='{0}_{1}'.format(channel, period)) 

        ### Generate toy data
        ks_sups = []
        sims = ft.generator(sig_model.pdf, n_total, ntoys=100)
        for sim in sims:
            sig_result = sig_fitter.fit(sim, calculate_corr=False)
            ks_toy = ft.ks_test(sim, sig_model.pdf)
            ks_sups.append(np.max(ks_toy))
        
        plt.hist(ks_res, bins=25, histtype='step', range=(0., 0.05))
        plt.hist(ks_sups, bins=25, histtype='stepfilled', alpha=0.5, range=(0., 0.05))
        plt.title('{0} {1}'.format(channel, period))
        plt.ylabel(r'Entries')
        plt.xlabel(r'$|\rm CDF_{model} - CDF_{data}|$')
        plt.legend(['ks residuals (data)', 'sup(ks residuals) (toys)'])

        plt.savefig('plots/ks_test_{0}_{1}.pdf'.format(channel, period))
        plt.close()

        print 'Carrying out KS test'
        print 'data: D_n = {0:.3f}'.format(np.max(ks_res))
        print 'toys: D_n = {0:.3f} +/- {1:.3f}'.format(np.mean(ks_sups), np.sqrt(np.var(ks_sups)))

    if doToys:

        ### Generate toy data
        sims = ft.generator(bg_model.pdf, n_total, ntoys=nsims)

        ### Fix mu and sigma
        sig_model.set_bounds('mu', sig_result.x[1], sig_result.x[1])
        sig_model.set_bounds('sigma', sig_result.x[2], sig_result.x[2])

        qmax = []
        for i, sim in enumerate(sims):
            if i%1000 == 0: print 'Carrying out fit to pseudodata {0}...'.format(i+1)

            bg_result  = bg_fitter.fit(sim, calculate_corr=False)
            sig_result = sig_fitter.fit(sim, calculate_corr=False)
            if sig_result.status == 0 and bg_result.status == 0:
                nll_bg  = bg_model.calc_nll(sim)
                nll_sig = sig_model.calc_nll(sim)
                qmax.append(2*(nll_bg - nll_sig))
            else:
                continue

        x = np.linspace(0, 20, 2000)
        plt.yscale('log')
        plt.hist(qmax, bins=50, normed=True, histtype='step', color='k', linewidth=2)
        plt.plot(x, 0.5*chi2.pdf(x, 1), 'r--', linewidth=2)

        plt.grid()
        plt.xlim(0, 15)
        plt.ylim(.5/nsims, 5)
        plt.title('{0} {1}: {2:d} toys'.format(period, channel, nsims))
        plt.legend([r'$\frac{1}{2}\chi^{2}_{1} + \frac{1}{4}\delta_{0}$', 'pseudodata'])
        plt.xlabel(r'$q$')
        plt.ylabel(r'Entries')

        plt.savefig('plots/q_distribution_{0}_{1}_{2}.pdf'.format(channel, period, model))
        plt.close()

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
