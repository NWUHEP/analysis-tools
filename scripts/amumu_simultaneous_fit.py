from __future__ import division

from timeit import default_timer as timer

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from lmfit import Parameter, Parameters

from nllfitter.fit_tools import get_data, fit_plot, scale_data
import nllfitter.fit_tools as ft
from nllfitter import Model, CombinedModel, NLLFitter


if __name__ == '__main__':

    ### Start the timer
    start = timer()

    ### get data and convert variables to be on the range [-1, 1]
    xlimits  = (12., 70.)
    period   = 2012
    channels = ['1b1f', '1b1c']

    ### For post fit tests
    doToys   = False
    nsims    = 10000

    datasets  = []
    if period == 2012:
        for channel in channels:
            data, n_total  = ft.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
            #data, n_total  = ft.get_data('data/test_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
            datasets.append(data)
    elif period == 2016:
        for channel in channels:
            data, n_total  = ft.get_data('data/muon_2016_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
            datasets.append(data)

    ### Fit single models to initialize parameters ###
    ### Define bg model and carry out fit ###
    bg1_params = Parameters()
    bg1_params.add_many(
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None)
                      )
    bg1_model = Model(ft.bg_pdf, bg1_params)
    bg_fitter = NLLFitter(bg1_model)
    bg_result = bg_fitter.fit(datasets[0])

    bg2_params = Parameters()
    bg2_params.add_many(
                        ('b1', 0., True, None, None, None),
                        ('b2', 0., True, None, None, None)
                       )
    bg2_model = Model(ft.bg_pdf, bg2_params)
    bg_fitter = NLLFitter(bg2_model)
    bg_result = bg_fitter.fit(datasets[1])

    ### Carry out combined background fit ###
    bg_model = CombinedModel([bg1_model, bg2_model])
    bg_fitter = NLLFitter(bg_model)
    bg_result = bg_fitter.fit(datasets)

    ### Define bg+sig model and carry out fit ###
    sig1_params = Parameters()
    sig1_params.add_many(
                        ('A1'    , 0.01 , True , 0.0 , 1.   , None),
                        ('mu'    , -0.5 , True , -0.9 , 0.9  , None),
                        ('sigma' , 0.01 , True , 0.02 , 1.   , None)
                       )
    sig1_params += bg1_params.copy()
    sig1_model = Model(ft.sig_pdf_alt, sig1_params)
    sig_fitter = NLLFitter(sig1_model)
    sig_result = sig_fitter.fit(datasets[0])

    sig2_params = Parameters()
    sig2_params.add_many(
                        ('A2'    , 0.01 , True , 0.0 , 1.   , None),
                        ('mu'    , -0.5 , True , -0.9 , 0.9  , None),
                        ('sigma' , 0.01 , True , 0.02 , 1.   , None)
                       )
    sig2_params += bg2_params.copy()
    sig2_model = Model(ft.sig_pdf_alt, sig2_params)
    sig_fitter = NLLFitter(sig2_model)
    sig_result = sig_fitter.fit(datasets[1])

    ### Carry out combined signal+background fit ###
    sig_model  = CombinedModel([sig1_model, sig2_model])
    sig_fitter = NLLFitter(sig_model)
    sig_result = sig_fitter.fit(datasets)

    q = 2*(bg_model.calc_nll(datasets) - sig_model.calc_nll(datasets))
    p_value = 0.5*chi2.sf(q, 1) + 0.25*chi2.sf(q, 2) # according to Chernoff 
    print '{0}: q = {1:.3f}'.format('a->mumu', q)
    print 'p_local = {0:.3e}'.format(p_value)
    print 'z_local = {0}'.format(-norm.ppf(p_value))
    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))

    ### Turn off fit verbosity for further tests
    bg_fitter.verbose  = False
    sig_fitter.verbose = False

    if doToys:

        ### Generate toy data
        sims = []
        for model, dataset in zip(bg_model.models, datasets):
            simdata = ft.generator(model.pdf, dataset.size, ntoys=nsims)
            sims.append(simdata)
        sims = zip(sims[0], sims[1])

        ### Fix mu and sigma
        sig_model.set_bounds('mu', sig_result.x[1], sig_result.x[1])
        sig_model.set_bounds('sigma', sig_result.x[2], sig_result.x[2])

        qmax = []
        params = []
        for i, sim in enumerate(sims):
            if i%1000 == 0: print 'Carrying out fit to pseudodata {0}...'.format(i+1)

            sim = list(sim) # something doesn't like tuples downstream...
            bg_result  = bg_fitter.fit(sim, calculate_corr=False)
            sig_result = sig_fitter.fit(sim, calculate_corr=False)
            if sig_result.status == 0 and bg_result.status == 0:
                nll_bg  = bg_model.calc_nll(sim)
                nll_sig = sig_model.calc_nll(sim)
                qmax.append(2*(nll_bg - nll_sig))
                params.append(sig_result.x)
            else:
                continue

        param_names = sig_model.get_parameters().keys()
        df_params = pd.DataFrame(params, columns=param_names)
        df_params['q'] = qmax

        x = np.linspace(0, 20, 2000)
        plt.yscale('log')
        plt.hist(qmax, bins=50, normed=True, histtype='step', color='k', linewidth=2)
        plt.plot(x, 0.5*chi2.pdf(x, 1) + 0.25*chi2.pdf(x, 2), 'r--', linewidth=2)

        plt.grid()
        plt.xlim(0, 15)
        plt.ylim(.5/nsims, 5)
        plt.title('{0} {1}: {2:d} toys'.format(period, 'combination', nsims))
        plt.legend([r'$\frac{1}{2}\chi^{2}_{1} + \frac{1}{4}\chi^{2}_{2} + \frac{1}{4}\delta_{0}$', 'pseudodata'])
        plt.xlabel(r'$q$')
        plt.ylabel(r'Entries')

        plt.savefig('plots/q_distribution_{0}_{1}.pdf'.format('combination', period))
        plt.close()

    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
