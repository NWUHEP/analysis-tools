from __future__ import division

import sys
from timeit import default_timer as timer

import numpy as np
from numpy.polynomial.legendre import legval
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from scipy import integrate
from lmfit import Parameter, Parameters
from tqdm import tqdm

from nllfitter import Model, NLLFitter
import nllfitter.fit_tools as ft
import nllfitter.plot_tools as pt


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

    if len(sys.argv) > 2:
        category = str(sys.argv[1])
        channel  = str(sys.argv[2])
        period   = int(sys.argv[3])
    else:
        category = 'mumu'
        channel  = '1b1f'
        period   = 2012

    ### Configuration
    pt.set_new_tdr()
    do_sync     = False
    do_mc       = False
    doToys      = False
    doKS        = False
    nsims       = 1000
    model       = 'Voigt'
    ntuple_dir  = 'data/flatuples/{0}_test_{1}'.format(category, period)
    output_path = 'plots/fits/{0}_{1}'.format(category, period)

    if period == 2012:
        datasets    = ['muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D']
    elif period == 2016:
        datasets    = [
                       'muon_2016B', 'muon_2016C', 'muon_2016D', 
                       #'muon_2016E', 'muon_2016F', 'muon_2016G', 
                       #'muon_2016H'
                      ]

    cuts        = 'lepton1_pt > 25 and abs(lepton1_eta) < 2.1 \
                   and lepton2_pt > 25 and abs(lepton2_eta) < 2.1 \
                   and lepton1_q != lepton2_q \
                   and 12 < dilepton_mass < 70'# \
                   #and dilepton_pt_over_m > 2'
                   #and n_bjets > 0 \

    if channel == '1b1f':
        cuts += ' and n_bjets == 1 and n_fwdjets > 0 and n_jets == 0'
    elif channel == '1b1c':
        cuts += ' and n_bjets > 0 and n_fwdjets == 0 and (n_bjets + n_jets) == 2 \
                  and four_body_delta_phi > 2.5 and met_mag < 40'
    elif channel == 'combined':
        cuts += ' and ((n_bjets == 1 and n_fwdjets > 0 and n_jets == 0) or \
                  (n_bjets > 0 and n_fwdjets == 0 and (n_bjets + n_jets) == 2 \
                  and four_body_delta_phi > 2.5 and met_mag < 40))'

    ### Get dataframes with features for each of the datasets ###
    xlimits = (12, 70)
    if do_sync:
        if channel == 'combined':
            data_1b1f, n_1b1f = ft.get_data('data/fit/events_pf_1b1f.csv', 'dimuon_mass')
            data_1b1c, n_1b1c = ft.get_data('data/fit/events_pf_1b1c.csv', 'dimuon_mass')
            data = np.concatenate((data_1b1f, data_1b1c))
        else:
            data, _ = ft.get_data('data/fit/events_pf_{0}.csv'.format(channel), 'dimuon_mass')
        data    = data[data <= 70]
        n_total = data.size
    elif do_mc:
        df_data = pd.read_csv('data/fit/dimuon_mc_mix_preselection.csv', sep=' ')
        df_data = df_data.query(cuts)
        data    = df_data['dilepton_mass'].values
        n_total = data.size

    else:
        data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                      dataset_names = datasets,
                                      selection     = category,
                                      period        = str(period),
                                      cuts          = cuts
                                     )
        df_data = data_manager.get_dataframe('data')
        df_data.to_csv('data/fit/amumu_{0}_{1}.csv'.format(period, channel), index=False)
        data = df_data['dilepton_mass'].values
        n_total = data.size

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
                            ('A'     , 0.01 , True , 0.0  , 1.  , None) ,
                            ('mu'    , 30.  , True , 20.  , 50. , None) ,
                            ('sigma' , 1.   , True , 0.45 , 3.  , None)
                           )
        sig_params += bg_params.copy()
        sig_model  = Model(ft.sig_pdf, sig_params)
    elif model == 'Voigt':
        sig_params.add_many(
                            ('A'     , 0.01 , True , 0.0 , 1.  , None) ,
                            ('mu'    , 29.  , True , 20. , 50. , None) ,
                            ('gamma' , 1.9  , True , 0.1 , 2.  , None)
                           )
        sig_params += bg_params.copy()
        sig_model  = Model(ft.sig_pdf_alt, sig_params)

    sig_fitter = NLLFitter(sig_model)
    sig_result = sig_fitter.fit(data)

    ### Plots!!! ###
    print 'Making plot of fit results...'
    pt.make_directory(output_path, clear=False)
    ft.fit_plot_1D(data, xlimits, 
                   sig_model, bg_model, 
                   (category, channel, model),
                   path=output_path
                  )

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q_max = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))
    p_value = 0.5*chi2.sf(q_max, 1)
    print 'q = {0:.3f}'.format(q_max)
    print 'p_local = {0:.3e}'.format(p_value)
    print 'z_local = {0}'.format(-norm.ppf(p_value))
    print ''

    ### Calculate the number of events in around the peak
    f_bg    = lambda x: ft.bg_pdf(x, (sig_result.x[3], sig_result.x[4]))
    if model == 'Gaussian':
        xlim    = (sig_result.x[1] - 2*sig_result.x[2], sig_result.x[1] + 2*sig_result.x[2])
        sig_b   = 4*n_total*f_bg(sig_result.x[1])*sig_model.get_parameters()['sigma'].stderr
    elif model == 'Voigt':
        xlim    = (sig_result.x[1] - sig_result.x[2], sig_result.x[1] + sig_result.x[2])
        sig_b   = 2*n_total*f_bg(sig_result.x[1])*sig_model.get_parameters()['gamma'].stderr

    N_b     = n_total*integrate.quad(f_bg, xlim[0], xlim[1])[0]
    N_s     = sig_result.x[0]*n_total
    sig_s   = n_total*sig_model.get_parameters()['A'].stderr

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

        plt.savefig('plots/fits/{0}/ks_test_{0}.pdf'.format(period, channel))
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
        for i, sim in enumerate(tqdm(sims, total=nsims, ncols=75)):

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
        plt.legend([r'$\frac{1}{2}\chi^{2}_{1} + \frac{1}{2}\delta_{0}$', 'pseudodata'])
        plt.xlabel(r'$q$')
        plt.ylabel(r'Entries')

        plt.savefig('plots/fits/{0}/q_distribution_{1}_{2}.pdf'.format(period, channel, model))
        plt.close()

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
