from __future__ import division

from timeit import default_timer as timer

import numpy as np
from numpy.polynomial.legendre import legval
import pandas as pd
from scipy.stats import norm, chi2
from scipy import integrate
import matplotlib.pyplot as plt
from lmfit import Parameter, Parameters
from tqdm import tqdm

import nllfitter.fit_tools as ft
from nllfitter.plot_tools import set_new_tdr
from nllfitter import Model, CombinedModel, NLLFitter

def bg_pdf(x, a): 
    '''
    Second order Legendre Polynomial with constant term set to 0.5.

    Parameters:
    ===========
    x: data
    a: model parameters (a1 and a2)
    '''
    z   = ft.scale_data(x, xmin=12, xmax=70)
    fx  = legval(z, [0.5, a[0], a[1]])*2/(70 - 12)
    return fx

def sig_pdf(x, a, normalize=False):
    '''
    Second order Legendre Polynomial (normalized to unity) plus a Gaussian.

    Parameters:
    ===========
    x: data
    a: model parameters (a1, a2, mu, and sigma)
    '''
    bg  = bg_pdf(x, a[3:5])
    sig = norm.pdf(x, a[1], a[2]) 
    sig_norm = 1.
    if normalize:
        sig_norm = integrate.quad(lambda z: norm.pdf(z, a[1], a[2]), -1, 1)[0]

    return (1 - a[0])*bg + a[0]*sig/sig_norm

def sig_pdf_alt(x, a, normalize=True):
    '''
    Second order Legendre Polynomial (normalized to unity) plus a Voigt
    profile. N.B. The width of the convolutional Gaussian is set to 0.155 which
    corresponds to a dimuon mass resolution 0.5 GeV.

    Parameters:
    ===========
    x: data
    a: model parameters (A, a1, a2, mu, and gamma)
    '''
    bg  = bg_pdf(x, a[3:5])
    sig = ft.voigt(x, [a[1], a[2], 0.45])
    if normalize:
        sig_norm = integrate.quad(lambda z: ft.voigt(z, [a[1], a[2], 0.45]), 12, 70)[0]
    else:
        sig_norm = 1.

    return (1 - a[0])*bg + a[0]*sig/sig_norm



if __name__ == '__main__':


    ### Start the timer
    start = timer()

    ### get data and convert variables to be on the range [-1, 1]
    set_new_tdr()
    do_sync  = False
    xlimits  = (12., 70.)
    period   = 2012
    channels = ['1b1f', '1b1c']

    ### For post fit tests
    doToys   = False
    model    = 'Voigt'
    nsims    = 50000

    datasets  = []
    for channel in channels:
        if do_sync:
            data, n_total  = ft.get_data('data/fit/events_pf_{0}.csv'.format(channel), 'dimuon_mass')
            data = data[data < 70]
            datasets.append(data)
        else:
            df = pd.read_csv('data/fit/amumu_2012_{0}.csv'.format(channel))
            datasets.append(df.dilepton_mass.values)

    ### Fit single models to initialize parameters ###
    ### Define bg model and carry out fit ###
    bg1_params = Parameters()
    bg1_params.add_many(
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None)
                      )
    bg1_model = Model(bg_pdf, bg1_params)
    bg_fitter = NLLFitter(bg1_model)
    bg_result = bg_fitter.fit(datasets[0])

    bg2_params = Parameters()
    bg2_params.add_many(
                        ('b1', 0., True, None, None, None),
                        ('b2', 0., True, None, None, None)
                       )
    bg2_model = Model(bg_pdf, bg2_params)
    bg_fitter = NLLFitter(bg2_model)
    bg_result = bg_fitter.fit(datasets[1])

    ### Carry out combined background fit ###
    bg_model = CombinedModel([bg1_model, bg2_model])
    bg_fitter = NLLFitter(bg_model)
    bg_result = bg_fitter.fit(datasets, calculate_corr=True)

    ### Define bg+sig model and carry out fit ###
    sig1_params = Parameters()
    if model == 'Gaussian':
        sig1_params.add_many(
                             ('A1'    , 0.01 , True , 0.0  , 1.  , None) ,
                             ('mu'    , 30.  , True , 20.  , 50. , None) ,
                             ('sigma' , 1.   , True , 0.45 , 3.  , None)
                            )
        sig1_params += bg1_params.copy()
        sig1_model  = Model(sig_pdf, sig1_params)
    elif model == 'Voigt':
        sig1_params.add_many(
                             ('A1'    , 0.01 , True , 0.0  , 1.  , None) ,
                             ('mu'    , 30.  , True , 20.  , 50. , None) ,
                             ('gamma' , 1.   , True , 0.25 , 3.  , None)
                            )
        sig1_params += bg1_params.copy()
        sig1_model  = Model(sig_pdf_alt, sig1_params)

    sig_fitter = NLLFitter(sig1_model)
    sig_result = sig_fitter.fit(datasets[0])

    sig2_params = Parameters()
    if model == 'Gaussian':
        sig2_params.add_many(
                             ('A2'    , 0.01 , True , 0.0  , 1.  , None) ,
                             ('mu'    , 30.  , True , 20.  , 50. , None) ,
                             ('sigma' , 1.   , True , 0.45 , 3.  , None)
                            )
        sig2_params += bg2_params.copy()
        sig2_model  = Model(sig_pdf, sig2_params)
    elif model == 'Voigt':
        sig2_params.add_many(
                             ('A2'    , 0.01 , True , 0.0  , 1.  , None) ,
                             ('mu'    , 30.  , True , 20.  , 50. , None) ,
                             ('gamma' , 1.   , True , 0.25 , 3.  , None)
                            )
        sig2_params += bg2_params.copy()
        sig2_model  = Model(sig_pdf_alt, sig2_params)

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
    print ''

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
        for i, sim in enumerate(tqdm(sims, desc='Fitting toys', unit_scale=True, total=nsims)):

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
        plt.title('{0} {1}: {2:d} toys'.format(period, 'simulataneous', nsims))
        plt.legend([r'$\frac{1}{2}\chi^{2}_{1} + \frac{1}{4}\chi^{2}_{2} + \frac{1}{4}\delta_{0}$', 'pseudodata'])
        plt.xlabel(r'$q$')
        plt.ylabel(r'Entries')

        plt.savefig('plots/fits/{0}/q_distribution_{1}.pdf'.format(period, 'simultaneous'))
        plt.close()
    

    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
