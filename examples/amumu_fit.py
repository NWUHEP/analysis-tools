from __future__ import division

from timeit import default_timer as timer

import numpy as np
import pandas as pd
from scipy.stats import norm
from lmfit import Parameter, Parameters

import nllfitter.fit_tools as ft
from nllfitter import Model, NLLFitter

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
    a: model parameters (A, mu, sigma, a1, a2)
    '''
    return a[0]*norm.pdf(x, a[1], a[2]) + (1 - a[0])*bg_pdf(x, a[3:5])

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
    verbose = True

    ### get data and convert variables to be on the range [-1, 1]
    xlimits = (12., 70.)
    channel = '1b1f'
    doCI    = False

    print 'Getting data and scaling to lie in range [-1, 1].'
    if channel == 'combined':
        data_1b1f, n_1b1f = ft.get_data('data/events_pf_1b1f.csv', 'dimuon_mass', xlimits)
        data_1b1c, n_1b1c = ft.get_data('data/events_pf_1b1c.csv', 'dimuon_mass', xlimits)
        data = np.concatenate((data_1b1f, data_1b1c))
        n_total = n_1b1f + n_1b1c
    if channel == 'test':
        data = pd.read_csv('data/null_spectrum_1.txt', header=None).values
        data = ft.scale_data(data)
        n_total = data.size
    else:
        data, n_total = ft.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    print 'Analyzing {0} events...\n'.format(n_total)

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None)
                      )

    bg_model  = Model(bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model)
    bg_result = bg_fitter.fit(data)

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    sig_params.add_many(
                        ('A'     , 0.01 , True , 0.01 , 1.   , None),
                        ('mu'    , -0.5 , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.01 , True , 0.02 , 1.   , None)
                       )
    sig_params += bg_params.copy()
    sig_model  = Model(ft.sig_pdf_alt, sig_params)
    sig_fitter = NLLFitter(sig_model, fcons=sig_constraint)
    sig_result = sig_fitter.fit(data)

    ### Plots!!! ###
    print 'Making plot of fit results...'
    ft.fit_plot(data, xlimits, sig_model, bg_model, channel)

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q_max = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))
    print 'q = {0}'.format(q_max)

    if doCI:
        ### Calculate confidence interval on the likelihood ratio at the +/- 1, 2
        ### sigma levels
        bg_fitter.verbose  = False
        sig_fitter.verbose = False
        nsims = 1000

        print 'Generating {0} pseudo-datasets from bg+signal fit and determining distribution of q'.format(nsims)
        sims = ft.generator(sig_model.pdf, n_total, ntoys=nsims)
        q_sim = []
        for sim in sims:
            bg_result = bg_fitter.fit(sim)
            sig_result = sig_fitter.fit(sim) 
            if bg_result.status == 0 and sig_result.status == 0:
                nll_bg = bg_model.calc_nll(sim)
                nll_sig = sig_model.calc_nll(sim)
                q = 2*(nll_bg - nll_sig)
                q_sim.append(q)
            else:
                print bg_result.status, sig_result.status

        q_sim = np.array(q_sim)
        q_sim.sort()
        q_upper = q_sim[q_sim > q_max]
        q_lower = q_sim[q_sim < q_max]

        n_upper = q_upper.size
        n_lower = q_lower.size

        one_sigma_up   = q_upper[int(0.34*n_upper)]
        two_sigma_up   = q_upper[int(0.475*n_upper)]
        one_sigma_down = q_lower[int(-0.34*n_lower)]
        two_sigma_down = q_lower[int(-0.475*n_lower)]

        print '{0}: q = {1:.2f}'.format(channel, q_max)
        print '1 sigma c.i.: {0:.2f} -- {1:.2f}'.format(one_sigma_down, one_sigma_up)
        print '2 sigma c.i.: {0:.2f} -- {1:.2f}'.format(two_sigma_down, two_sigma_up)

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
