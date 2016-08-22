from __future__ import division

from timeit import default_timer as timer

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import integrate
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
    period  = 2016
    channel = 'combined'

    print 'Getting data and scaling to lie in range [-1, 1].'
    if period == 2012:
        if channel == 'combined':
            data_1b1f, n_1b1f = ft.get_data('data/test_1b1f.csv', 'dimuon_mass', xlimits)
            data_1b1c, n_1b1c = ft.get_data('data/test_1b1c.csv', 'dimuon_mass', xlimits)
            #data_1b1f, n_1b1f = ft.get_data('data/events_pf_1b1f.csv', 'dimuon_mass', xlimits)
            #data_1b1c, n_1b1c = ft.get_data('data/events_pf_1b1c.csv', 'dimuon_mass', xlimits)
            data = np.concatenate((data_1b1f, data_1b1c))
            n_total = n_1b1f + n_1b1c
        else:
            data, n_total = ft.get_data('data/test_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
            #data, n_total = ft.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    elif period == 2016:
        if channel == 'combined':
            data_1b1f, n_1b1f = ft.get_data('data/muon_2016_1b1f.csv', 'dimuon_mass', xlimits)
            data_1b1c, n_1b1c = ft.get_data('data/muon_2016_1b1c.csv', 'dimuon_mass', xlimits)
            data = np.concatenate((data_1b1f, data_1b1c))
            n_total = n_1b1f + n_1b1c
        else:
            data, n_total = ft.get_data('data/muon_2016_{0}.csv'.format(channel), 'dimuon_mass', xlimits)

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
                        ('A'     , 0.01 , True , 0.0 , 1.   , None),
                        ('mu'    , -0.43 , True , -0.4 , -0.45  , None),
                        ('sigma' , 0.04 , True , 0.03 , 0.05  , None)
                       )
    sig_params += bg_params.copy()
    sig_model  = Model(ft.sig_pdf, sig_params)
    sig_fitter = NLLFitter(sig_model, fcons=sig_constraint)
    sig_result = sig_fitter.fit(data)

    ### Plots!!! ###
    print 'Making plot of fit results...'
    ft.fit_plot(data, xlimits, sig_model, bg_model, '{0}_{1}'.format(channel, period))

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q_max = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))
    #print 'q = {0}'.format(q_max)
    print 'z = {0}'.format(np.sqrt(q_max))

    ### Calculate the number of events in around the peak
    f_bg    = lambda x: bg_pdf(x, (sig_result.x[3], sig_result.x[4]))
    xlim    = (sig_result.x[1] - 2*sig_result.x[2], sig_result.x[1] + 2*sig_result.x[2])
    N_b     = (1 - sig_result.x[0])*n_total*integrate.quad(f_bg, xlim[0], xlim[1])[0]
    sig_b   = N_b/np.sqrt(n_total*sig_result.x[0])
    N_s     = n_total*sig_result.x[0]
    sig_s   = np.sqrt(N_s)
	#sig_s   = n_total*comb_sigma[0]

    print 'N_b = {0:.2f} +\- {1:.2f}'.format(N_b, sig_b)
    print 'N_s = {0:.2f} +\- {1:.2f}'.format(N_s, sig_s)

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
