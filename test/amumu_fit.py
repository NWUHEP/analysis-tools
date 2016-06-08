from __future__ import division

from timeit import default_timer as timer
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from nllfitter.fit_tools import get_data, fit_plot, get_corr, scale_data
from lmfit import Parameter, Parameters, Minimizer, report_fit
from nllfitter.future_fitter import Model, nll

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


if __name__ == '__main__':

    ### Start the timer
    start = timer()
    verbose = True

    ### get data and convert variables to be on the range [-1, 1]
    xlimits = (12., 70.)
    channel = '1b1f'

    print 'Getting data and scaling to lie in range [-1, 1].'
    data, n_total  = get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    print 'Analyzing {0} events...\n'.format(n_total)

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None)
                       )

    bg_model  = Model(bg_pdf, bg_params)
    bg_fitter = Minimizer(nll, bg_params, fcn_args=(data, bg_pdf))
    bg_result = bg_fitter.minimize('SLSQP')
    bg_params = bg_result.params
    sigma, corr = get_corr(partial(nll, data=data, pdf=bg_pdf), 
                           [p.value for p in bg_params.values()]) 
    bg_model.update_parameters(bg_params, (sigma, corr))
    report_fit(bg_params, show_correl=False)
    print ''
    print '[[Correlation matrix]]\n'
    print corr, '\n'

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    sig_params.add_many(
                        ('A'     , 0.05 , True , 0.01 , 0.5  , None),
                        ('mu'    , 0.   , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.02 , True , 0.01 , 0.15 , None),
                        ('a1'    , 0.   , True , 0.01 , 0.5  , None),
                        ('a2'    , 0.   , True , 0.01 , 0.2  , None)
                       )

    sig_model  = Model(sig_pdf, sig_params)
    sig_fitter = Minimizer(nll, sig_params, fcn_args=(data, sig_pdf))
    sig_result = sig_fitter.minimize('SLSQP')
    sig_params = sig_result.params
    sigma, corr = get_corr(partial(nll, data=data, pdf=sig_pdf), 
                           [p.value for p in sig_params.values()]) 
    sig_model.update_parameters(sig_params, (sigma, corr))
    report_fit(sig_model.get_parameters(), show_correl=False)
    print ''
    print '[[Correlation matrix]]\n'
    print corr, '\n'

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q = 2*(bg_model.nll(data) - sig_model.nll(data))
    print '{0}: q = {1:.2f}'.format('h->gg', q)

    ### Plots!!! ###
    print 'Making plot of fit results...'
    fit_plot(scale_data(data, xmin=12, xmax=70, invert=True), xlimits,
             sig_pdf, sig_model.get_parameters(by_val=True),    
             bg_pdf, bg_model.get_parameters(by_val=True), '{0}'.format(channel))
    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))

