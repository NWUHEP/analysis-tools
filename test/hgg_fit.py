from __future__ import division

from timeit import default_timer as timer

import numpy as np
from scipy.stats import norm 

from nllfitter import NLLFitter, Model
from nllfitter.fit_tools import get_data, fit_plot, scale_data
from lmfit import Parameters

# global options
np.set_printoptions(precision=3.)

### PDF definitions ###
def bg_pdf(x, a): 
    '''
    Third order Legendre Polynomial with constant term set to 0.5.

    Parameters:
    ===========
    x: data
    a: model parameters (a1 and a2)
    '''
    return 0.5 + a[0]*x + 0.5*a[1]*(3*x**2 - 1) + 0.5*a[2]*(5*x**3 - 3*x)

def sig_pdf(x, a):
    '''
    Second order Legendre Polynomial (normalized to unity) plus a Gaussian.

    Parameters:
    ===========
    x: data
    a: model parameters (a1, a2, mu, and sigma)
    '''
    return (1 - a[0])*bg_pdf(x, a[3:6]) + a[0]*norm.pdf(x, a[1], a[2])

if __name__ == '__main__':

    ### Start the timer
    start = timer()
    verbose = True

    ### get data and convert variables to be on the range [-1, 1]
    xlimits  = (100., 180.)

    print 'Getting data and scaling to lie in range [-1, 1].'
    data, n_total  = get_data('data/toy_hgammagamma.txt', 'diphoton_mass', xlimits)
    print 'Analyzing {0} events...\n'.format(n_total)

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None),
                       ('a3', 0., True, None, None, None)
                      )

    bg_model  = Model(bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model)
    bg_result = bg_fitter.fit(data)

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    sig_params.add_many(
                        ('A'     , 0.01 , True , 0.   , 1.   , None),
                        ('mu'    , -0.3 , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.01 , True , 0.01 , 1.   , None),
                       )
    sig_params += bg_params.copy()
    sig_model  = Model(sig_pdf, sig_params)
    sig_fitter = NLLFitter(sig_model)
    sig_result = sig_fitter.fit(data)

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))
    print '{0}: q = {1:.2f}'.format('h->gg', q)

    ### Plots!!! ###
    print 'Making plot of fit results...'
    fit_plot(data, xlimits, sig_model, bg_model, 'hgg')
    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
