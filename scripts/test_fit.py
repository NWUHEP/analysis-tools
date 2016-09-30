from __future__ import division

import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Parameter, Parameters

from nllfitter import Model, NLLFitter
import nllfitter.fit_tools as ft
import nllfitter.plot_tools as pt

def legendre_polynomial(x, a):
    '''
    Generates Legendre polynomial based on the size of a.  The generation of
    the function is done by making use of Bonnet's recursion formula.

    Parameters:
    ===========
    x : datapoint(s) to evaluate the Legendre polynomial at
    a : model parameters; the order of the polynomial is determined by the size of a
    '''
    order = len(a) - 1
    if order == 0:
        return 1
    elif order == 1:
        return x
    else:
        return ((2*order + 1)*legendre_polynomial(x, a[:-1]) - order*legendre_polynomial(x, a[:-2]))/(order + 1)

def bg_pdf(x, a):
    '''
    Legendre polynomial background pdf
    '''
    pass

if __name__ == '__main__':

    ### Start the timer
    start = timer()

    ### Configuration
    pt.set_new_tdr()
    ntuple_dir  = 'data/flatuples/mumu_2012'
    lumi        = 19.8e3
    selection   = ('mumu', 'combined')
    period      = 2012
    model       = 'Gaussian'
    output_path = 'plots/fits/{0}_{1}'.format('_'.join(selection), period)

    datasets = ['muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D'] 
    features = ['dimuon_mass', 'dimuon_b_mass']
    cuts     = 'lepton1_q != lepton2_q and \
                n_bjets == 1 and \
                ((n_fwdjets > 0 and n_jets == 0) or \
                (n_fwdjets == 0 and n_jets == 1 and four_body_delta_phi > 2.5 and met_mag < 40))'

    ### Get dataframes with features for each of the datasets ###
    data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                  dataset_names = datasets,
                                  selection     = selection[0],
                                  scale         = lumi,
                                  cuts          = cuts
                                 )
    data = data_manager.get_dataframe('data')

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None),
                       ('b1', 0., True, None, None, None),
                       ('b2', 0., True, None, None, None),
                       ('b3', 0., True, None, None, None),
                       ('b4', 0., True, None, None, None),
                       ('b5', 0., True, None, None, None)
                      )

    #bg_model  = Model(bg_pdf, bg_params)
    #bg_fitter = NLLFitter(bg_model)
    #bg_result = bg_fitter.fit(data)

    '''
    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    if model == 'Gaussian':
        sig_params.add_many(
                            ('A'     , 0.01  , True , 0.0   , 1.   , None),
                            ('mu'    , -0.43 , True , -0.44 , -0.42 , None),
                            ('sigma' , 0.04  , True , 0.015 , 0.2  , None)
                           )
        sig_params += bg_params.copy()
        sig_model  = Model(ft.sig_pdf, sig_params)
    elif model == 'Voigt':
        sig_params.add_many(
                            ('A'     , 0.01   , True , 0.0   , 1.    , None),
                            ('mu'    , -0.43  , True , -0.8  , 0.8   , None),
                            ('gamma' , 0.033  , True , 0.01  , 0.1   , None),
                           )
        sig_params += bg_params.copy()
        sig_model  = Model(ft.sig_pdf_alt, sig_params)

    sig_fitter = NLLFitter(sig_model)
    sig_result = sig_fitter.fit(data)

    ### Plots!!! ###
    print 'Making plot of fit results...'
    ft.fit_plot(data, xlimits, sig_model, bg_model, 
                '{0}_{1}'.format(channel, model), path='plots/fits/{0}'.format(period))

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q_max = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))
    p_value = 0.5*chi2.sf(q_max, 1)
    z_score = -norm.ppf(p_value)
    print 'q       = {0:.3f}'.format(q_max)
    print 'p_local = {0:.3e}'.format(p_value)
    print 'z_local = {0}'.format(z_score)

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
    '''

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
