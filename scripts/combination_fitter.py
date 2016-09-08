#!/usr/bin/env python

from timeit import default_timer as timer
from collections import OrderedDict

import pandas as pd
import numpy as np

import nllfitter.nllfitter as nll
import nllfitter.fit_tools as ft


if __name__ == '__main__':

    ### Start the timer
    start = timer()
    verbose = True

    ### get data and convert variables to be on the range [-1, 1]
    minalgo  = 'SLSQP'
    channels = ['1b1f', '1b1c']
    xlimits  = (12., 70.)
    sdict    = {'mu': lambda x: scale_data(x, invert = True),
                'sigma': lambda x: x*(xlimits[1] - xlimits[0])/2.,
               }

    ### Fits for individual channels
    datas      = OrderedDict()
    bg_models  = OrderedDict() 
    sig_models = OrderedDict() 
    for channel in channels:

        print 'Getting data for {0} channel and scaling to lie in range [-1, 1].'.format(channel)
        data, n_total  = ft.get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
        datas[channel] = data
        print 'Analyzing {0} events...'.format(n_total)

        ### Define bg model and carry out fit ###
        bg_model = Model(bg_pdf, ['a1', 'a2'])
        bg_model.set_bounds([(0., 1.), (0., 1.)])
        bg_models[channel] = bg_model
        bg_fitter = NLLFitter(bg_model, data)
        bg_result = bg_fitter.fit([0.5, 0.05])

        ### Define bg+sig model and carry out fit ###
        sig_model = Model(sig_pdf, ['A', 'mu', 'sigma', 'a1', 'a2'])
        sig_model.set_bounds([(0., .5), 
                              (-0.8, -0.2), (0., 0.5),
                              (0., 1.), (0., 1.)])
        sig_models[channel] = sig_model
        sig_fitter = NLLFitter(sig_model, data, scaledict=sdict)
        sig_result = sig_fitter.fit((0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1]))

        q = calculate_likelihood_ratio(bg_model, sig_model, data)
        print '{0}: q = {1:.2f}'.format(channel, q)
        ### Plots!!! ###
        print 'Making plot of fit results.'
        fit_plot(scale_data(data, invert=True), xlimits, sig_pdf, sig_result.x, bg_pdf, bg_result.x, channel)

    
    ### Prepare data for combined fit
    ### Parameter naming is important.  If a parameter name is the same between
    ### multiple models it will be fixed between each model. 

    bg_models['1b1f'].parnames  = ['a1', 'a2']
    bg_models['1b1c'].parnames  = ['b1', 'b2']
    combined_bg_model   = CombinedModel([bg_models[ch] for ch in channels])

    sig_models['1b1f'].parnames = ['A1', 'mu', 'sigma', 'a1', 'a2']
    sig_models['1b1c'].parnames = ['A2', 'mu', 'sigma', 'b1', 'b2']
    combined_sig_model = CombinedModel([sig_models[ch] for ch in channels])

    ### Perform combined bg fit
    combination_bg_fitter = NLLFitter(combined_bg_model, [datas[ch] for ch in channels])
    param_init = combined_bg_model.get_params().values()
    bg_result = combination_bg_fitter.fit(param_init)
    
    ### Perform combined signal+bg fit
    combination_sig_fitter = NLLFitter(combined_sig_model, [datas[ch] for ch in channels], scaledict=sdict)
    param_init = combined_sig_model.get_params().values()
    sig_result = combination_sig_fitter.fit(param_init)

    q = calculate_likelihood_ratio(combined_bg_model, combined_sig_model, datas.values())
    print 'combined: q = {0:.2f}'.format(q)
    ### Plot results.  Overlay signal+bg fit, bg-only fit, and data
    for ch in channels:
        fit_plot(scale_data(datas[ch], invert=True), xlimits,
                            sig_pdf, sig_models[ch].params,    
                            bg_pdf, bg_models[ch].params, '{0}_combined'.format(ch))
    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
