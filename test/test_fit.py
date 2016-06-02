#!/usr/bin/env python

import pickle
from scipy.stats import norm
import nllfitter.future_fitter as ff
from nllfitter.fitter import scale_data, fit_plot

if __name__ == '__main__':

    pfile = open('data/hgg_bg.p', 'r')
    data  = pickle.load(pfile)
    data  = data.query('100 <= Mgg <= 180')
    data  = scale_data(data['Mgg'].values, xlow=100, xhigh=180)

    xlimits  = (100., 180.)
    sdict    = {'mu': lambda x: scale_data(x, xlow=100, xhigh=180, invert = True),
                'sigma': lambda x: x*(xlimits[1] - xlimits[0])/2.,
               }


    ### Define bg model and carry out fit ###
    bg_pdf  = lambda x, a:  0.5 + a[0]*x + 0.5*a[1]*(3*x**2 - 1) + 0.5*a[2]*(5*x**3 - 3*x)
    bg_model = ff.Model(bg_pdf, ['a1', 'a2', 'a3'])
    bg_model.set_bounds([(-1., 1.), (-1., 1.), (-1., 1.)])
    bg_fitter = ff.NLLFitter(bg_model, data)
    bg_result = bg_fitter.fit([0.0, 0.0, 0.0])

    ### Define bg+sig model and carry out fit ###
    sig_pdf = lambda x, a: (1 - a[0])*bg_pdf(x, a[3:6]) + a[0]*norm.pdf(x, a[1], a[2])
    sig_model = ff.Model(sig_pdf, ['A', 'mu', 'sigma', 'a1', 'a2', 'a3'])
    sig_model.set_bounds([(0., .5), 
                          (-0.8, -0.2), (0., 0.2),
                          (-1., 1.), (-1., 1.), (-1., 1.)])
    sig_fitter = ff.NLLFitter(sig_model, data, scaledict=sdict)
    sig_result = sig_fitter.fit((0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1], bg_result.x[2]))
    #sig_result.x = (0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1])

    ### Plots!!! ###
    print 'Making plot of fit results.'
    fit_plot(scale_data(data, xlow=xlimits[0], xhigh=xlimits[1], invert=True), xlimits, sig_pdf, sig_result.x, bg_pdf, bg_result.x, 'test')
    #fit_plot(scale_data(data, xlow=xlimits[0], xhigh=xlimits[1], invert=True), xlimits, None, None, bg_pdf, bg_result.x, 'test')
