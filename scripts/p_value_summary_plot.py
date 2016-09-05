#!/usr/bin/env python

import sys, pickle
from timeit import default_timer as timer
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

from nllfitter import Parameters, ScanParameters, Model, NLLFitter
import nllfitter.fit_tools as ft
import nllfitter.lookee as lee

if __name__ == '__main__':

    start = timer()

    ndim       = 1
    minalgo    = 'SLSQP'
    xlimits    = (12., 70.)
    #channels   = ['1b1f_2016', '1b1c_2016', '1b1f_2012', '1b1c_2012', '2016', '2012', 'all']      
    channels   = ['2016', '2012', 'all']      

    data = {}
    data['1b1f_2016'], _ = ft.get_data('data/muon_2016_1b1f.csv', 'dimuon_mass', xlimits)
    data['1b1c_2016'], _ = ft.get_data('data/muon_2016_1b1c.csv', 'dimuon_mass', xlimits)
    data['1b1f_2012'], _ = ft.get_data('data/muon_2012_1b1f.csv', 'dimuon_mass', xlimits)
    data['1b1c_2012'], _ = ft.get_data('data/muon_2012_1b1c.csv', 'dimuon_mass', xlimits)
    data['2016']         = np.concatenate((data['1b1f_2016'], data['1b1c_2016']))
    data['2012']         = np.concatenate((data['1b1f_2012'], data['1b1c_2012']))
    data['all']          = np.concatenate((data['2012'], data['2016']))

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None)
                      )

    bg_model  = Model(ft.bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model, verbose=False)

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    sig_params.add_many(
                        ('A'     , 0.01  , True , 0. , 0.2   , None),
                        ('mu'    , -0.5 , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.03 , True , 0.02 , 1.   , None)
                       )
    sig_params += bg_params.copy()

    sig_model  = Model(ft.sig_pdf, sig_params)
    sig_fitter = NLLFitter(sig_model, verbose=False)#, fcons=sig_constraint)

    #######################################################
    ### Scan over search dimensions/nuisance parameters ###
    #######################################################

    ### Define scan values here ### 
    print 'Preparing scan parameters...'
    if ndim == 1:
        scan_params = ScanParameters(names = ['mu', 'sigma'],
                                     bounds = [(-0.9, 0.9), (0.03,0.06)],
                                     nscans = [200, 1]
                                    )
    elif ndim == 2:
        scan_params = ScanParameters(names = ['mu', 'sigma'],
                                     bounds = [(-0.9, 0.9), (0.015,0.15)],
                                     nscans = [200, 200]
                                    )

    mu = np.linspace(17.8, 64.2, 200)
    pscan = {}
    for channel in channels:
        # fit background model
        bg_result = bg_fitter.fit(data[channel])
        nll_bg = bg_model.calc_nll(data[channel])

        # scan over signal parameters
        nllscan, params, dof = sig_fitter.scan(scan_params, data[channel]) 
        qscan = -2*(nllscan - nll_bg)
        p_val = 0.5*chi2.sf(qscan, 1)
        pscan[channel] = p_val

        print '{0} p_local = {1:.3e}'.format(channel, np.min(p_val))
        plt.plot(mu, p_val)
    
    # Draw significance lines
    ones = np.ones(mu.size)
    plt.plot(mu, norm.sf(1)*ones, 'r--')
    plt.plot(mu, norm.sf(2)*ones, 'r--')
    plt.text(60, norm.sf(2)*1.25, r'$2 \sigma$', color='red')
    plt.plot(mu, norm.sf(3)*ones, 'r--')
    plt.text(60, norm.sf(3)*1.25, r'$3 \sigma$', color='red')
    plt.plot(mu, norm.sf(4)*ones, 'r--')
    plt.text(60, norm.sf(4)*1.25, r'$4 \sigma$', color='red')
    plt.plot(mu, norm.sf(5)*ones, 'r--')
    plt.text(60, norm.sf(5)*1.25, r'$5 \sigma$', color='red')
    plt.plot(mu, norm.sf(6)*ones, 'r--')
    plt.text(60, norm.sf(6)*1.25, r'$6 \sigma$', color='red')

    plt.yscale('log')
    plt.title(r'')
    #plt.ylim([0., 1.5*np.max(h[0])])
    plt.xlim([mu[0], mu[-1]])
    plt.xlabel(r'$m_{\mu\mu}$ [GeV]')
    plt.ylabel(r'$p_{local}$')
    plt.savefig('plots/pvalue_scans.pdf')
    plt.savefig('plots/pvalue_scans.png')
    plt.close()

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
