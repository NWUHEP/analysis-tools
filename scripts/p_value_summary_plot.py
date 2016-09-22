#!/usr/bin/env python

import sys, pickle
from timeit import default_timer as timer
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm, chi2

from nllfitter import Parameters, ScanParameters, Model, NLLFitter
import nllfitter.fit_tools as ft
import nllfitter.lookee as lee

if __name__ == '__main__':

    start = timer()

    ndim       = 1
    minalgo    = 'SLSQP'
    xlimits    = (12., 70.)
    model      = 'Gaussian'
    #channels   = ['1b1f_2016', '1b1c_2016', '1b1f_2012', '1b1c_2012', '2016', '2012', 'all']      
    #channels   = ['2016', '2012', 'all']      
    channels   = ['1b1f_2012', '1b1c_2012', 'combined_2012']      

    ### Get the data
    data = {}
    #data['1b1f_2012'], _ = ft.get_data('data/muon_2012_1b1f.csv', 'dimuon_mass', xlimits)
    #data['1b1c_2012'], _ = ft.get_data('data/muon_2012_1b1c.csv', 'dimuon_mass', xlimits)
    data['1b1f_2012'], _  = ft.get_data('data/events_pf_1b1f.csv', 'dimuon_mass', xlimits)
    data['1b1c_2012'], _  = ft.get_data('data/events_pf_1b1c.csv', 'dimuon_mass', xlimits)
    data['combined_2012'] = np.concatenate((data['1b1f_2012'], data['1b1c_2012']))

    data['1b1f_2016'], _  = ft.get_data('data/muon_2016_1b1f.csv', 'dimuon_mass', xlimits)
    data['1b1c_2016'], _  = ft.get_data('data/muon_2016_1b1c.csv', 'dimuon_mass', xlimits)
    data['combined_2016'] = np.concatenate((data['1b1f_2016'], data['1b1c_2016']))

    data['all']          = np.concatenate((data['combined_2012'], data['combined_2016']))

    ### Initialize fit models
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
    sig_model  = Model(ft.sig_pdf_alt, sig_params)
    sig_fitter = NLLFitter(sig_model, verbose=False)#, fcons=sig_constraint)

    if ndim == 1:
        print 'Preparing scan parameters...'
        scan_params = ScanParameters(names = ['mu', 'sigma'],
                                     bounds = [(-0.9, 0.9), (0.03,0.06)],
                                     nscans = [100, 1]
                                    )

        mu = np.linspace(17.8, 64.2, 100)
        pscan = {}
        ymin = 1
        for channel in channels:
            print 'Scanning channel {0}'.format(channel)

            # fit background model
            bg_result = bg_fitter.fit(data[channel])
            nll_bg = bg_model.calc_nll(data[channel])

            # scan over signal parameters
            nllscan, params, dof = sig_fitter.scan(scan_params, data[channel]) 
            qscan = -2*(nllscan - nll_bg)
            pval = 0.5*chi2.sf(qscan, 1)
            pscan[channel] = pval
            
            pmin = np.min(pval)
            if pmin < ymin:
                ymin = pmin

            print '{0} p_local = {1:.3e}'.format(channel, pmin)
            plt.plot(mu, pval, linewidth=2.)
        
        # Draw significance lines
        ones = np.ones(mu.size)
        plt.plot(mu, norm.sf(1)*ones, 'k--', linewidth=2.)
        for i in xrange(2, 7):
            plt.plot(mu, norm.sf(i)*ones, 'k--', linewidth=2.)
            plt.text(60, norm.sf(i)*1.25, r'${0} \sigma$'.format(i), color='red', fontsize=16)

        plt.yscale('log')
        plt.title(r'')
        plt.ylim([0.5*ymin, 1.])
        plt.xlim([mu[0], mu[-1]])
        plt.xlabel(r'$m_{\mu\mu}$ [GeV]')
        plt.ylabel(r'$p_{local}$')
        plt.legend([ch.split('_')[0] for ch in channels], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
        plt.savefig('plots/pvalue_scans_{0}_1D.pdf'.format(model))
        plt.savefig('plots/pvalue_scans_{0}_1D.png'.format(model))
        plt.close()

    elif ndim == 2:
        scan_params = ScanParameters(names  = ['mu', 'sigma'],
                                     bounds = [(-0.9, 0.9), (0.015,0.15)],
                                     nscans = [100, 100]
                                    )

        for channel in channels:
            print 'Scanning channel {0}'.format(channel)

            # fit background model
            bg_result = bg_fitter.fit(data[channel])
            nll_bg = bg_model.calc_nll(data[channel])

            # scan over signal parameters
            nllscan, params, dof = sig_fitter.scan(scan_params, data[channel]) 
            qscan = -2*(nllscan - nll_bg)
            x = np.linspace(14.9, 67.1, 100)
            y = np.linspace(0.435, 4.35, 100)
            ft.plot_pvalue_scan_2D(qscan, x, y, path='plots/pvalue_scans_{0}_{1}_2D.pdf'.format(channel, model))
            ft.plot_pvalue_scan_2D(qscan, x, y, path='plots/pvalue_scans_{0}_{1}_2D.png'.format(channel, model))


    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
