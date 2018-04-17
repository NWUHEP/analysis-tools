#!/usr/bin/env

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import ROOT as r

import matplotlib
matplotlib.style.use('default')

if __name__ == '__main__':

    mc_bins = np.arange(1, 101, 1)
    mc_file     = r.TFile('data/bltuples/output_single_lepton.root')
    mc_tree     = mc_file.Get('mu4j/bltTree_ttbar_inclusive')
    h_pileup_mc = r.TH1F('h_pileup_mc', '', 100, 0., 100.)
    mc_tree.Draw('nPU>>h_pileup_mc', '', 'goff') 
    mc_pileup = np.array([h_pileup_mc.GetBinContent(int(i))/h_pileup_mc.GetSum() for i in mc_bins])/10
    mc_file.Close()

    # get spline of mc sample
    mc_spline = interp1d(mc_bins, mc_pileup, kind='linear')
    spline_bins = np.arange(1, 100., 0.1)

    # get data pileup
    nbins       = 990
    ibin        = np.arange(1, nbins+1)
    labels      = ['nominal', 'up', 'down']
    data_pileup = {}
    for l in labels:
        data_file = r.TFile(f'data/pileup_{l}.root')
        h_pileup_data = data_file.Get('pileup')
        data_pileup[l] = np.array([h_pileup_data.GetBinContent(int(i))/h_pileup_data.GetSum() for i in ibin])
        data_file.Close()

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=False, gridspec_kw={'height_ratios':[1,2]})
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    pubin = ibin/10
    ax = axes[0]
    mc_pileup = mc_spline(spline_bins)
    ax.plot(spline_bins, mc_spline(spline_bins), c='k')
    #ax.plot(mc_bins, mc_pileup, c='k')
    ax.plot(pubin, data_pileup['nominal'], c='C1')
    ax.fill_between(pubin, data_pileup['up'], data_pileup['down'], color='C1', alpha=0.5, label='_nolegend_')
    ax.plot(pubin, data_pileup['up'], c='C0', linestyle='--')
    ax.plot(pubin, data_pileup['down'], c='C2', linestyle='--')

    ax.set_xlim(0, 59.9)
    ax.set_ylim(0., 1.25*mc_pileup.max())
    ax.set_ylabel(r'$\sf P(n_{pu})$', fontsize=16)
    ax.legend(['simulation', 'nominal', r'$+\sigma$', r'$-\sigma$'], fontsize=14)
    ax.grid()

    ax = axes[1]
    sf_nominal = data_pileup['nominal']/mc_pileup
    sf_up = data_pileup['up']/mc_pileup
    sf_down = data_pileup['down']/mc_pileup
    ax.plot(pubin, sf_nominal, c='C1')
    ax.fill_between(pubin, sf_up, sf_down, color='C1', alpha=0.5,)
    ax.plot(pubin, sf_up, c='C0', linestyle='--')
    ax.plot(pubin, sf_down, c='C2', linestyle='--')
    ax.set_xlim(0, 59.9)
    ax.set_ylim(0, 1.95)
    ax.set_xlabel(r'$\sf n_{pu}$', fontsize=16)
    ax.set_ylabel(r'$\sf w_{pu}$', fontsize=16)
    ax.grid()

    outfile = open('data/pileup_sf.pkl', 'wb')
    pickle.dump(pubin, outfile)
    pickle.dump(sf_nominal, outfile)
    pickle.dump(sf_up, outfile)
    pickle.dump(sf_down, outfile)
    outfile.close()

    #plt.tight_layout()
    plt.savefig('plots/pileup_systematics.png')
    plt.savefig('plots/pileup_systematics.pdf')
    plt.close()
