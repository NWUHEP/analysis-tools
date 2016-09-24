#!/usr/bin/env python

import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('classic')
plt.rcParams['font.family']       = 'serif'
plt.rcParams['font.serif']        = 'Ubuntu'
plt.rcParams['font.monospace']    = 'Ubuntu Mono'
plt.rcParams['mathtext.fontset']  = 'custom'
plt.rcParams['mathtext.sf']       = 'Ubuntu'
plt.rcParams['font.size']         = 16
plt.rcParams['axes.labelsize']    = 18
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize']   = 16
plt.rcParams['ytick.labelsize']   = 16
plt.rcParams['legend.fontsize']   = 18
plt.rcParams['figure.titlesize']  = 20
plt.rcParams['figure.figsize']    = [11, 8]
plt.rcParams['legend.numpoints']  = 1


def hist_to_errorbar(data, nbins, xlim, normed=False):
    y, bins = np.histogram(data, bins=nbins, range=xlim)
    x = (bins[1:] + bins[:-1])/2.
    yerr = np.sqrt(y) 

    return y, x, yerr

if __name__ == '__main__':

    ### Start the timer
    start = timer()

    ### Configuration
    ntuple_dir = 'data/flatuples/mumu_2012'
    lumi       = 19.7e3
    selection  = 'mumu'
    period     = 2012
    datasets   = [
                  'muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D', 
                  'ttbar_lep', 
                  'zjets_m-50', 'zjets_m-10to50',
                  't_s', 't_t', 't_tw', 'tbar_s', 'tbar_t', 'tbar_tw', 
                  'ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q', 'zz_2l2nu'
                 ]

    features = [
                 'muon1_pt', 'muon1_eta', 'muon1_phi', 'muon1_iso', 
                 'muon2_pt', 'muon2_eta', 'muon2_phi', 'muon2_iso', 
                 'muon_delta_eta', 'muon_delta_phi', 'muon_delta_r',
                 'dimuon_mass', 'dimuon_pt', 'dimuon_eta', 'dimuon_phi', 
                 'dimuon_pt_over_m',
                 'n_jets', 'n_fwdjets', 'n_bjets',
                 'bjet_pt', 'bjet_eta', 'bjet_phi', 'bjet_d0',
                 'jet_pt', 'jet_eta', 'jet_phi', 'jet_d0', 
                 'dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi', 
                 #'dijet_pt_over_m',
                 'dimuon_b_mass', 'dimuon_b_pt', 'dimuon_b_delta_eta', 'dimuon_b_delta_phi',
                 'four_body_delta_phi', 'four_body_delta_eta', 'four_body_mass',
                 'met_mag', 'met_phi',
               ]
    #selection_cut = 'n_fwdjets > 0 and n_bjets == 1 and n_jets == 0'
    selection_cut = '(n_fwdjets > 0 or (n_jets == 1 and four_body_delta_phi > 2.5)) and n_bjets == 1'

    ### Get dataframes with styles, cross-sections, event_counts, etc.
    lut_datasets = pd.read_excel('data/plotting_lut.xlsx', sheetname='datasets', index_col='dataset_name').dropna()
    lut_features = pd.read_excel('data/plotting_lut.xlsx', sheetname='variables', index_col='variable_name').dropna()
    event_counts = pd.read_csv('{0}/event_counts.csv'.format(ntuple_dir))

    ### Get dataframes with features for each of the datasets ###
    dataframes = {}
    for dataset in datasets:
        df        = pd.read_csv('{0}/ntuple_{1}.csv'.format(ntuple_dir, dataset))
        lut_entry = lut_datasets.loc[dataset]
        label     = lut_entry.label

        ### apply selection cuts ###
        df = df.query(selection_cut)
        
        ### update weights with lumi scale factors ###
        if label != 'data':
            scale = lumi*lut_entry.cross_section*lut_entry.branching_fraction/event_counts[dataset][0]
            df.weight = df.weight.multiply(scale)

        ### combined datasets if desired ###
        if label not in dataframes.keys():
            dataframes[label] = df
        else:
            dataframes[label] = dataframes[label].append(df)

    ### Loop over features and make the plots ###
    stack_labels   = ['t', 'diboson', 'ttbar', 'zjets']
    stack_colors   = [lut_datasets.loc[l].color for l in stack_labels]

    overlay_labels = ['data']
    overlay_colors = [lut_datasets.loc[l].color for l in overlay_labels]
    for feature in features:
        print feature

        lut_entry = lut_features.loc[feature]

        ### initialize figure ###
        fig, axes = plt.subplots(1, 1)

        ### Make stacks ###
        x_stack       = [dataframes[label][feature].values for label in stack_labels]
        event_weights = [dataframes[label]['weight'].values for label in stack_labels]
        stack = axes.hist(x_stack, 
                      bins=lut_entry.n_bins, range=(lut_entry.xmin, lut_entry.xmax), 
                      color=stack_colors, alpha=1., linewidth=0.5,
                      stacked=True, histtype='stepfilled',
                      weights=event_weights)
        stack_sum = np.sum(np.array(stack[0]), axis=0)

        ### Make overlays ###
        x_overlay     = [dataframes[label][feature].values for label in overlay_labels if label != 'data']
        event_weights = [dataframes[label]['weight'].values for label in overlay_labels if label != 'data']
        if len(x_overlay) > 0:
            axes.hist(x_overlay,
                     bins=lut_entry.n_bins, range=(lut_entry.xmin, lut_entry.xmax), 
                     color=overlay_colors, alpha=0.9,
                     stacked=True, histtype='step',
                     linewidth=2., 
                     weights=event_weights)

        ### Make datapoints with errors ###
        if 'data' in overlay_labels:
            y, x, yerr = hist_to_errorbar(dataframes['data'][feature].values, nbins=lut_entry.n_bins, xlim=(lut_entry.xmin, lut_entry.xmax)) 
            y, x, yerr = y[y>0], x[y>0], yerr[y>0]
            axes.errorbar(x, y, yerr=yerr, fmt='ko', capsize=0, elinewidth=2)

        ### make the legend ###
        axes.legend([lut_datasets.loc[label].text for label in stack_labels[::-1]+overlay_labels])

        ### maker 'er pretty ###
        axes.set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))
        axes.set_ylabel(r'$\sf {0}$'.format(lut_entry.y_label))
        axes.set_xlim((lut_entry.xmin, lut_entry.xmax))
        axes.grid()

        ### Save output plot ###
        axes.set_ylim((0., 0.8*np.max(stack_sum)))
        fig.savefig('plots/overlays/{0}_{1}/linear/{2}.png'.format(selection, period, feature))
        fig.savefig('plots/overlays/{0}_{1}/linear/{2}.pdf'.format(selection, period, feature))

        axes.set_yscale('log')
        axes.set_ylim((0.1*np.min(stack_sum), 5.*np.max(stack_sum)))
        fig.savefig('plots/overlays/{0}_{1}/log/{2}.png'.format(selection, period, feature))
        fig.savefig('plots/overlays/{0}_{1}/log/{2}.pdf'.format(selection, period, feature))
        fig.clear()
        plt.close()

