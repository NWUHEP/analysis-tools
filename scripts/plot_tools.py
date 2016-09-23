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
plt.rcParams['mathtext.sf']       = 'Ubuntu'
plt.rcParams['font.size']         = 14
plt.rcParams['axes.labelsize']    = 16
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize']   = 14
plt.rcParams['ytick.labelsize']   = 14
plt.rcParams['legend.fontsize']   = 14
plt.rcParams['figure.titlesize']  = 18


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

    features       = ['muon1_pt', 'muon2_pt', 'dimuon_mass', 'met_mag']
    stack_labels   = ['t', 'diboson', 'ttbar', 'zjets']
    overlay_labels = ['data']
    
    ### Get dataframes with styles, cross-sections, event_counts, etc.
    lut_datasets = pd.read_excel('data/plotting_lut.xlsx', sheetname='datasets', index_col='dataset_name')
    lut_features = pd.read_excel('data/plotting_lut.xlsx', sheetname='variables', index_col='variable_name')
    event_counts = pd.read_csv('{0}/event_counts.csv'.format(ntuple_dir))

    ### Get dataframes with features for each of the datasets ###
    dataframes = {}
    for dataset in datasets:
        df        = pd.read_csv('{0}/ntuple_{1}.csv'.format(ntuple_dir, dataset))
        lut_entry = lut_datasets.loc[dataset]
        label     = lut_entry.label

        if label != 'data':
            scale = lumi*lut_entry.cross_section*lut_entry.branching_fraction/event_counts[dataset][0]
            df.weight = df.weight.multiply(scale)

        if label not in dataframes.keys():
            dataframes[label] = df
        else:
            dataframes[label] = dataframes[label].append(df)

    ### Loop over features and make the plots ###
    stack_colors = [lut_datasets.loc[l].color for l in stack_labels]
    overlay_colors = [lut_datasets.loc[l].color for l in overlay_labels]
    for feature in features:
        lut_entry = lut_features.loc[feature]

        ### initialize figure ###
        fig, axes = plt.subplots(1, 1)

        ### Make stacks ###
        x_stack       = [dataframes[label][feature] for label in stack_labels]
        event_weights = [dataframes[label]['weight'] for label in stack_labels]
        axes.hist(x_stack, 
                 bins=lut_entry.n_bins, range=(lut_entry.xmin, lut_entry.xmax), 
                 color=stack_colors,
                 stacked=True, histtype='stepfilled', alpha=1.,
                 weights=event_weights)

        ### Make overlays ###
        x_overlay     = [dataframes[label][feature] for label in overlay_labels if label != 'data']
        event_weights = [dataframes[label]['weight'] for label in overlay_labels if label != 'data']
        if len(x_overlay) > 0:
            axes.hist(x_overlay,
                     bins=lut_entry.n_bins, range=(lut_entry.xmin, lut_entry.xmax), 
                     color=overlay_colors,
                     stacked=True, histtype='step', alpha=0.9,
                     linewidth=2., 
                     weights=event_weights)

        ### Make datapoints with errors ###
        if 'data' in overlay_labels:
            y, x, yerr = hist_to_errorbar(dataframes['data'][feature], nbins=lut_entry.n_bins, xlim=(lut_entry.xmin, lut_entry.xmax)) 
            axes.errorbar(x, y, yerr=yerr, fmt='ko')

        ### maker 'er pretty ###
        axes.set_xlabel(r'{0}'.format(lut_entry.x_label))
        axes.set_ylabel(r'{0}'.format(lut_entry.y_label))
        axes.set_xlim((lut_entry.xmin, lut_entry.xmax))
        axes.grid()

        ### make the legend ###
        axes.legend([lut_datasets.loc[label].text for label in stack_labels[::-1]+overlay_labels])

        ### Save output plot ###
        fig.savefig('plots/overlays/{0}_{1}/linear/{2}.png'.format(selection, period, feature))
        axes.set_yscale('log')
        fig.savefig('plots/overlays/{0}_{1}/log/{2}.png'.format(selection, period, feature))
        fig.clear()
        plt.close()

