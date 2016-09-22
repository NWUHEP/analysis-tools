#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PlotMaker():
    def __init__(self, input_file, datasets):
        self._input_file = input_file
        self._datasets   = datasets
        self._dataframes = get_dataframes()

    def get_dataframes():
        '''
        Initialization for getting dataframes that will be used for plotting.
        '''
        pass

if __name__ == '__main__':

    ### Configuration
    ntuple_dir = 'data/flatuples/mumu_2012'
    lumi       = 19.7e3
    selection  = 'mumu'
    datasets   = ['muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D', 
                  'ttbar_lep', 'zjets_m-50']

    features       = ['muon1_pt', 'muon2_pt', 'dimuon_mass']
    stack_labels   = ['ttbar', 'zjets']
    overlay_labels = ['data']
    
    ### Get dataframes with styles, cross-sections, event_counts, etc.
    lut_datasets = pd.read_csv('data/plotting_lut_datasets.csv', index_col='dataset_name')
    lut_features = pd.read_csv('data/plotting_lut_variables.csv', index_col='variable_name')

    ### Get dataframes with features for each of the datasets ###
    dataframes = {}
    for dataset in datasets:
        df        = pd.read_csv('{0}/ntuple_{1}.csv'.format(ntuple_dir, dataset))
        lut_entry = lut_datasets.loc[dataset]
        label     = lut_entry.label

        if label != 'data':
            scale = lumi*lut_entry.cross_section*lut_entry.branching_fraction/1e7
            df.weight = df.weight.multiply(scale)

        if label not in dataframes.keys():
            dataframes[label] = df
        else:
            dataframes[label] = dataframes[label].append(df)

    ### Loop over features and make the plots ###
    for feature in features:
        lut_entry = lut_features.loc[feature]

        ### Make stacks ###
        x_stack       = [dataframes[label][feature] for label in stack_labels]
        event_weights = [dataframes[label]['weight'] for label in stack_labels]
        plt.hist(x_stack, 
                 bins=lut_entry.n_bins, range=(lut_entry.xmin, lut_entry.xmax), 
                 stacked=True, histtype='stepfilled', alpha=0.9,
                 weights=event_weights)

        ### Make overlays ###
        x_overlay     = [dataframes[label][feature] for label in overlay_labels]
        event_weights = [dataframes[label]['weight'] for label in overlay_labels]
        plt.hist(x_overlay,
                 bins=lut_entry.n_bins, range=(lut_entry.xmin, lut_entry.xmax), 
                 stacked=True, histtype='step', alpha=0.9,
                 linewidth=2., 
                 weights=event_weights)

        ### maker 'er pretty ###
        plt.xlabel(r'{0}'.format(lut_entry.x_label))
        plt.ylabel(r'{0}'.format(lut_entry.y_label))
        plt.xlim((lut_entry.xmin, lut_entry.xmax))

        ### Save output plot ###
        plt.savefig('plots/test/{0}.png'.format(feature))
        plt.close()

