#!/usr/bin/env python

import os, sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def make_directory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)

    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')

def set_new_tdr():
    plt.style.use('classic')
    plt.rcParams['font.family']       = 'serif'
    plt.rcParams['font.size']         = 18
    plt.rcParams['font.serif']        = 'Ubuntu'
    plt.rcParams['font.monospace']    = 'Ubuntu Mono'
    plt.rcParams['mathtext.fontset']  = 'custom'
    plt.rcParams['mathtext.sf']       = 'Ubuntu'

    plt.rcParams['axes.labelsize']    = 20
    plt.rcParams['xtick.labelsize']   = 18
    plt.rcParams['ytick.labelsize']   = 18
    plt.rcParams['figure.titlesize']  = 20
    plt.rcParams['figure.figsize']    = [11, 8]
    plt.rcParams['legend.fontsize']   = 20
    plt.rcParams['legend.numpoints']  = 1

def add_lumi_text(ax):
    ax.text(0.06, 0.9, r'$\bf CMS$', fontsize=30, transform=ax.transAxes)
    ax.text(0.17, 0.9, r'$\it Preliminary $', fontsize=20, transform=ax.transAxes)
    ax.text(0.68, 1.01, r'$\sf{19.7\,fb^{-1}}\,(\sqrt{\it{s}}=8\,\sf{TeV})$', fontsize=20, transform=ax.transAxes)

def hist_to_errorbar(data, nbins, xlim, normed=False):
    y, bins = np.histogram(data, bins=nbins, range=xlim)
    x = (bins[1:] + bins[:-1])/2.
    yerr = np.sqrt(y) 

    return y, x, yerr

def get_data_and_weights(dataframes, feature, labels, condition):
    data    = []
    weights = []
    for label in labels:
        if condition == 'None':
            df = dataframes[label]
        else:
            df = dataframes[label].query(condition)
        data.append(df[feature].values)
        weights.append(df['weight'].values)

    return data, weights

class DataManager():
    def __init__(self, input_dir, dataset_names, selection, scale=1, cuts=''):
        self._input_dir     = input_dir
        self._dataset_names = dataset_names
        self._selection     = selection
        self._scale         = scale
        self._cuts          = cuts
        self._load_luts()
        self._load_dataframes()

    def _load_luts(self):
        '''
        Retrieve look-up tables for datasets and variables
        '''
        self._event_counts = pd.read_csv('{0}/event_counts.csv'.format(self._input_dir, self._selection))
        self._lut_datasets = pd.read_excel('data/plotting_lut.xlsx', 
                                          sheetname='datasets', 
                                          index_col='dataset_name'
                                         ).dropna(how='all')
        lut_features_default = pd.read_excel('data/plotting_lut.xlsx', 
                                              sheetname='variables',
                                              index_col='variable_name'
                                              ).dropna(how='all')
        lut_features_select = pd.read_excel('data/plotting_lut.xlsx', 
                                              sheetname='variables_{0}'.format(self._selection),
                                              index_col='variable_name'
                                              ).dropna(how='all')
        self._lut_features = pd.concat([lut_features_default, lut_features_select])

    def _load_dataframes(self):
        ''' 
        Get dataframes from input directory.  This method is only for execution
        while initializing the class instance.
        '''
        dataframes = {}
        for dataset in tqdm(self._dataset_names, desc='Loading dataframes', unit_scale=True, ncols=75, total=len(self._dataset_names)):
            df         = pd.read_csv('{0}/ntuple_{1}.csv'.format(self._input_dir, dataset))
            init_count = self._event_counts[dataset][0]
            lut_entry  = self._lut_datasets.loc[dataset]
            label      = lut_entry.label

            ### apply selection cuts ###
            if self._cuts != '':
                df = df.query(self._cuts)
            
            ### update weights with lumi scale factors ###
            if label != 'data':
                scale = self._scale*lut_entry.cross_section*lut_entry.branching_fraction/init_count
                df.weight = df.weight.multiply(scale)

            ### combined datasets if required ###
            if label not in dataframes.keys():
                dataframes[label] = df
            else:
                dataframes[label] = dataframes[label].append(df)

        self._dataframes =  dataframes

    def get_dataframe(self, dataset_name, condition=''):
        df = self._dataframes[dataset_name]
        if condition != '':
            return df.query(condition)
        else:
            return df

    def get_dataset_names(self):
        return self._dataset_names

class PlotManager():
    def __init__(self, data_manager, features, stack_labels, overlay_labels, plot_data=True):
        self._dm             = data_manager
        self._features       = features
        self._stack_labels   = stack_labels
        self._overlay_labels = overlay_labels
        self._plot_data      = plot_data
        self._initialize_colors()

    def _initialize_colors(self):
        lut = self._dm._lut_datasets
        self._stack_colors   = [lut.loc[l].color for l in self._stack_labels]
        self._overlay_colors = [lut.loc[l].color for l in self._overlay_labels]

    def make_overlays(self, features, output_path='plots', file_ext='png', do_cms_text=True, do_ratio=False):
        dm = self._dm
        make_directory(output_path)

        ### alias dataframes and datasets lut###
        dataframes   = dm._dataframes
        lut_datasets = dm._lut_datasets

        ### initialize legend text ###
        legend_text = []
        legend_text.extend([lut_datasets.loc[label].text for label in self._stack_labels[::-1]])
        legend_text.extend([lut_datasets.loc[label].text for label in self._overlay_labels[::-1]])

        if len(self._stack_labels) > 0:
            legend_text.append('BG error')
        if self._plot_data:
            legend_text.append('Data')

        for feature in tqdm(features, desc='Plotting', unit_scale=True, ncols=75, total=len(features)):
            if feature not in self._features:
                print '{0} not in features.'
                continue
            #else:
            #    print feature

            ### Get style data for the feature ###
            lut_entry = dm._lut_features.loc[feature]

            ### initialize figure ###
            fig, axes      = plt.subplots(1, 1)
            #legend_handles = []

            ### Get stack data and apply mask if necessary ###
            stack_max = 0
            if len(self._stack_labels) > 0:
                stack_data, stack_weights = get_data_and_weights(dataframes, feature, self._stack_labels, lut_entry.condition)
                stack, bins, p = axes.hist(stack_data, 
                                           bins      = lut_entry.n_bins,
                                           range     = (lut_entry.xmin, lut_entry.xmax),
                                           color     = self._stack_colors,
                                           alpha     = 1.,
                                           linewidth = 0.5,
                                           stacked   = True,
                                           histtype  = 'stepfilled',
                                           weights   = stack_weights
                                          )
                #legend_handles.append(p)

                ### Need to histogram the stack with the square of the weights to get the errors ### 
                stack_noscale = np.histogram(np.concatenate(stack_data), 
                                             bins  = lut_entry.n_bins,
                                             range = (lut_entry.xmin, lut_entry.xmax),
                                             weights = np.concatenate(stack_weights)**2
                                            )[0] 
                stack_sum = stack[-1]
                stack_x   = (bins[1:] + bins[:-1])/2.
                stack_err = np.sqrt(stack_noscale)
                no_blanks = stack_sum > 0
                stack_sum, stack_x, stack_err = stack_sum[no_blanks], stack_x[no_blanks], stack_err[no_blanks]
                eb = axes.errorbar(stack_x, stack_sum, yerr=stack_err, 
                              fmt        = 'none',
                              ecolor     = 'k',
                              capsize    = 0,
                              elinewidth = 10,
                              alpha      = 0.15
                             )
                stack_max = np.max(stack_sum)
                #legend_handles.append(eb[0])

            ### Get overlay data and apply mask if necessary ###
            overlay_max = 0
            if len(self._overlay_labels) > 0:
                overlay_data, overlay_weights = get_data_and_weights(dataframes, feature, self._overlay_labels, lut_entry.condition)
                hists, bins, p = axes.hist(overlay_data,
                                           bins      = lut_entry.n_bins,
                                           range     = (lut_entry.xmin, lut_entry.xmax),
                                           color     = self._overlay_colors,
                                           alpha     = 0.9,
                                           histtype  = 'step',
                                           linewidth = 2.,
                                           weights   = overlay_weights
                                          )
                overlay_max = np.max(hists.flatten())
                #legend_handles.append(p)

            ### If there's data to overlay: apply feature condition and get
            ### datapoints plus errors
            data_max = 0
            if self._plot_data:
                data, _ = get_data_and_weights(dataframes, feature, ['data'], lut_entry.condition)
                y, x, yerr = hist_to_errorbar(data, 
                                              nbins = lut_entry.n_bins,
                                              xlim  = (lut_entry.xmin, lut_entry.xmax)
                                             )
                y, x, yerr = y[y>0], x[y>0], yerr[y>0]
                eb = axes.errorbar(x, y, yerr=yerr, 
                              fmt        = 'ko',
                              capsize    = 0,
                              elinewidth = 2
                             )
                data_max = np.max(y)
                #legend_handles.append(eb[0])

            ### make the legend ###
            axes.legend(legend_text)

            ### labels and x limits ###
            axes.set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))
            axes.set_ylabel(r'$\sf {0}$'.format(lut_entry.y_label))
            axes.set_xlim((lut_entry.xmin, lut_entry.xmax))
            axes.grid()

            ### Add lumi text ###
            if do_cms_text:
                add_lumi_text(axes)

            ### Make output directory if it does not exist ###
            make_directory('{0}/linear/{1}'.format(output_path, lut_entry.category), False)
            make_directory('{0}/log/{1}'.format(output_path, lut_entry.category), False)

            ### Save output plot ###
            ### linear scale ###
            y_max = np.max((stack_max, overlay_max, data_max))
            axes.set_ylim((0., 1.8*y_max))
            fig.savefig('{0}/linear/{1}/{2}.{3}'.format(output_path, lut_entry.category, feature, file_ext))

            ### log scale ###
            axes.set_yscale('log')
            axes.set_ylim((0.1*np.min(stack_sum), 15.*y_max))
            fig.savefig('{0}/log/{1}/{2}.{3}'.format(output_path, lut_entry.category, feature, file_ext))

            fig.clear()
            plt.close()

