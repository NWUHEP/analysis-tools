#!/usr/bin/env python

import os, sys
from timeit import default_timer as timer
from collections import OrderedDict

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

def add_lumi_text(ax, lumi, period):
    ax.text(0.06, 0.9, r'$\bf CMS$', fontsize=30, transform=ax.transAxes)
    ax.text(0.17, 0.9, r'$\it Preliminary $', fontsize=20, transform=ax.transAxes)
    if period == 2012:
        ax.text(0.68, 1.01, r'$\sf{19.7\,fb^{-1}}\,(\sqrt{\it{s}}=8\,\sf{TeV})$', fontsize=20, transform=ax.transAxes)
    elif period == 2016:
        ax.text(0.68, 1.01, 
                r'$\sf{{ {0:.1f}\,fb^{{-1}}}}\,(\sqrt{{\it{{s}}}}=13\,\sf{{TeV}})$'.format(lumi/1000.), 
                fontsize=20, 
                transform=ax.transAxes)

def hist_to_errorbar(data, nbins, xlim, normed=False):
    y, bins = np.histogram(data, bins=nbins, range=xlim)
    x = (bins[1:] + bins[:-1])/2.
    yerr = np.sqrt(y) 

    return x, y, yerr

def ratio_errors(num, den):
    return np.sqrt(num + num**2/den)/den

def get_data_and_weights(dataframes, feature, labels, condition='None'):
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
    def __init__(self, input_dir, dataset_names, selection, 
                 period  = 2012,
                 scale   = 1,
                 cuts    = '',
                 combine = True
                ):
        self._input_dir     = input_dir
        self._dataset_names = dataset_names
        self._selection     = selection
        self._period        = period
        self._scale         = scale
        self._cuts          = cuts
        self._combine       = combine
        self._load_luts()
        self._load_dataframes()

    def _load_luts(self):
        '''
        Retrieve look-up tables for datasets and variables
        '''
        self._event_counts = pd.read_csv('{0}/event_counts.csv'.format(self._input_dir, self._selection))
        self._lut_datasets = pd.read_excel('data/plotting_lut.xlsx', 
                                          sheetname='datasets_{0}'.format(self._period), 
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
        for dataset in tqdm(self._dataset_names, 
                            desc       = 'Loading dataframes',
                            unit_scale = True,
                            ncols      = 75,
                            total      = len(self._dataset_names)
                           ):

            df         = pd.read_pickle('{0}/ntuple_{1}.pkl'.format(self._input_dir, dataset))
            init_count = self._event_counts[dataset][0]
            lut_entry  = self._lut_datasets.loc[dataset]
            label      = lut_entry.label
            df['label'] = df.shape[0]*[label,]

            #### apply selection cuts ###
            if self._cuts != '':
                df = df.query(self._cuts)
            
            #### update weights with lumi scale factors ###
            if label.split('_')[0] != 'data':
                scale = self._scale*lut_entry.cross_section*lut_entry.branching_fraction/init_count
                df.loc[:,'weight'] = df['weight'].multiply(scale)
            else:
                df.loc[:,'weight'] = df['weight'].multiply(lut_entry.cross_section)

            ### combined datasets if required ###
            if self._combine:
                if label not in dataframes.keys():
                    dataframes[label] = df
                else:
                    dataframes[label] = dataframes[label].append(df)
            else:
                dataframes[dataset] = df
        self._dataframes = dataframes

    def get_dataframe(self, dataset_name, condition=''):
        df = self._dataframes[dataset_name]
        if condition != '':
            return df.query(condition)
        else:
            return df

    def get_dataframes(self, dataset_names, condition=''):
        dataframes = {}
        for dataset in dataset_names:
            df = self._dataframes[dataset]
            if condition == '':
                dataframes[dataset] = df
            else:
                dataframes[dataset] = df.query(condition)
        return dataframes

    def get_dataset_names(self):
        return self._dataset_names

    def print_yields(self, dataset_names, 
                     exclude    = [],
                     conditions = [''],
                     mc_scale   = True,
                     do_string  = True,
                     fmt        = 'markdown'
                    ):
        '''
        Prints sum of the weights for the provided datasets

        Parameters 
        ==========
        dataset_names : list of datasets to print
        exclude       : list of datasets to exclude from sum background calculation
        conditions    : list of conditions to apply
        mc_scale      : scale MC according to weights and scale
        do_string     : format of output cells: if True then string else float
        fmt           : formatting of the table (default:markdown)
        '''

        # print header
        table = OrderedDict()
        dataframes = self.get_dataframes(dataset_names)
        for condition in conditions:
            table[condition] = []
            bg_total = [0., 0.]
            for dataset in dataset_names:
                df = dataframes[dataset]
                if condition != '' and condition != 'preselection':
                    df = df.query(condition) 

                if mc_scale:
                    n     = df.weight.sum()
                    n_err = np.sqrt(np.sum(df.weight**2))
                else:
                    n     = df.shape[0]
                    n_err = np.sqrt(n)

                # calculate sum of bg events
                if dataset not in exclude and dataset != 'data':
                    bg_total[0] += n
                    bg_total[1] += n_err**2

                if do_string:
                    if dataset == 'data':
                        table[condition].append('${0}$'.format(int(n)))
                    else:
                        table[condition].append('${0:.1f} \pm {1:.1f}$'.format(n, n_err))
                else:
                    table[condition].append(n)
                     

                dataframes[dataset] = df # update dataframes so cuts are applied sequentially
            if do_string:
                table[condition].append('${0:.1f} \pm {1:.1f}$'.format(bg_total[0], np.sqrt(bg_total[1])))
            else:
                table[condition].append(bg_total[0])

        if do_string:
            labels = [self._lut_datasets.loc[d].text for d in dataset_names]
        else:
            labels = dataset_names
        table = pd.DataFrame(table, index=labels+['background'])
        return table


class PlotManager():
    def __init__(self, data_manager, features, stack_labels, overlay_labels, 
                 top_overlay = False,
                 output_path = 'plots',
                 file_ext    = 'png'
                ):
        self._dm             = data_manager
        self._features       = features
        self._stack_labels   = stack_labels
        self._overlay_labels = overlay_labels
        self._top_overlay    = top_overlay
        self._output_path    = output_path
        self._file_ext       = file_ext
        self._initialize_colors()

    def _initialize_colors(self):
        lut = self._dm._lut_datasets
        self._stack_colors   = [lut.loc[l].color for l in self._stack_labels]
        self._overlay_colors = [lut.loc[l].color for l in self._overlay_labels]

    def make_overlays(self, features, 
                      plot_data     = True,
                      normed        = False,
                      do_cms_text   = True,
                      overlay_style = 'line',
                      do_ratio      = False
                     ):
        dm = self._dm
        make_directory(self._output_path)

        ### alias dataframes and datasets lut###
        dataframes   = dm._dataframes
        lut_datasets = dm._lut_datasets

        ### initialize legend text ###
        legend_text = []
        legend_text.extend([lut_datasets.loc[label].text for label in self._stack_labels[::-1]])
        #legend_text.extend([lut_datasets.loc[label].text for label in self._overlay_labels[::-1]])

        if len(self._stack_labels) > 0:
            legend_text.append('BG error')
        if plot_data:
            #legend_text.append('Data')
            legend_text.append('Data (prompt)')
            legend_text.append('Data (rereco)')

        for feature in tqdm(features, desc='Plotting', unit_scale=True, ncols=75, total=len(features)):
            if feature not in self._features:
                print '{0} not in features.'
                continue
            #else:
            #    print feature

            ### Get style data for the feature ###
            lut_entry = dm._lut_features.loc[feature]

            ### initialize figure ###
            if do_ratio:
                fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[2,1]})
                fig.subplots_adjust(hspace=0)
                ax = axes[0]

            else:
                fig, ax = plt.subplots(1, 1)
            #legend_handles = []

            ### Get stack data and apply mask if necessary ###
            y_min, y_max = 1e9, 0.
            if len(self._stack_labels) > 0:
                stack_data, stack_weights = get_data_and_weights(dataframes, feature, self._stack_labels, lut_entry.condition)
                stack, bins, p = ax.hist(stack_data, 
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
                                             bins    = lut_entry.n_bins,
                                             range   = (lut_entry.xmin, lut_entry.xmax),
                                             weights = np.concatenate(stack_weights)**2
                                            )[0] 
                stack_sum = stack[-1] if len(stack_data) > 1 else stack
                stack_x   = (bins[1:] + bins[:-1])/2.
                stack_err = np.sqrt(stack_noscale)

                if do_ratio:
                    denominator = (stack_x, stack_sum, stack_err)

                no_blanks = stack_sum > 0
                stack_sum, stack_x, stack_err = stack_sum[no_blanks], stack_x[no_blanks], stack_err[no_blanks]
                eb = ax.errorbar(stack_x, stack_sum, yerr=stack_err, 
                              fmt        = 'none',
                              ecolor     = 'k',
                              capsize    = 0,
                              elinewidth = 10,
                              alpha      = 0.15
                             )
                if stack_sum.min() < y_min and stack_sum.min() > 0.:
                    y_min = stack_sum.min() 
                if stack_sum.max() > y_max:
                    y_max = stack_sum.max() 
                #legend_handles.append(eb[0])

            ### Get overlay data and apply mask if necessary ###
            if len(self._overlay_labels) > 0:
                overlay_data, overlay_weights = get_data_and_weights(dataframes, feature, self._overlay_labels, lut_entry.condition)
                if overlay_style == 'line':
                    hists, bins, p = ax.hist(overlay_data,
                                               bins      = lut_entry.n_bins,
                                               range     = (lut_entry.xmin, lut_entry.xmax),
                                               color     = self._overlay_colors,
                                               alpha     = 1.,
                                               histtype  = 'step',
                                               linewidth = 2.,
                                               #linestyle = '--',
                                               normed    = normed,
                                               bottom    = 0 if y_max == 0 or not self._top_overlay else stack[-1],
                                               weights   = overlay_weights
                                              )

                    hists = np.array(hists).flatten()
                    if hists.min() < y_min and hists.min() > 0.:
                        y_min = hists.min()
                    if hists.max() > y_max:
                        y_max = hists.max() 
                    #legend_handles.append(p)
                elif overlay_style == 'errorbar':
                    x, y, yerr = hist_to_errorbar(overlay_data, 
                                                  nbins = lut_entry.n_bins,
                                                  xlim  = (lut_entry.xmin, lut_entry.xmax)
                                                 )
                    if do_ratio:
                        numerator = (x, y, yerr)

                    x, y, yerr = x[y>0], y[y>0], yerr[y>0]
                    eb = ax.errorbar(x, y, yerr=yerr, 
                                  fmt        = 'bo',
                                  capsize    = 0,
                                  elinewidth = 2
                                 )

            ### If there's data to overlay: apply feature condition and get
            ### datapoints plus errors
            data_limit = [0., 0.]
            if plot_data:
                data, _ = get_data_and_weights(dataframes, feature, ['data'], lut_entry.condition)
                x, y, yerr = hist_to_errorbar(data, 
                                              nbins = lut_entry.n_bins,
                                              xlim  = (lut_entry.xmin, lut_entry.xmax)
                                             )
                if do_ratio:
                    numerator = (x, y, yerr)

                x, y, yerr = x[y>0], y[y>0], yerr[y>0]
                eb = ax.errorbar(x, y, yerr=yerr, 
                              fmt        = 'ko',
                              capsize    = 0,
                              elinewidth = 2
                             )
                if y.size > 0:
                    if y.min() < y_min and y.min() > 0.:
                        y_min = y.min()
                    if y.max() > y_max:
                        y_max = y.max() 
                #legend_handles.append(eb[0])

            ### make the legend ###
            #ax.legend(legend_text)

            ### labels and x limits ###
            if do_ratio:
                axes[1].set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))
                axes[1].set_ylabel(r'Data/MC')
                axes[1].set_ylim((0.5, 1.99))
                axes[1].grid()

                ### calculate ratios 
                ratio = numerator[1]/denominator[1]
                error = ratio*np.sqrt(numerator[2]**2/numerator[1]**2 + denominator[2]**2/denominator[1]**2)
                axes[1].errorbar(numerator[0], ratio, yerr=error,
                                 fmt = 'ko',
                                 capsize = 0,
                                 elinewidth = 2
                                )
                axes[1].plot([lut_entry.xmin, lut_entry.xmax], [1., 1.], 'r--')
            else:
                ax.set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))

            ax.set_ylabel(r'$\sf {0}$'.format(lut_entry.y_label))
            ax.set_xlim((lut_entry.xmin, lut_entry.xmax))
            ax.grid()

            ### Add lumi text ###
            if do_cms_text:
                add_lumi_text(ax, dm._scale, dm._period)

            ### Make output directory if it does not exist ###
            make_directory('{0}/linear/{1}'.format(self._output_path, lut_entry.category), False)
            make_directory('{0}/log/{1}'.format(self._output_path, lut_entry.category), False)

            ### Save output plot ###
            ### linear scale ###
            ax.set_ylim((0., 1.8*y_max))
            fig.savefig('{0}/linear/{1}/{2}.{3}'.format(self._output_path, 
                                                        lut_entry.category, 
                                                        feature, 
                                                        self._file_ext
                                                      ))

            ### log scale ###
            ax.set_yscale('log')
            ax.set_ylim(y_min/10., 15.*y_max)
            fig.savefig('{0}/log/{1}/{2}.{3}'.format(self._output_path, 
                                                     lut_entry.category, 
                                                     feature, 
                                                     self._file_ext
                                                    ))

            fig.clear()
            plt.close()

    def make_sideband_overlays(self, label, cuts, features, 
                               do_cms_text = True,
                               do_ratio    = False,
                               do_stacked  = False
                              ):

        ### alias dataframes and datasets lut###
        df_pre = self._dm.get_dataframe(label)
        df_sr  = df_pre.query(cuts[0])
        df_sb  = df_pre.query(cuts[1])
        for feature in tqdm(features, 
                            desc       = 'Plotting',
                            unit_scale = True,
                            ncols      = 75,
                            total      = len(features
                           )):
            if feature not in self._features:
                print '{0} not in features.'
                continue

            fig, ax = plt.subplots(1, 1)
            lut_entry = self._dm._lut_features.loc[feature]
            x_sr = df_sr[feature].values
            x_sb = df_sb[feature].values
            hist, bins, _ = ax.hist([x_sr, x_sb],
                                    bins      = lut_entry.n_bins,
                                    range     = (lut_entry.xmin, lut_entry.xmax),
                                    color     = ['k', 'r'],
                                    alpha     = 0.9,
                                    histtype  = 'step',
                                    linewidth = 2.,
                                    normed    = True,
                                    stacked   = do_stacked
                                   )

            ### make the legend ###
            #legend_text = cuts # Need to do something with this
            legend_text = [r'$\sf M_{\mu\mu}\,\notin\,[24,33]$', r'$\sf M_{\mu\mu}\,\in\,[24, 33]$']
            ax.legend(legend_text)

            ### labels and x limits ###
            ax.set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))
            ax.set_ylabel(r'$\sf {0}$'.format(lut_entry.y_label))
            ax.set_xlim((lut_entry.xmin, lut_entry.xmax))
            ax.grid()

            ### Add lumi text ###
            add_lumi_text(ax)

            ### Make output directory if it does not exist ###
            make_directory('{0}/linear/{1}'.format(self._output_path, lut_entry.category), False)
            make_directory('{0}/log/{1}'.format(self._output_path, lut_entry.category), False)

            ### Save output plot ###
            ### linear scale ###
            y_max = np.max(hist)
            ax.set_ylim((0., 1.8*y_max))
            fig.savefig('{0}/linear/{1}/{2}.{3}'.format(self._output_path, 
                                                        lut_entry.category, 
                                                        feature, 
                                                        self._file_ext
                                                       ))

            ### log scale ###
            ax.set_yscale('log')
            ax.set_ylim((0.1*np.min(hist), 15.*y_max))
            fig.savefig('{0}/log/{1}/{2}.{3}'.format(self._output_path, 
                                                     lut_entry.category, 
                                                     feature, 
                                                     self._file_ext
                                                    ))
            fig.clear()
            plt.close()
