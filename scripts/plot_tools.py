#!/usr/bin/env python

import os, sys
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
plt.rcParams['font.size']         = 18
plt.rcParams['axes.labelsize']    = 20
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize']   = 18
plt.rcParams['ytick.labelsize']   = 18
plt.rcParams['legend.fontsize']   = 20
plt.rcParams['figure.titlesize']  = 20
plt.rcParams['figure.figsize']    = [11, 8]
plt.rcParams['legend.numpoints']  = 1

def hist_to_errorbar(data, nbins, xlim, normed=False):
    y, bins = np.histogram(data, bins=nbins, range=xlim)
    x = (bins[1:] + bins[:-1])/2.
    yerr = np.sqrt(y) 

    return y, x, yerr

def make_directory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)

    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')

def get_data_and_weights(dataframes, labels, condition):
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

if __name__ == '__main__':

    ### Start the timer
    start = timer()

    ### Configuration
    ntuple_dir = 'data/flatuples/mumu_2012'
    lumi       = 19.8e3
    selection  = 'mumu_2b'
    period     = 2012
    plot_data  = True
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
                 'dijet_pt_over_m',
                 'dimuon_b_mass', 'dimuon_b_pt', 'dimuon_b_delta_eta', 'dimuon_b_delta_phi',
                 'four_body_delta_phi', 'four_body_delta_eta', 'four_body_mass',
                 'met_mag', 'met_phi',
               ]

    ### Cuts ###
    selection_cut = {}
    selection_cut['mumu_preselection'] = ''
    selection_cut['mumu_2b']           = 'n_bjets == 2' 
    selection_cut['mumu_1b1f']         = 'n_fwdjets > 0 and n_bjets == 1 and n_jets == 0'
    selection_cut['mumu_1b1c']         = 'n_fwdjets == 0 and n_bjets == 1 and n_jets == 1 \
                                          and four_body_delta_phi > 2.5 and met_mag < 40'
    selection_cut['mumu_combined']     = 'n_bjets == 1 and \
                                          ((n_fwdjets > 0 and n_jets == 0) or \
                                          (n_fwdjets == 0 and n_jets == 1 and \
                                          four_body_delta_phi > 2.5 and met_mag < 40))'

    ### Get dataframes with styles, cross-sections, event_counts, etc.
    lut_datasets = pd.read_excel('data/plotting_lut.xlsx', sheetname='datasets', index_col='dataset_name').dropna(how='all')
    lut_features = pd.read_excel('data/plotting_lut.xlsx', sheetname='variables', index_col='variable_name').dropna(how='all')
    event_counts = pd.read_csv('{0}/event_counts.csv'.format(ntuple_dir))

    ### Get dataframes with features for each of the datasets ###
    dataframes = {}
    for dataset in datasets:
        df        = pd.read_csv('{0}/ntuple_{1}.csv'.format(ntuple_dir, dataset))
        lut_entry = lut_datasets.loc[dataset]
        label     = lut_entry.label

        ### apply selection cuts ###
        if selection_cut[selection] != '':
            df = df.query(selection_cut[selection])
        
        ### update weights with lumi scale factors ###
        if label != 'data':
            scale = lumi*lut_entry.cross_section*lut_entry.branching_fraction/event_counts[dataset][0]
            df.weight = df.weight.multiply(scale)

        ### combined datasets if required ###
        if label not in dataframes.keys():
            dataframes[label] = df
        else:
            dataframes[label] = dataframes[label].append(df)

    ### Loop over features and make the plots ###
    stack_labels   = ['t', 'diboson', 'ttbar', 'zjets']
    stack_colors   = [lut_datasets.loc[l].color for l in stack_labels]
    overlay_labels = []
    overlay_colors = [lut_datasets.loc[l].color for l in overlay_labels]
    for feature in features:
        print feature

        ### Get style data for the feature ###
        lut_entry = lut_features.loc[feature]

        ### initialize figure ###
        fig, axes = plt.subplots(1, 1)

        ### Get stack data and apply mask if necessary ###
        stack_data, stack_weights = get_data_and_weights(dataframes, stack_labels, lut_entry.condition)
        stack = axes.hist(stack_data, 
                          bins      = lut_entry.n_bins,
                          range     = (lut_entry.xmin, lut_entry.xmax),
                          color     = stack_colors,
                          alpha     = 1.,
                          linewidth = 0.5,
                          stacked   = True,
                          histtype  = 'stepfilled',
                          weights   = stack_weights
                        )

        ### Need to histogram the stack without weights to get the errors ### 
        stack_noscale = np.histogram(np.concatenate(stack_data), 
                                     bins  = lut_entry.n_bins,
                                     range = (lut_entry.xmin, lut_entry.xmax),
                                     weights = np.concatenate(stack_weights)**2
                                    )[0] 
        stack_sum = stack[0][-1]
        stack_x   = (stack[1][1:] + stack[1][:-1])/2.
        stack_err = np.sqrt(stack_noscale)
        no_blanks = stack_sum > 0
        stack_sum, stack_x, stack_err = stack_sum[no_blanks], stack_x[no_blanks], stack_err[no_blanks]
        axes.errorbar(stack_x, stack_sum, yerr=stack_err, 
                      fmt        = 'none',
                      ecolor     = 'k',
                      capsize    = 0,
                      elinewidth = 10,
                      alpha      = 0.15
                     )

        ### Get overlay data and apply mask if necessary ###
        if len(overlay_labels) > 0:
            overlay_data, overlay_weights = get_data_and_weights(dataframes, overlay_labels, lut_entry.condition)
            axes.hist(overlay_data,
                     bins      = lut_entry.n_bins,
                     range     = (lut_entry.xmin, lut_entry.xmax),
                     color     = overlay_colors,
                     alpha     = 0.9,
                     stacked   = True,
                     histtype  = 'step',
                     linewidth = 2.,
                     weights   = overlay_weights
                    )

        ### Make datapoints with errors ###
        if plot_data:
            data, weights = get_data_and_weights(dataframes, ['data'], lut_entry.condition)
            y, x, yerr = hist_to_errorbar(data, 
                                          nbins = lut_entry.n_bins,
                                          xlim  = (lut_entry.xmin, lut_entry.xmax)
                                         )
            y, x, yerr = y[y>0], x[y>0], yerr[y>0]
            axes.errorbar(x, y, yerr=yerr, 
                          fmt='ko', 
                          capsize=0, 
                          elinewidth=2
                         )

        ### make the legend ###
        legend_text = [lut_datasets.loc[label].text for label in stack_labels[::-1]] 
        legend_text += ['BG error']
        legend_text += [lut_datasets.loc[label].text for label in overlay_labels[::-1]]
        if plot_data:
            legend_text += ['Data']
        axes.legend(legend_text)

        ### labels and x limits ###
        axes.set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))
        axes.set_ylabel(r'$\sf {0}$'.format(lut_entry.y_label))
        axes.set_xlim((lut_entry.xmin, lut_entry.xmax))
        axes.grid()

        ### Add lumi text ###
        axes.text(0.06, 0.9, r'$\bf CMS$', fontsize=30, transform=axes.transAxes)
        axes.text(0.17, 0.9, r'$\it Preliminary $', fontsize=20, transform=axes.transAxes)
        axes.text(0.68, 1.01, r'$\sf{19.7\,fb^{-1}}\,(\sqrt{\it{s}}=8\,\sf{TeV})$', fontsize=20, transform=axes.transAxes)

        ### Make output directory if it does not exist ###
        make_directory('plots/overlays/{0}_{1}/linear/{2}'.format(selection, period, lut_entry.category), False)
        make_directory('plots/overlays/{0}_{1}/log/{2}'.format(selection, period, lut_entry.category), False)

        ### Save output plot ###
        ### linear scale ###
        axes.set_ylim((0., 1.8*np.max(stack_sum)))
        fig.savefig('plots/overlays/{0}_{1}/linear/{2}/{3}.png'.format(selection, period, lut_entry.category, feature))
        #fig.savefig('plots/overlays/{0}_{1}/linear/{2}/{3}.pdf'.format(selection, period, lut_entry.category, feature))

        ### log scale ###
        axes.set_yscale('log')
        axes.set_ylim((0.1*np.min(stack_sum), 15.*np.max(stack_sum)))
        fig.savefig('plots/overlays/{0}_{1}/log/{2}/{3}.png'.format(selection, period, lut_entry.category, feature))
        #fig.savefig('plots/overlays/{0}_{1}/log/{2}/{3}.pdf'.format(selection, period, lut_entry.category, feature))

        fig.clear()
        plt.close()

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
