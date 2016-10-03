#!/usr/bin/env python

import os, sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nllfitter.plot_tools as pt

if __name__ == '__main__':

    pt.set_new_tdr()

    ### Start the timer
    start = timer()

    ### Configuration
    ntuple_dir  = 'data/flatuples/ee_2012'
    lumi        = 19.8e3
    selection   = ('ee', 'preselection')
    period      = 2012
    output_path = 'plots/overlays/{0}_{1}'.format('_'.join(selection), period)
    plot_data   = True

    datasets   = [
                  #'muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D', 
                  'electron_2012A', 'electron_2012B', 'electron_2012C', 'electron_2012D', 
                  'ttbar_lep', 
                  'zjets_m-50', 'zjets_m-10to50',
                  't_s', 't_t', 't_tw', 'tbar_s', 'tbar_t', 'tbar_tw', 
                  'ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q', 'zz_2l2nu',
                  'bprime_xb'
                 ]

    features = [
                 'lepton1_pt', 'lepton1_eta', 'lepton1_phi', 'lepton1_iso', 
                 'lepton2_pt', 'lepton2_eta', 'lepton2_phi', 'lepton2_iso', 
                 'lepton_delta_eta', 'lepton_delta_phi', 'lepton_delta_r',
                 'dilepton_mass', 'dilepton_pt', 'dilepton_eta', 'dilepton_phi', 
                 'dilepton_pt_over_m',

                 'met_mag', 'met_phi',
                 'n_jets', 'n_fwdjets', 'n_bjets',
                 'bjet_pt', 'bjet_eta', 'bjet_phi', 'bjet_d0',
                 'jet_pt', 'jet_eta', 'jet_phi', 'jet_d0', 
                 'dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi', 
                 'dijet_pt_over_m',

                 'dilepton_b_mass', 'dilepton_b_pt', 
                 'dilepton_b_delta_r', 'dilepton_b_delta_eta', 'dilepton_b_delta_phi',
                 'dilepton_j_mass', 'dilepton_j_pt', 
                 'dilepton_j_delta_r', 'dilepton_j_delta_eta', 'dilepton_j_delta_phi',
                 'four_body_mass',
                 'four_body_delta_r', 'four_body_delta_eta', 'four_body_delta_phi', 
               ]

    ### Cuts ###
    cuts = {
            'opp-sign'  : 'lepton1_q != lepton2_q',
            'same-sign' : 'lepton1_q == lepton2_q',
            '2b'        : 'n_bjets == 2', 
            '1b1f'      : 'n_fwdjets > 0 and n_bjets == 1 and n_jets == 0',
            '1b1c'      : 'n_fwdjets == 0 and n_bjets == 1 and n_jets == 1 \
                           and four_body_delta_phi > 2.5 and met_mag < 40',
            'combined'  : 'n_bjets == 1 and \
                           ((n_fwdjets > 0 and n_jets == 0) or \
                           (n_fwdjets == 0 and n_jets == 1 and \
                           four_body_delta_phi > 2.5 and met_mag < 40))',
            #'enhance'   : '100 < dilepton_b_mass < 200 and dilepton_pt_over_m > 2',
            'enhance'   : 'dilepton_pt_over_m > 2',
           }

    cuts['combined_sideband'] = cuts['combined'] + \
                                'and (dilepton_mass > 36 or dilepton_mass < 24)'
    cuts['combined_signal']   = cuts['combined'] + \
                                'and (dilepton_mass > 36 or dilepton_mass < 24)'
    cuts['combined_enhance']  = cuts['combined'] + ' and ' + cuts['enhance']

    if selection[1] not in cuts.keys():
        cuts[selection[1]] = ''
    elif selection[1] != 'same-sign':
        cuts[selection[1]] += ' and ' + cuts['opp-sign']

    ### Get dataframes with features for each of the datasets ###
    data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                  dataset_names = datasets,
                                  selection     = selection[0],
                                  scale         = lumi,
                                  cuts          = cuts[selection[1]]
                                 )

    ### Loop over features and make the plots ###
    plot_manager = pt.PlotManager(data_manager,
                                  features       = features,
                                  stack_labels   = ['t', 'diboson', 'ttbar', 'zjets'],
                                  overlay_labels = []
                                 )
    plot_manager.make_overlays(features, 
                               output_path = output_path,
                               file_ext    = 'png'
                              )
    '''

    ### Overlay sideband and signal region ###
    label       = 'data'
    do_stacked  = False
    output_path = 'plots/overlays/sb_over_sr/{0}_overlayed'.format(selection[1])
    file_ext    = 'png'
    conditions  = ['dilepton_mass < 24 or dilepton_mass > 33', '24 < dilepton_mass < 33']

    df_pre      = data_manager.get_dataframe(label, cuts[selection[1]])
    df_sr       = df_pre.query(conditions[0])
    df_sb       = df_pre.query(conditions[1])
    for feature in features:
        print feature

        fig, ax = plt.subplots(1, 1)
        lut_entry = data_manager._lut_features.loc[feature]
        x_sr = df_sr[feature].values
        x_sb = df_sb[feature].values
        hist, bins, _ = ax.hist([x_sr, x_sb],
                                bins      = lut_entry.n_bins,
                                range     = (lut_entry.xmin, lut_entry.xmax),
                                color     = ['k', 'r'],
                                alpha     = 0.9,
                                histtype  = 'step',
                                linewidth = 2.,
                                stacked   = True if do_stacked else False
                               )

        ### make the legend ###
        legend_text = [r'$\sf M_{\mu\mu}\,\in\,[24, 33]$', r'$\sf M_{\mu\mu}\,\notin\,[24,33]$']
        ax.legend(legend_text)

        ### labels and x limits ###
        ax.set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))
        ax.set_ylabel(r'$\sf {0}$'.format(lut_entry.y_label))
        ax.set_xlim((lut_entry.xmin, lut_entry.xmax))
        ax.grid()

        ### Add lumi text ###
        pt.add_lumi_text(ax)

        ### Make output directory if it does not exist ###
        pt.make_directory('{0}/linear/{1}'.format(output_path, lut_entry.category), False)
        pt.make_directory('{0}/log/{1}'.format(output_path, lut_entry.category), False)

        ### Save output plot ###
        ### linear scale ###
        y_max = np.max(hist)
        ax.set_ylim((0., 1.8*y_max))
        fig.savefig('{0}/linear/{1}/{2}.{3}'.format(output_path, lut_entry.category, feature, file_ext))

        ### log scale ###
        ax.set_yscale('log')
        ax.set_ylim((0.1*np.min(hist), 15.*y_max))
        fig.savefig('{0}/log/{1}/{2}.{3}'.format(output_path, lut_entry.category, feature, file_ext))
        fig.clear()
        plt.close()
    '''

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
