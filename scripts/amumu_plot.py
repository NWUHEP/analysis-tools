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
    selection   = ('mumu', 'combined')
    period      = 2012
    ntuple_dir  = 'data/flatuples/{0}_{1}'.format(selection[0], period)
    lumi        = 19.8e3 if period == 2012 else 12e3
    plot_data   = True

    if period == 2016:
        bg_labels     = ['t', 'ttbar', 'zjets']
        signal_labels = ['bprime_xb']
        datasets = [
                    'muon_2016B', 'muon_2016C', 'muon_2016D', #'muon_2016E', 'muon_2016F', 
                    'ttjets', 
                    't_t', 't_tw', 'tbar_t', 'tbar_tw', 
                    'zjets_m-50', 'zjets_m-10to50',
                    'bprime_xb'
                   ]
    elif period == 2012:
        bg_labels     = ['t', 'diboson', 'ttbar', 'zjets']
        #signal_labels = ['bprime_xb', 'fcnc', 'bprime_xbxb_semilep']
        signal_labels = ['bprime_t-channel']
        datasets = [
                    'muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D', 
                    #'electron_2012A', 'electron_2012B', 'electron_2012C', 'electron_2012D', 
                    'ttbar_lep', 'ttbar_semilep',
                    'zjets_m-50',  'z1jets_m-50', 'z2jets_m-50', 
                    'z3jets_m-50', 'z4jets_m-50', 
                    'zjets_m-10to50', 'z1jets_m-10to50', 'z2jets_m-10to50', 
                    'z3jets_m-10to50', 'z4jets_m-10to50',
                    't_s', 't_t', 't_tw', 'tbar_s', 'tbar_t', 'tbar_tw', 
                    'ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q', 'zz_2l2nu',
                    'bprime_t-channel', #'bprime_bb_semilep', 'fcnc_s-channel'
                   ]

    features = [
                 'n_pu',

                 'lepton1_pt', 'lepton1_eta', 'lepton1_phi', 'lepton1_iso', 
                 'lepton2_pt', 'lepton2_eta', 'lepton2_phi', 'lepton2_iso', 
                 'lepton2_cos_theta',
                 'lepton_delta_eta', 'lepton_delta_phi', 'lepton_delta_r',
                 'dilepton_mass', 'dilepton_pt', 'dilepton_eta', 'dilepton_phi', 
                 'dilepton_pt_over_m',

                 'met_mag', 'met_phi',
                 'n_jets', 'n_fwdjets', 'n_bjets',
                 'bjet_pt', 'bjet_eta', 'bjet_phi', 'bjet_d0',
                 'jet_pt', 'jet_eta', 'jet_phi', 'jet_d0', 
                 'jet_delta_eta', 'jet_delta_phi', 'jet_delta_r',
                 'dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi', 
                 'dijet_pt_over_m',

                 'dilepton_b_mass', 'dilepton_b_pt', 
                 'dilepton_b_delta_r', 'dilepton_b_delta_eta', 'dilepton_b_delta_phi',
                 'dilepton_j_mass', 'dilepton_j_pt', 
                 'dilepton_j_delta_r', 'dilepton_j_delta_eta', 'dilepton_j_delta_phi',
                 'four_body_mass',
                 'four_body_delta_r', 'four_body_delta_eta', 'four_body_delta_phi', 
                 #'n_partons'
               ]

    ### Cuts ###
    cuts = {
            'test' : '(lepton1_pt > 25 and abs(lepton1_eta) < 2.1 \
                       and lepton2_pt > 20 and abs(lepton2_eta) < 2.4 \
                       and lepton1_q != lepton2_q \
                       and lepton1_trigger == True\
                       and n_jets == 0 and n_bjets == 0)',
            'preselection' : '(lepton1_pt > 25 and abs(lepton1_eta) < 2.1 \
                               and lepton2_pt > 25 and abs(lepton2_eta) < 2.1 \
                               and lepton1_q != lepton2_q and 12 < dilepton_mass < 70)', 
            'same-sign' : 'lepton1_q == lepton2_q',
            '2b'        : 'n_bjets == 2', 
            '1b1f'      : 'n_fwdjets > 0 and n_bjets == 1 and n_jets == 0',
            '1b1c'      : 'n_fwdjets == 0 and n_bjets >= 1 \
                           and n_bjets + n_jets == 2 \
                           and four_body_delta_phi > 2.5 and met_mag < 40',
            'combined'  : '((n_bjets >= 1 and n_fwdjets > 0 and n_jets == 0) \
                           or \
                           (n_bjets >= 1 and n_fwdjets == 0 and n_jets + n_bjets == 2 \
                           and four_body_delta_phi > 2.5 and met_mag < 40))',
            'enhance'   : 'dilepton_pt_over_m > 2 and met_mag < 40 and four_body_delta_phi > 2.5' 
            }

    cuts['combined_sideband'] = cuts['combined'] + \
                                'and (dilepton_mass > 36 or dilepton_mass < 24)'
    cuts['combined_signal']   = cuts['combined'] + \
                                'and (dilepton_mass > 36 or dilepton_mass < 24)'
    cuts['combined_enhance']  = cuts['combined'] + ' and ' + cuts['enhance']
    cuts['1b1f_enhance']      = cuts['1b1f'] + ' and ' + cuts['enhance']
    cuts['1b1c_enhance']      = cuts['1b1c'] + ' and ' + cuts['enhance']

    if selection[1] not in cuts.keys():
        cut = ''
    else:
        cut = cuts[selection[1]]
        if selection[1] not in ['same-sign', 'test']:
            cut += ' and ' + cuts['preselection']

    ### Get dataframes with features for each of the datasets ###
    data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                  dataset_names = datasets,
                                  selection     = selection[0],
                                  period        = period,
                                  scale         = lumi,
                                  cuts          = cut
                                 )

    ### Loop over features and make the plots ###
    output_path  = 'plots/overlays/{0}_{1}'.format('_'.join(selection), period)
    plot_manager = pt.PlotManager(data_manager,
                                  features       = features,
                                  stack_labels   = bg_labels,
                                  overlay_labels = signal_labels,
                                  top_overlay    = True,
                                  output_path    = output_path,
                                  file_ext       = 'png'
                                 )

    if True:
        #plot_manager.make_overlays(features, plot_data=False, normed=True)
        plot_manager.make_overlays(features)
    else:
        regions = ['26 < dilepton_mass < 32', 'dilepton_mass < 26 or dilepton_mass > 32']
        plot_manager.make_sideband_overlays('data', regions, features)

    ### Dalitz plots ###


    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
