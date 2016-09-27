#!/usr/bin/env python

import os, sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nllfitter.plot_tools as dt


if __name__ == '__main__':

    dt.set_new_tdr()

    ### Start the timer
    start = timer()

    ### Configuration
    ntuple_dir = 'data/flatuples/mumu_2012'
    lumi       = 19.8e3
    selection  = 'mumu_combined_sb'
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
                 'dimuon_b_mass', 'dimuon_b_pt', 
                 'dimuon_b_delta_r', 'dimuon_b_delta_eta', 'dimuon_b_delta_phi',
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
    selection_cut['mumu_combined_sr']  = selection_cut['mumu_combined'] + 'and 24 < dimuon_mass < 36'
    selection_cut['mumu_combined_sb']  = selection_cut['mumu_combined'] + 'and (dimuon_mass > 36 or dimuon_mass < 24)'


    ### Get dataframes with features for each of the datasets ###
    data_manager = dt.DataManager(input_dir     = ntuple_dir,
                                  dataset_names = datasets,
                                  features      = features,
                                  scale         = lumi,
                                  cuts          = selection_cut[selection]
                                 )

    ### Loop over features and make the plots ###
    plot_manager = dt.PlotManager(data_manager,
                                  stack_labels   = ['t', 'diboson', 'ttbar', 'zjets'],
                                  overlay_labels = []
                                 )
    plot_manager.make_overlays(features, 
                               output_path = 'plots/overlays/{0}_{1}'.format(selection, period),
                               file_ext    = 'png'
                               )

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
