#!/usr/bin/env python

import argparse
from itertools import chain

import matplotlib as mpl
mpl.use('Agg')

import scripts.plot_tools as pt

if __name__ == '__main__':

    pt.set_new_tdr()

    # input arguments
    parser = argparse.ArgumentParser(description='Produce data/MC overlays')
    parser.add_argument('input',
                        help = 'specify input directory',
                        type = str
                        )
    parser.add_argument('-s', '--selection',
                        help = 'selection type',
                        default = 'mumu',
                        type = str
                        )
    parser.add_argument('-p', '--period',
                        help = 'data gathering period',
                        default = 2016,
                        type = int
                        )
    parser.add_argument('-l', '--lumi',
                        help = 'integrated luminosity for data',
                        default = 35.9e3,
                        type = float
                        )
    args = parser.parse_args()
    ###

    selection = args.selection
    data_labels  = ['muon', 'electron']
    model_labels = ['diboson', 'zjets', 't', 'wjets', 'ttbar']

    if selection in ['mu4j', 'e4j']: 
        model_labels = ['fakes'] + model_labels
    elif selection in ['mutau', 'etau']:
        model_labels = ['fakes_ss'] + model_labels

    # data samples
    features = [
                #'lepton1_reco_weight', 'lepton2_reco_weight', 'trigger_weight', 
                #'pileup_weight', 'top_pt_weight', 'event_weight',
                'gen_cat',

                'n_pv', 'n_muons', 'n_electrons', 'n_taus',
                'n_jets', 'n_fwdjets', 'n_bjets',
                'met_mag', 'met_phi', 'ht_mag', 'ht_phi',

                'lepton1_pt', 'lepton1_eta', 'lepton1_phi', #'lepton1_mt', 
                'lepton1_iso', 'lepton1_reliso', 
                #'jet1_pt', 'jet1_eta', 'jet1_phi',
                #'jet2_pt', 'jet2_eta', 'jet2_phi',
               ]

    if selection not in ['e4j', 'mu4j']:
        features.extend([
                         'lead_lepton_pt', 'trailing_lepton_pt',
                         'lepton2_pt', 'lepton2_eta', 'lepton2_phi', 'lepton2_mt', 
                         'lepton2_iso', 'lepton2_reliso', 
                         'lepton1_d0', 'lepton1_dz', 
                         'lepton2_d0', 'lepton2_dz',

                         'dilepton1_delta_eta', 'dilepton1_delta_phi', 'dilepton1_delta_r',
                         'dilepton1_mass', 'dilepton1_pt', 'dilepton1_eta', 'dilepton1_phi', 
                         'dilepton1_pt_over_m', 'dilepton1_pt_asym',

                         #'jet_delta_eta', 'jet_delta_phi', 'jet_delta_r',
                         #'dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi', 
                         #'dijet_pt_over_m',
                         ])
        if selection in ['etau', 'mutau']:
            features.append('tau_decay_mode')

    ### Cuts ###
    if selection in ['e4j', 'mu4j']:
        cut = 'n_jets >= 4 and n_bjets >= 1'
    elif selection == 'emu':
        cut = 'n_jets >= 2 and n_bjets >= 0'
    elif selection in ['etau', 'mutau']:
        cut = 'n_jets >= 2 and n_bjets >= 1'
    else:
        cut = 'n_jets >= 2 and n_bjets >= 1'
    cut += ' and ' + pt.cuts[selection]
            
    ### Get dataframes with features for each of the datasets ###
    data_manager = pt.DataManager(input_dir     = f'{args.input}/{args.selection}_{args.period}',
                                  dataset_names = [d for l in data_labels+model_labels for d in pt.dataset_dict[l]],
                                  selection     = selection,
                                  period        = args.period,
                                  scale         = args.lumi,
                                  cuts          = cut
                                 )

    ### Loop over features and make the plots ###
    output_path = f'plots/overlays/{selection}_{args.period}'
    plot_manager = pt.PlotManager(data_manager,
                                  features       = features,
                                  stack_labels   = model_labels,
                                  overlay_labels = [],
                                  top_overlay    = False,
                                  output_path    = output_path,
                                  file_ext       = 'png'
                                 )

    pt.make_directory(output_path, clear=True)
    plot_manager.make_overlays(features, do_ratio=True, overlay_style='errorbar')

    table = data_manager.print_yields(dataset_names=['data'] + model_labels, conditions=['n_bjets >= 1', 'n_bjets >= 2'])
    table.to_csv(f'{output_path}/yields_{selection}.csv')
