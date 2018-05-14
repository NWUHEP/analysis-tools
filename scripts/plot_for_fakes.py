#!/usr/bin/env python

import argparse
from itertools import chain

import scripts.plot_tools as pt

dataset_dict = dict(
                    muon     = ['muon_2016B', 'muon_2016C', 'muon_2016D', 'muon_2016E', 'muon_2016F', 'muon_2016G', 'muon_2016H'],
                    electron = ['electron_2016B', 'electron_2016C', 'electron_2016D', 'electron_2016E', 'electron_2016F', 'electron_2016G', 'electron_2016H'],
                    ttbar    = ['ttbar_inclusive'], #'ttbar_lep', 'ttbar_semilep',
                    t        = ['t_tw', 'tbar_tw'], #'t_t', 'tbar_t',
                    wjets    = ['w1jets', 'w2jets', 'w3jets', 'w4jets'],
                    zjets    = ['zjets_m-50',  'zjets_m-10to50', 
                                'z1jets_m-50', 'z1jets_m-10to50', 
                                'z2jets_m-50', 'z2jets_m-10to50', 
                                'z3jets_m-50', 'z3jets_m-10to50', 
                                'z4jets_m-50', 'z4jets_m-10to50'
                                ],
                    qcd      = ['qcd_ht100to200', 'qcd_ht200to300', 'qcd_ht300to500', 
                                'qcd_ht500to1000', 'qcd_ht1000to1500', 'qcd_ht1500to2000', 
                                'qcd_ht2000'
                                ],
                    diboson  = ['ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q'], #'zz_4l',
                    fakes    = ['fakes'],
                    )

if __name__ == '__main__':

    # input arguments
    parser = argparse.ArgumentParser(description='Produce data/MC overlays')
    parser.add_argument('input',
                        help = 'specify input directory',
                        type = str
                        )
    parser.add_argument('-s', '--selection',
                        help = 'selection type',
                        default = 'fakes',
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

    features = [
                #'lepton1_reco_weight', 'lepton2_reco_weight', 'trigger_weight', 
                #'pileup_weight', 'top_pt_weight', 'event_weight',

                'n_pv', 'n_muons', 'n_electrons',
                'n_jets', 'n_fwdjets', 'n_bjets',
                'met_mag', 'met_phi', 'ht_mag', 'ht_phi',

                'lepton1_pt', 'lepton1_eta', 'lepton1_phi', 
                'lepton1_iso', 'lepton1_reliso', 'lepton1_d0', 'lepton1_dz',
                'lepton2_pt', 'lepton2_eta', 'lepton2_phi',
                'lepton2_iso', 'lepton2_reliso', 'lepton2_d0', 'lepton2_dz',
                'lepton3_pt', 'lepton3_eta', 'lepton3_phi',
                'lepton3_iso', 'lepton3_reliso', 'lepton3_d0', 'lepton3_dz',

                'dilepton1_delta_eta', 'dilepton1_delta_phi', 'dilepton1_delta_r',
                'dilepton1_mass', 'dilepton1_pt', 'dilepton1_eta', 'dilepton1_phi', 
                'dilepton1_pt_over_m', 'dilepton1_pt_asym',

                'dilepton_probe_delta_eta', 'dilepton_probe_delta_phi', 'dilepton_probe_delta_r', 
                'dilepton_probe_mass', 'dilepton_probe_pt_asym',  

                'jet1_pt', 'jet1_eta', 'jet1_phi',
                'jet2_pt', 'jet2_eta', 'jet2_phi',

                'jet_delta_eta', 'jet_delta_phi', 'jet_delta_r',
                'dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi', 
                #'dijet_pt_over_m',
               ]


    ### Cuts ###
    cut = 'n_muons == 3 \
           and lepton1_q != lepton2_q \
           and abs(dilepton1_mass - 91) < 15 \
           and lepton3_pt > 10' 
           #and dilepton_probe_mass > 175 \
           #and lepton3_reliso < 0.15'
           #and lepton3_pt > 20 \
           #and n_bjets == 0 \

    datasets  = dataset_dict['muon'] 
    datasets.extend(['ttbar_lep', 'ttbar_semilep', 'wz_3lnu', 'zz_4l', 'zjets_m-50'])
    mc_labels = ['ttbar', 'zz_4l', 'wz_3lnu', 'zjets']
            
    ### Get dataframes with features for each of the datasets ###
    data_manager = pt.DataManager(input_dir     = args.input,
                                  dataset_names = datasets,
                                  selection     = args.selection,
                                  period        = args.period,
                                  scale         = args.lumi,
                                  cuts          = cut
                                 )

    ### Loop over features and make the plots ###
    pt.set_new_tdr()
    output_path = f'plots/overlays/wz_cr'
    plot_manager = pt.PlotManager(data_manager,
                                  features       = features,
                                  stack_labels   = mc_labels,
                                  output_path    = output_path,
                                 )

    pt.make_directory(output_path, clear=True)
    plot_manager.make_overlays(features, do_ratio=True, overlay_style='errorbar')

    table = data_manager.print_yields(dataset_names = mc_labels + ['data'])
    table.to_csv(f'data/yields_wz_cr.csv')
