#!/usr/bin/env python

import sys
import scripts.plot_tools as pt

if __name__ == '__main__':

    pt.set_new_tdr()


    ### Configuration
    if len(sys.argv) > 1:
        selection     = sys.argv[1]
    else:
        selection     = 'mumu'

    period        = 2016
    ntuple_dir    = f'data/flatuples/{selection}_test_{period}'
    lumi          = 19.8e3 if period == 2012 else 35.9e3
    plot_data     = True
    bg_labels     = ['diboson', 'wjets', 'zjets', 't', 'ttbar']
    signal_labels = []

    datasets      = []
    # data samples
    datasets.extend([
                    'muon_2016B', 'muon_2016C', 'muon_2016D', 
                    'muon_2016E', 'muon_2016F', 'muon_2016G', 'muon_2016H', 
                    'electron_2016B', 'electron_2016C', 'electron_2016D', 
                    'electron_2016E', 'electron_2016F', 'electron_2016G', 'electron_2016H', 
                    ])

    # MC samples
    datasets.extend([
                     'ttbar_inclusive',
                     #'ttbar_lep', 'ttbar_semilep',
                     't_tw', 'tbar_tw', #'t_t', 'tbar_t', 
                     'w1jets', 'w2jets', 'w3jets', 'w4jets',
                     #'zjets_m-50',  'zjets_m-10to50',
                     'z1jets_m-50', 'z1jets_m-10to50',
                     'z2jets_m-50', 'z2jets_m-10to50',
                     'z3jets_m-50', 'z3jets_m-10to50',
                     'z4jets_m-50', 'z4jets_m-10to50',
                     #'qcd_ht100to200', 'qcd_ht200to300',
                     #'qcd_ht300to500', 'qcd_ht500to1000',
                     #'qcd_ht1000to1500', 'qcd_ht1500to2000',
                     #'qcd_ht2000'
                     'ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q', 
                     #'zz_4l',
                     'fakes'
                    ])

    features = [
                #'lepton1_reco_weight', 'lepton2_reco_weight', 'trigger_weight', 
                #'pileup_weight', 'top_pt_weight', 'event_weight',
                'n_pv', 'n_muons', 'n_electrons', 'n_taus',
                'n_jets', 'n_fwdjets', 'n_bjets',
                'met_mag', 'met_phi', 'ht_mag', 'ht_phi',

                'lepton1_pt', 'lepton1_eta', 'lepton1_phi', 
                'lepton1_iso', 'lepton1_reliso', #'lepton1_d0', 'lepton1_dz',

                'jet1_pt', 'jet1_eta', 'jet1_phi',
                'jet2_pt', 'jet2_eta', 'jet2_phi',
               ]

    if selection in ['mu4j', 'e4j']:
        features.extend([
                    'jet1_pt', 'jet1_eta', 'jet1_phi',
                    'jet2_pt', 'jet2_eta', 'jet2_phi',
                    'jet3_pt', 'jet3_eta', 'jet3_phi',
                    'jet4_pt', 'jet4_eta', 'jet4_phi',

                    'lepton_j1_mass', 'lepton_j1_pt', 
                    'lepton_j1_delta_r', 'lepton_j1_delta_eta', 'lepton_j1_delta_phi',
                    'lepton_j2_mass', 'lepton_j2_pt', 
                    'lepton_j2_delta_r', 'lepton_j2_delta_eta', 'lepton_j2_delta_phi',
                    'lepton_j3_mass', 'lepton_j3_pt', 
                    'lepton_j3_delta_r', 'lepton_j3_delta_eta', 'lepton_j3_delta_phi',
                    'lepton_j4_mass', 'lepton_j4_pt', 
                    'lepton_j4_delta_r', 'lepton_j4_delta_eta', 'lepton_j4_delta_phi',

                    'w_pt', 'w_eta', 'w_phi', 'w_mass', 
                    'w_delta_r', 'w_delta_eta', 'w_delta_phi',
                    'htop_pt', 'htop_eta', 'htop_phi', 'htop_mass', 
                    'htop_delta_r', 'htop_delta_eta', 'htop_delta_phi',
                    ])
        if selection == 'mu4j':
            bg_labels = ['fakes'] + bg_labels
    else:
        features.extend([
                     'lepton2_pt', 'lepton2_eta', 'lepton2_phi',
                     'lepton2_iso', 'lepton2_reliso', #'lepton2_d0', 'lepton2_dz',

                     'lead_lepton_pt', 'trailing_lepton_pt',
                     'dilepton1_delta_eta', 'dilepton1_delta_phi', 'dilepton1_delta_r',
                     'dilepton1_mass', 'dilepton1_pt', 'dilepton1_eta', 'dilepton1_phi', 
                     'dilepton1_pt_over_m', 'dilepton1_pt_asym',

                     'jet_delta_eta', 'jet_delta_phi', 'jet_delta_r',
                     'dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi', 
                     'dijet_pt_over_m',

                     #'lepton1_j1_mass', 'lepton1_j1_pt', 
                     #'lepton1_j1_delta_r', 'lepton1_j1_delta_eta', 'lepton1_j1_delta_phi',
                     #'lepton1_j2_mass', 'lepton1_j2_pt', 
                     #'lepton1_j2_delta_r', 'lepton1_j2_delta_eta', 'lepton1_j2_delta_phi',
                     #'lepton2_j1_mass', 'lepton2_j1_pt', 
                     #'lepton2_j1_delta_r', 'lepton2_j1_delta_eta', 'lepton2_j1_delta_phi',
                     #'lepton2_j2_mass', 'lepton2_j2_pt', 
                     #'lepton2_j2_delta_r', 'lepton2_j2_delta_eta', 'lepton2_j2_delta_phi',

                     #'dilepton_j1_mass', 'dilepton_j1_pt', 
                     #'dilepton_j1_delta_r', 'dilepton_j1_delta_eta', 'dilepton_j1_delta_phi',
                     #'dilepton_j2_mass', 'dilepton_j2_pt', 
                     #'dilepton_j2_delta_r', 'dilepton_j2_delta_eta', 'dilepton_j2_delta_phi',
                     #'four_body_mass',
                     #'four_body_delta_r', 'four_body_delta_eta', 'four_body_delta_phi', 
                    ])

        if selection == '4l':
            features.extend([
                             'lepton3_pt', 'lepton3_eta', 'lepton3_phi', 
                             'lepton3_iso', 'lepton3_reliso',
                             'lepton4_pt', 'lepton4_eta', 'lepton4_phi', 
                             'lepton4_iso', 'lepton4_reliso',

                             'dilepton2_delta_eta', 'dilepton2_delta_phi', 'dilepton2_delta_r',
                             'dilepton2_mass', 'dilepton2_pt', 'dilepton2_eta', 'dilepton2_phi', 
                             'dilepton2_pt_over_m',

                             'tetralepton_delta_eta', 'tetralepton_delta_phi', 'tetralepton_delta_r',
                             'tetralepton_mass', 'tetralepton_pt', 'tetralepton_eta', 'tetralepton_phi', 
                             'tetralepton_pt_over_m',
                             ])
        if selection == 'mutau':
            features.extend(['tau_decay_mode', 'tau_mva'])

    ### Cuts ###
    cut = 'lepton1_pt > 25 and n_bjets >= 1'
            
    if selection == 'ee':
        cut += ' and dilepton1_mass > 12 \
                and (dilepton1_mass < 80 or dilepton1_mass > 102) \
                and lepton1_q != lepton2_q'
    elif selection == 'mumu':
        cut += 'and lepton1_pt > 25 and abs(lepton1_eta) < 2.4 \
                and lepton1_reliso < 0.15 \
                and lepton2_pt > 10 and lepton2_reliso < 0.15 \
                and dilepton1_mass > 12 \
                and (dilepton1_mass < 80 or dilepton1_mass > 102) \
                and lepton1_q != lepton2_q'
    elif selection == 'emu':
        cut += ' and lepton2_pt > 15 \
                and dilepton1_mass > 12 \
                and lepton1_q != lepton2_q'
    elif selection == 'mutau':
        cut += ' and lepton1_pt > 25 \
                     and dilepton1_mass > 12 \
                     and lepton1_q != lepton2_q' 
    elif selection == 'etau':
        cut += ' and lepton1_pt > 30 \
                 and dilepton1_mass > 12 \
                 and lepton1_q != lepton2_q' 


    ### Get dataframes with features for each of the datasets ###
    data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                  dataset_names = datasets,
                                  selection     = selection,
                                  period        = period,
                                  scale         = lumi,
                                  cuts          = cut
                                 )
    table = data_manager.print_yields(dataset_names=bg_labels+['data'], conditions=['n_bjets >= 1', 'n_bjets >= 2'])
    table.to_csv(f'data/yields_{selection}.csv')

    ### Loop over features and make the plots ###
    output_path = f'plots/overlays/{selection}_{period}'
    plot_manager = pt.PlotManager(data_manager,
                                  features       = features,
                                  stack_labels   = bg_labels,
                                  overlay_labels = signal_labels,
                                  top_overlay    = False,
                                  output_path    = output_path,
                                  file_ext       = 'png'
                                 )

    pt.make_directory(output_path, clear=True)
    plot_manager.make_overlays(features, plot_data, do_ratio=True, overlay_style='errorbar')

    if selection == 'mutau':
        conditions = [
                      'gen_cat == 15', 'gen_cat == 17', 'gen_cat == 8', 'gen_cat == 19',
                      'gen_cat != 15 and gen_cat != 17 and gen_cat != 8 and gen_cat != 19'
                     ]
        legend_labels = [
                         r'V+jets',
                         r'$\sf t\bar{t}/tW\rightarrow other$',
                         r'$\sf t\bar{t}/tW\rightarrow \tau_{\mu} + h$',
                         r'$\sf t\bar{t}/tW\rightarrow \tau_{\mu} + \tau_{h}$', 
                         r'$\sf t\bar{t}/tW\rightarrow \mu + h$', 
                         r'$\sf t\bar{t}/tW\rightarrow \mu + \tau_{h}$', 
                         ]
        bg_labels = ['zjets', 'wjets']
    elif selection == 'mumu':
        conditions = [
                      'gen_cat == 2', 'gen_cat == 14', 'gen_cat == 5', 
                      'gen_cat != 2 and gen_cat != 5 and gen_cat != 14',
                     ]
        legend_labels = [
                         r'V+jets',
                         r'$\sf t\bar{t}/tW\rightarrow other$',
                         r'$\sf t\bar{t}/tW\rightarrow \tau_{\mu} + \tau_{\mu}$', 
                         r'$\sf t\bar{t}/tW\rightarrow \mu + \tau_{\mu}$', 
                         r'$\sf t\bar{t}/tW\rightarrow \mu + \mu$', 
                         ]
        bg_labels = ['zjets', 'wjets']
    elif selection == 'emu':
        conditions = [
                      'gen_cat == 3', 'gen_cat == 11', 'gen_cat == 13', 
                      'gen_cat != 3 and gen_cat != 11 and gen_cat != 13'
                     ]
        legend_labels = [
                         r'V+jets',
                         r'$\sf t\bar{t}/tW\rightarrow other$',
                         r'$\sf t\bar{t}/tW\rightarrow \mu + \tau_{e}$', 
                         r'$\sf t\bar{t}/tW\rightarrow e + \tau_{\mu}$', 
                         r'$\sf t\bar{t}/tW\rightarrow e + \mu$', 
                         ]
        bg_labels = ['zjets', 'wjets']
    elif selection == 'mu4j':
        conditions = [
                      'gen_cat == 17', 'gen_cat == 19', 'gen_cat == 15', 'gen_cat == 3', 
                      'gen_cat != 3 and gen_cat != 15 and gen_cat != 17 and gen_cat != 19'
                     ]
        legend_labels = [
                         'W+jets, fakes',
                         r'$\sf t\bar{t}/tW\rightarrow other$',
                         r'$\sf t\bar{t}/tW\rightarrow e + \mu$', 
                         r'$\sf t\bar{t}/tW\rightarrow \mu + \tau_{h}$', 
                         r'$\sf t\bar{t}/tW\rightarrow \tau_{\mu} + h$', 
                         r'$\sf t\bar{t}/tW\rightarrow \mu + h$', 
                         ]
        bg_labels = ['wjets', 'fakes']

    output_path += '_split'
    plot_manager._output_path = output_path
    #plot_manager.make_conditional_overlays(['ttbar', 't'], features, conditions, 
    #                                       bg_labels  = bg_labels,
    #                                       legend     = legend_labels,
    #                                       do_stacked = True,
    #                                       do_data    = True
    #                                      )
                                       
