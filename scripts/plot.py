#!/usr/bin/env python

import argparse
import pandas as pd
from tqdm import tqdm
import scripts.plot_tools as pt

if __name__ == '__main__':

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
    model_labels = ['diboson', 'zjets_alt', 'wjets', 't', 'ttbar']

    if selection in ['mu4j', 'e4j']: 
        model_labels = ['fakes'] + model_labels
    elif selection in ['mutau', 'etau']:
        model_labels = ['fakes_ss'] + model_labels

    # data samples
    features = [
                #'lepton1_reco_weight', 'lepton2_reco_weight', 'trigger_weight', 
                #'pileup_weight', 'top_pt_weight', 'event_weight',
                #'gen_cat',

                'n_pv', 'n_muons', 'n_electrons', 'n_taus',
                'n_jets', 'n_fwdjets', 'n_bjets',
                'met_mag', 'met_phi', 'ht_mag', 'ht_phi',

                'lepton1_pt', 'lepton1_eta', 'lepton1_phi', 'lepton1_mt', 
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
    elif selection in ['etau', 'mutau']:
        cut = 'n_jets >= 0 and n_bjets >= 0'
    else:
        cut = 'n_jets >= 2 and n_bjets >= 0'
    cut += ' and ' + pt.cuts[selection]
            
    ### Get dataframes with features for each of the datasets ###
    output_path = f'plots/overlays/{selection}_{args.period}'
    pt.make_directory(output_path, clear=True)
    pt.set_default_style()
    data_manager = pt.DataManager(input_dir     = f'{args.input}/{args.selection}_{args.period}',
                                  dataset_names = [d for l in data_labels+model_labels for d in pt.dataset_dict[l]],
                                  selection     = selection,
                                  period        = args.period,
                                  scale         = args.lumi,
                                  cuts          = cut
                                 )
    jet_cuts = [cat_items.cut for cat, cat_items in pt.categories.items() if selection in cat_items.selections]
    table = data_manager.print_yields(dataset_names=['data'] + model_labels, conditions=jet_cuts, do_string=True)
    table.transpose().to_latex(f'{output_path}/yields_{selection}.tex', escape=False)
    table.transpose().to_csv(f'{output_path}/yields_{selection}.csv')

    ### Loop over features and make the plots ###
    plot_manager = pt.PlotManager(data_manager,
                                  features       = features,
                                  stack_labels   = model_labels,
                                  top_overlay    = False,
                                  output_path    = output_path,
                                  file_ext       = 'png'
                                 )

    #plot_manager.make_overlays(features, do_ratio=True, overlay_style='errorbar')

    ### conditional overlays
    decay_map = pd.read_csv('data/decay_map.csv').set_index('id')
    decay_map = decay_map.query(f'{selection} == 1')
    conditions = [f'gen_cat == {ix}' for ix in decay_map.index.values]
    else_condition = 'not (' + 'or '.join(conditions) + ')'
    conditions.append(else_condition)

    bg_labels = ['wjets', 'diboson', 'zjets_alt']
    if selection in ['mu4j', 'e4j']: 
        bg_labels = ['fakes'] + bg_labels
    elif selection in ['mutau', 'etau']:
        bg_labels = ['fakes_ss'] + bg_labels

    #inclusive_cut = 'n_bjets >= 1'
    if selection == 'emu':
        inclusive_cut = 'n_jets >= 2'
    elif selection in ['ee', 'mumu']:
        inclusive_cut = 'n_jets >= 2 and (dilepton1_mass < 81 or dilepton1_mass > 101)'
    elif selection in ['etau', 'mutau']:
        inclusive_cut = 'n_jets >= 0'
    else:
        inclusive_cut = 'n_jets >= 2 and n_bjets >= 1'

    colors = ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef']
    plot_manager.set_output_path(f'{output_path}/inclusive')
    plot_manager.make_conditional_overlays(features, ['ttbar', 't'], conditions,
                                           cut         = inclusive_cut,
                                           legend      = list(decay_map.fancy_label) + [r'$\sf t\bar{t}/tW\rightarrow other$'],
                                           c_colors    = colors[:len(conditions) - 1] + ['gray'],
                                           aux_labels  = bg_labels,
                                           do_ratio    = True,
                                           do_cms_text = True
                                          )

    for i, (category, cat_items) in enumerate(tqdm(pt.categories.items(),
                             desc       = 'plotting jet categories...',
                             unit_scale = True,
                             ncols      = 75,
                             )):

        if selection in cat_items.selections:
            plot_manager.set_output_path(f'{output_path}/{category}')
            plot_manager.make_conditional_overlays(features, ['ttbar', 't'], conditions,
                                                   cut        = cat_items.cut,
                                                   legend     = list(decay_map.fancy_label) + [r'$\sf t\bar{t}/tW\rightarrow other$'],
                                                   #c_colors   = list(decay_map.colors) + ['gray'],
                                                   c_colors   = colors[:len(conditions) - 1] + ['gray'],
                                                   aux_labels = bg_labels,
                                                   do_ratio   = True,
                                                   do_cms_text = True
                                                  )
