#!/usr/bin/env python

import argparse
from itertools import chain

import pandas as pd
#import matplotlib as mpl
#mpl.use('Agg')

import scripts.plot_tools as pt

if __name__ == '__main__':

    pt.set_default_style()

    # input arguments
    parser = argparse.ArgumentParser(description='Produce data/MC overlays')
    parser.add_argument('input',
                        help = 'specify input directory',
                        type = str
                        )
    parser.add_argument('-s', '--selection',
                        help = 'selection type',
                        default = 'mutau',
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
    sim_labels = ['diboson', 'wjets', 'zjets_alt', 't', 'ttbar']

    ### Get dataframes with features for each of the datasets ###
    input_dir = f'{args.input}/{args.selection}_{args.period}'
    data_manager = pt.DataManager(input_dir     = input_dir,
                                  dataset_names = [d for l in data_labels+sim_labels for d in pt.dataset_dict[l]],
                                  selection     = selection,
                                  period        = args.period,
                                  scale         = args.lumi,
                                  cuts          = 'lepton1_q == lepton2_q'
                                 )
    
    # combine ss data with ss simulation
    df_data = data_manager.get_dataframe('data')
    df_sim  = data_manager.get_dataframes(sim_labels, concat=True)

    if df_data.shape[0] == 0:
        print('No same-sign events in dataset.')
    else:
        df_sim.loc[:,'weight'] *= -1
        df_qcd = pd.concat([df_data, df_sim])
        df_qcd.loc[:,'lepton2_q'] = -1*df_qcd['lepton1_q']

        # save output and update event counts
        df_qcd.to_pickle(f'{input_dir}/ntuple_fakes_ss.pkl')

        ec = pd.read_csv(f'{input_dir}/event_counts.csv')
        ec['fakes_ss'] = 10*[1., ]
        ec.to_csv(f'{input_dir}/event_counts.csv')
