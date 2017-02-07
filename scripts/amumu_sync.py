#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import nllfitter.plot_tools as pt

if __name__ == '__main__':

    if len(sys.argv) > 1:
        cat = sys.argv[1]
    else:
        cat = '1b1f'

    period      = '2016'
    output_path = 'data/sync/rereco_{0}'.format(period)
    datasets    = [
                   'muon_2016B', 'muon_2016C', 'muon_2016D', 
                   #'muon_2016E', 'muon_2016F', 'muon_2016G', 'muon_2016H'
                  ]

    dm_rereco = pt.DataManager(input_dir     = 'data/flatuples/mumu_rereco_2016',
                               dataset_names = datasets,
                               period        = period,
                               selection     = 'mumu'
                              )
    dm_prompt = pt.DataManager(input_dir     = 'data/flatuples/mumu_prompt_2016',
                               dataset_names = datasets,
                               period        = period,
                               selection     = 'mumu'
                              )

    # conditions for querying non-zero jet/b jet events
    cuts = [  
            '(lepton1_pt > 25 and abs(lepton1_eta) < 2.1 \
              and lepton2_pt > 25 and abs(lepton2_eta) < 2.1 \
              and lepton1_q != lepton2_q \
              and 12 < dilepton_mass < 70 \
              and trigger_status)',
            '(n_jets > 0 or n_bjets > 0)',
            'n_bjets > 0',
           ]
    cut_names = ['preselection', 'at least one central jet', 'at least one b jet']

    if cat == '1b1f':
        cuts.extend(['n_bjets == 1', 'n_jets == 0', 'n_fwdjets > 0'])
        cut_names.extend(['exactly one b jet', 
                          'no additional central jets', 
                          'at least one forward jet', 
                          ])
    elif cat == '1b1c':
        cuts.extend(['(n_bjets + n_jets == 2 and n_fwdjets == 0)', 
                     'met_mag < 40', 
                     'four_body_delta_phi > 2.5',
                    ])

        cut_names.extend(['at least two central jets, no foward jets', 
                          'MET < 40', 
                          'delta_phi(mumu, jj) > 2.5', 
                          ])
    elif cat == 'combined':
        cuts.extend(['((n_bjets == 1 and n_jets == 0 and n_fwdjets > 0) \
                     or (n_jets == 1 and n_fwdjets == 0 \
                     and met_mag < 40 and four_body_delta_phi > 2.5))',
                    ])
    else:
        print 'what are you doing, man!?'

    #cuts.append('dilepton_pt_over_m > 2')
    #cut_names.append('qt/M > 2')

    pt.make_directory(output_path, clear=False) 
    df_prompt = dm_prompt.get_dataframe('data', '')
    df_rereco = dm_rereco.get_dataframe('data', '')

    print cat
    print '|*cut* | *prompt* | *rereco* | *overlap*|'
    print '|---|---|---|---|'
    for i in range(len(cuts)):
        print '| {0}'.format(cut_names[i]),

        df1 = df_prompt.query(' and '.join(cuts[:i+1]))
        df1.to_csv('{0}/events_prompt_cut{1}_{2}.csv'.format(output_path, i, cat), index=False) 
        print ' | {0}'.format(df1.shape[0]),

        df2 = df_rereco.query(' and '.join(cuts[:i+1]))
        df2.to_csv('{0}/events_rereco_cut{1}_{2}.csv'.format(output_path, i, cat), index=False) 
        print ' | {0}'.format(df2.shape[0]),

        en    = df1.event_number
        en0   = df2.event_number
        mask  = en.apply(lambda x: x not in en0.values)
        mask0 = en0.apply(lambda x: x not in en.values)

        #print ''
        overlap = df1.shape[0] - df1[mask].shape[0]
        print ' | {0} ({1:.1%}) |'.format(overlap, float(overlap)/df1.shape[0])

        if i == 5:
            df1[mask][['run_number', 'lumi', 'event_number']].to_csv('data/sync/prompt_only_{0}.csv'.format(cat), index=False)
            df2[mask0][['run_number', 'lumi', 'event_number']].to_csv('data/sync/rereco_only_{0}.csv'.format(cat), index=False)
