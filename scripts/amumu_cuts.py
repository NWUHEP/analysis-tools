#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import nllfitter.plot_tools as pt

if __name__ == '__main__':

    if len(sys.argv) > 3:
        indir   = sys.argv[1]
        cat     = sys.argv[2]
        period  = sys.argv[3]
    else:
        indir   = 'data/flatuples/mumu_2016'
        cat     = '1b1f'
        period  = '2016'

    do_sync     = True
    output_path = 'data/sync/{0}'.format(period)

    if period == '2012':
        datasets    = ['muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D'] 
    elif period == '2016':
        datasets    = [
                       #'muon_2016C'
                       'muon_2016B', 'muon_2016C', 'muon_2016D', 
                       'muon_2016E', 'muon_2016F', 'muon_2016G', 'muon_2016H'
                      ] 
        #datasets    = ['muon_2016C'] 

    data_manager = pt.DataManager(input_dir     = indir,
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
    if cat == '1b1f':
        cuts.extend(['n_bjets == 1', 'n_jets == 0', 'n_fwdjets > 0'])
    elif cat == '1b1c':
        cuts.extend(['(n_bjets + n_jets == 2 and n_fwdjets == 0)', 
                     'met_mag < 40', 
                     'four_body_delta_phi > 2.5'
                    ])
    elif cat == 'combined':
        cuts.extend(['((n_bjets == 1 and n_jets == 0 and n_fwdjets > 0) \
                     or (n_jets == 1 and n_fwdjets == 0 \
                     and met_mag < 40 and four_body_delta_phi > 2.5))',
                     'dilepton_pt_over_m > 2. and 125 < dilepton_b_mass < 190'
                    ])
    else:
        print 'what are you doing, man!?'

    #cuts.append('26 < dilepton_mass < 32')

    pt.make_directory(output_path, clear=False) 
    for i in range(len(cuts)):
        df = data_manager.get_dataframe('data', ' and '.join(cuts[:i+1]))
        df = df[['run_number', 'lumi', 'event_number', 'dilepton_mass']]
        df.to_csv('{0}/events_cut{1}_{2}.csv'.format(output_path, i, cat), 
                  #columns = ['event_number', 'run_number', 'lumi', 'dilepton_mass'],
                  index=False
                 ) 
        print 'cut {0}: {1}'.format(i, df.shape[0])

    if do_sync and period == '2016':
        evt_index = ['run_number', 'event_number']
        df.set_index(evt_index)
        df0 = pd.read_csv('data/sync/Olga/events_{0}.txt'.format(cat), header=None, sep=',')
        df0.columns = ['run_number', 'event_number', 'lumi', 'dilepton_mass']
        df0.event_number[df0.event_number < 0] = 2**32 + df0.event_number[df0.event_number < 0]
        df0.set_index(evt_index)

        #df0 = df0.query('272007 <= run_number <= 276283') # only consider BCD

        en    = df.event_number
        en0   = df0.event_number
        mask  = en.apply(lambda x: x not in en0.values)
        mask0 = en0.apply(lambda x: x not in en.values)

        print ''
        print '{0} events in my dataset, but not in sync dataset:'.format(df[mask].shape[0])
        print df[mask][['run_number', 'lumi', 'event_number', 'dilepton_mass']].to_string(index=False)
        print ''
        print '{0} events in sync dataset, but not in my dataset:'.format(df0[mask0].shape[0])
        print df0[mask0][['run_number', 'lumi', 'event_number', 'dilepton_mass']].to_string(index=False)

