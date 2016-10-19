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
        indir   = 'data/flatuples/mumu_2012'
        cat     = '1b1f'
        period  = '2012'

    output_path = 'data/amumu_sync/{0}'.format(period)
    if period == '2012':
        datasets    = ['muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D'] 
    elif period == '2016':
        datasets    = ['muon_2016B', 'muon_2016C', 'muon_2016D'] 

    data_manager = pt.DataManager(input_dir     = indir,
                                  dataset_names = datasets,
                                  period        = period,
                                  selection     = 'mumu'
                                 )

    # conditions for querying non-zero jet/b jet events
    cuts = [
            '(lepton1_pt > 25 and abs(lepton1_eta) < 2.1 \
              and lepton2_pt > 25 and abs(lepton2_eta) < 2.1 \
              and lepton1_q != lepton2_q and 12 < dilepton_mass < 70)',
            '(n_jets > 0 or n_bjets > 0)',
            'n_bjets > 0',
            'n_bjets == 1',

           ]
    if cat == '1b1f':
        cuts.extend(['n_jets == 0', 'n_fwdjets > 0'])
    elif cat == '1b1c':
        cuts.extend(['(n_jets == 1 and n_fwdjets == 0)', 'met_mag < 40', 'four_body_delta_phi > 2.5'])
    else:
        print 'what are you doing, man!?'

    #cuts.append('25 < dilepton_mass < 32')

    pt.make_directory(output_path, clear=False) 
    for i in range(len(cuts)):
        df = data_manager.get_dataframe('data', ' and '.join(cuts[:i+1]))
        df.to_csv('{0}/cut{1}_{2}.csv'.format(output_path, i, cat), 
                  columns = ['event_number', 'run_number'],
                  index=False
                 ) 
        print 'cut {0}: {1}'.format(i, df.shape[0])

    if period == '2016':
        evt_columns = ['run_number', 'event_number']
        df0 = pd.read_csv('data/amumu_sync/Anton/cut_{0}.txt'.format(cat), header=None, sep=' ')
        df0.columns = evt_columns
        df0.event_number[df0.event_number < 0] = 2**32 + df0.event_number[df0.event_number < 0]

        en    = df.event_number
        en0   = df0.event_number
        mask  = en.apply(lambda x: x not in en0.values)
        mask0 = en0.apply(lambda x: x not in en.values)

        print 'Events in my dataset, but not in sync dataset:'
        print df[mask][evt_columns]
        print ''
        print 'Events in sync dataset, but not in my datset:'
        print df0[mask0]
