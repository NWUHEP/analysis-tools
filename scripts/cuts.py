#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import python.plot_tools as pt

if __name__ == '__main__':

    if len(sys.argv) > 3:
        indir   = sys.argv[1]
        period  = sys.argv[3]
    else:
        indir   = 'data/flatuples/4l_2016'
        period  = '2016'

    do_sync     = True
    output_path = 'data/sync/{0}'.format(period)

    if period == '2012':
        datasets    = ['muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D'] 
    elif period == '2016':
        datasets    = [
                       #'zjets_m-50',  'zjets_m-10to50',
                       #'z1jets_m-50', 'z1jets_m-10to50',
                       #'z2jets_m-50', 'z2jets_m-10to50',
                       #'z3jets_m-50', 'z3jets_m-10to50',
                       #'z4jets_m-50', 'z4jets_m-10to50',

                       'muon_2016B', 'muon_2016C', 'muon_2016D', 
                       'muon_2016E', 'muon_2016F', 'muon_2016G', 'muon_2016H'
                      ] 

    data_manager = pt.DataManager(input_dir     = indir,
                                  dataset_names = datasets,
                                  period        = period,
                                  scale         = 36e3,
                                  selection     = 'mumu'
                                 )

    # conditions for querying non-zero jet/b jet events
    cuts = [  
            '(lepton1_pt > 25 and abs(lepton1_eta) < 2.1 \
              and lepton2_pt > 25 and abs(lepton2_eta) < 2.1 \
              and lepton1_q != lepton2_q \
              and 12 < dilepton_mass < 70 \
              and trigger_status)',
           ]
    #cuts.append('26 < dilepton_mass < 32')

