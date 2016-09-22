#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#def stack_by_group(df, by, var, histtype='stepfilled', bins=30., range=None):
#
#    g = df.groupby(by)
#    data = [g.get_group(group)[var] for group in g.groups.keys()] 
#    plt.hist(data, histtype=histtype, bins=bins, stacked=True)

def save_histograms(df, cat, level, prefix, columns=None, period='2016'):
    if columns:
        df[columns].hist(bins=40, histtype='step')
    else:
        df.hist(bins=40, histtype='step')

    plt.savefig('plots/amumu_cuts/{0}_{1}_cut{2}_{3}.pdf'.format(prefix, cat, level, period))
    plt.close()

if __name__ == '__main__':

    if len(sys.argv) > 3:
        infile  = sys.argv[1]
        cat     = sys.argv[2]
        period  = sys.argv[3]
    else:
        infile  = 'data/ntuple_dimuon.csv'
        cat     = '1b1f'
        period  = 2012

    # split features into categories for improved plotting
    muon_features = [
                     'muon1_pt', 'muon1_eta', 'muon1_phi', 'muon1_iso', 
                     'muon2_pt', 'muon2_eta', 'muon2_phi', 'muon2_iso', 
                    ]
    dimuon_features = [
                     'muon_delta_eta', 'muon_delta_phi', 'muon_delta_r',
                     'dimuon_mass', 'dimuon_pt', 'dimuon_eta', 'dimuon_phi'
                     ]
    four_body_features = ['delta_phi', 'delta_eta', 'four_body_mass']
    jet_features       = ['jet_pt', 'jet_eta', 'jet_phi', 'n_jets', 'n_fwdjets']
    bjet_features      = ['bjet_pt', 'bjet_eta', 'bjet_phi', 'bjet_d0', 'n_bjets']
    dijet_features     = ['dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi']
    met_features       = ['met_mag', 'met_phi']
    misc               = ['event_number', 'run_number', 'lumi']

    # conditions for querying non-zero jet/b jet events
    jet_condition   = 'n_jets + n_fwdjets > 0'
    bjet_condition  = 'n_bjets > 0'
    dijet_condition = '(n_jets + n_fwdjets) > 0 and (n_bjets > 0)'

    # Add columns telling us whether has passed each analysis cut
    data = pd.read_csv(infile)
    cut_list = []
    data['cut1'] = data['n_jets'] + data['n_bjets'] > 0
    data['cut2'] = data['n_bjets'] > 0
    data['cut3'] = data['n_bjets'] == 1
    if cat == '1b1f':
        data['cut4'] = data['n_jets'] == 0 
        data['cut5'] = data['n_fwdjets'] > 0
        cut_list = ['cut1', 'cut2', 'cut3', 'cut4', 'cut5']

    elif cat == '1b1c':
        data['cut4'] = data['n_jets'] == 1 
        data['cut5'] = data['met_mag'] < 40
        data['cut6'] = np.abs(data['delta_phi']) > 2.5
        cut_list = ['cut1', 'cut2', 'cut3', 'cut4', 'cut5', 'cut6']

    # Save some rough histograms for sanity checks and print yields for each cut level
    save_histograms(data                       , cat, 0, 'muon'     , muon_features)
    save_histograms(data                       , cat, 0, 'dimuon'   , dimuon_features)
    save_histograms(data                       , cat, 0, 'met'      , met_features)
    save_histograms(data.query(jet_condition)  , cat, 0, 'jet'      , jet_features)
    save_histograms(data.query(bjet_condition) , cat, 0, 'bjet'     , bjet_features)
    save_histograms(data.query(dijet_condition), cat, 0, 'dijet'    , dijet_features)
    save_histograms(data.query(dijet_condition), cat, 0, 'four_body', four_body_features)
    print 'cut 0: {0}'.format(data.shape[0])
    data[misc].to_csv('data/amumu_sync/event_list_cut0_{0}.csv'.format(period), index=False) 

    for level in xrange(1,len(cut_list)+1):
        cut_matrix = data[cut_list[:level]]
        data_cut   = data[cut_matrix.all(axis = 1)]

        # Save some rough histograms
        save_histograms(data_cut                       , cat, level, 'muon'     , muon_features)
        save_histograms(data_cut                       , cat, level, 'dimuon'   , dimuon_features)
        save_histograms(data_cut                       , cat, level, 'met'      , met_features)
        save_histograms(data_cut.query(jet_condition)  , cat, level, 'jet'      , jet_features)
        save_histograms(data_cut.query(bjet_condition) , cat, level, 'bjet'     , bjet_features)
        save_histograms(data_cut.query(dijet_condition), cat, level, 'dijet'    , dijet_features)
        save_histograms(data_cut.query(dijet_condition), cat, level, 'four_body', four_body_features)

        print 'cut {0}: {1}'.format(level, data_cut.shape[0])
        data_cut[misc].to_csv('data/amumu_sync/event_list_{0}_cut{1}_{2}.csv'.format(cat, level, period), index=False) 
        data_cut[['dimuon_mass']].to_csv('data/dimuon_mass_{0}_cut{1}_{2}.csv'.format(cat, level, period), index=False) 
