#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scale_features(data):
    '''
    Scales features to lie between 0 and 1.
    '''

if __name__ == '__main__':

    data = pd.read_csv('data/test_1b1f.csv')
    features = [    
                'muon1_pt', 'muon1_eta', 'muon1_phi',
                'muon2_pt', 'muon2_eta', 'muon2_phi',
                'muon_delta_eta', 'muon_delta_phi', 'muon_deltaR',
                'dimuon_pt', 'dimuon_eta', 'dimuon_phi',
                'dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi',
                'delta_phi', #'delta_eta', 'four_body_mass',
                'bjet_pt', 'bjet_eta', 'bjet_phi', 'bjet_d0',
                'jet_pt', 'jet_eta', 'jet_phi', 'jet_d0', 
                'met_mag', 'met_phi',
                ]

    data = data[features]

