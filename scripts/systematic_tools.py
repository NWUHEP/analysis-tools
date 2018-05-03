
import pickle
import numpy as np
import pandas as pd

import scripts.plot_tools as pt


def pileup_morph(df, bins, selection):
    '''
    Generates templates for morphing of distributions due to pileup variance.
    '''

    from scipy.interpolate import interp1d

    pileup_file = open('data/pileup_sf.pkl', 'rb')
    pu_bins     = pickle.load(pileup_file)
    sf_nominal  = interp1d(bins, pickle.load(pileup_file), kind='linear')
    sf_up       = interp1d(bins, pickle.load(pileup_file), kind='linear')
    sf_down     = interp1d(bins, pickle.load(pileup_file), kind='linear')

    df = df.query(f'n_pu > {bins.min()} and n_pu < {bins.max()}')

    w_up, w_down  = (df.weight/df.pileup)*sf_up(df.n_pu), (df_weight/df.pileup_weight)*sf_down(df.n_pu)
    h_plus, _, _  = np.histogram(df.lepton2_pt, bins=bins, weights=w_up)
    h_minus, _, _ = np.histogram(df.lepton2_pt, bins=bins, weights=w_down)
    
