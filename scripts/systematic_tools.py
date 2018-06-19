
import pickle
import numpy as np
import pandas as pd

import scripts.plot_tools as pt


def pileup_morph(df, feature, bins):
    '''
    Generates templates for morphing of distributions due to pileup variance.
    '''

    from scipy.interpolate import interp1d

    pileup_file = open('data/pileup_sf.pkl', 'rb')
    pu_bins     = pickle.load(pileup_file)
    sf_nominal  = interp1d(pu_bins, pickle.load(pileup_file), kind='linear')
    sf_up       = interp1d(pu_bins, pickle.load(pileup_file), kind='linear')
    sf_down     = interp1d(pu_bins, pickle.load(pileup_file), kind='linear')

    df_tmp     = df.query(f'n_pu > {pu_bins.min()} and n_pu < {pu_bins.max()}')
    #w_up       = df_tmp.weight*(sf_up(df_tmp.n_pu)/df_tmp.pileup_weight)
    #w_down     = df_tmp.weight*(sf_down(df_tmp.n_pu)/df_tmp.pileup_weight)
    w_up       = df_tmp.weight*(sf_up(df_tmp.n_pu)/sf_nominal(df_tmp.n_pu))
    w_down     = df_tmp.weight*(sf_down(df_tmp.n_pu)/sf_nominal(df_tmp.n_pu))
    h_up, _      = np.histogram(df_tmp[feature], bins=bins, weights=w_up)
    h_down, _    = np.histogram(df_tmp[feature], bins=bins, weights=w_down)
    h_nominal, _ = np.histogram(df_tmp[feature], bins=bins, weights=df_tmp.weight)
    
    return h_up/h_nominal, h_down/h_nominal
    
def les_morph(df, feature, bins, scale):
    '''
    lepton energy scale morphing
    '''

    h_up, _      = np.histogram((1+scale)*df[feature], bins=bins, weights=df.weight)
    h_down, _    = np.histogram((1-scale)*df[feature], bins=bins, weights=df.weight)
    h_nominal, _ = np.histogram(df[feature], bins=bins, weights=df.weight)

    return h_up/h_nominal, h_down/h_nominal

def jet_scale(df, feature, bins, sys_type, jet_condition):
    '''
    Jet systematics are treated as shape systematics, but mostly vary depending
    on the jet/b tag multiplicity.  Nonetheless, it's easier to account for
    them as a shape systematic.
    '''

    # nominal histogram
    h_nominal, _ = np.histogram(df.query(jet_condition)[feature], bins=bins, weights=df.query(jet_condition).weight)

    # systematic up/down
    up_condition   = jet_condition.replace('n_bjets', f'n_bjets_{sys_type}_up')
    down_condition = jet_condition.replace('n_bjets', f'n_bjets_{sys_type}_down')
    if sys_type not in ['btag', 'mistag']:
        up_condition   = up_condition.replace('n_jets', f'n_jets_{sys_type}_up')
        down_condition = down_condition.replace('n_jets', f'n_jets_{sys_type}_down')

    h_up, _      = np.histogram(df.query(up_condition)[feature], bins=bins, weights=df.query(up_condition).weight)
    h_down, _    = np.histogram(df.query(down_condition)[feature], bins=bins, weights=df.query(down_condition).weight)

    return h_up/h_nominal, h_down/h_nominal
