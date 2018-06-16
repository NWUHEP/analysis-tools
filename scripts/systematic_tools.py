
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

def jet_scale(df, sys_type):
    '''
    jet systematics are treated as normalization systematics, but will vary
    depending on the jet/b tag multiplicity.
    '''
    h_nominal, b, _ = ax.hist(df.query('n_jets >= 2 and n_bjets >= 1')[feature], range=brange, bins=nbins, color='C1', linestyle='--', histtype='step')
    h_plus, _, _ = ax.hist(df.query(f'n_jets_{sys_type}_up >= 2 and n_bjets_{sys_type}_up >= 1')[feature], range=brange, bins=nbins, color='C0', histtype='step')
    h_minus, _, _ = ax.hist(df.query(f'n_jets_{sys_type}_down >= 2 and n_bjets_{sys_type}_down >= 1')[feature], range=brange, bins=nbins, color='C2', histtype='step')


