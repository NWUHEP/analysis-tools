

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import scripts.plot_tools as pt
import scripts.fit_helpers as fh
from scripts.blt_reader import jec_source_names, btag_source_names

def conditional_scaling(df, bins, scale, mask, feature, type='var'):
    '''
    Generates morphing templates based on systematic assigned to subset of the data.
    '''

    if type == 'var':
        df.loc[mask, feature] *= 1 + scale
        h_up, _   = np.histogram(df[feature], bins=bins, weights=df.weight)
        df.loc[mask, feature] *= (1 - scale)/(1 + scale)
        h_down, _ = np.histogram(df[feature], bins=bins, weights=df.weight)
        df.loc[mask, feature] /= (1 - scale)
    elif type == 'weight':
        df.loc[mask, 'weight'] *= 1 + scale
        h_up, _   = np.histogram(df[feature], bins=bins, weights=df.weight)
        df.loc[mask, 'weight'] *= (1 - scale)/(1 + scale)
        h_down, _ = np.histogram(df[feature], bins=bins, weights=df.weight)
        df.loc[mask, 'weight'] /= (1 - scale)

    return h_up, h_down

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
    w_up       = df_tmp.weight*(sf_up(df_tmp.n_pu)/sf_nominal(df_tmp.n_pu))
    w_down     = df_tmp.weight*(sf_down(df_tmp.n_pu)/sf_nominal(df_tmp.n_pu))
    h_up, _      = np.histogram(df_tmp[feature], bins=bins, weights=w_up)
    h_down, _    = np.histogram(df_tmp[feature], bins=bins, weights=w_down)
    
    return h_up, h_down
    
def les_morph(df, feature, bins, scale):
    '''
    lepton energy scale morphing
    '''

    h_up, _      = np.histogram((1+scale)*df[feature], bins=bins, weights=df.weight)
    h_down, _    = np.histogram((1-scale)*df[feature], bins=bins, weights=df.weight)

    return h_up, h_down

def jet_scale(df, feature, bins, sys_type, jet_cut):
    '''
    Jet systematics are treated as shape systematics, but mostly vary depending
    on the jet/b tag multiplicity.  Nonetheless, it's easier to account for
    them as a shape systematic.  N.B.: this variation is averaged over all bins
    in a distribution to avoid overconstraining.
    '''

    if sys_type in ['ctag', 'mistag'] or 'btag' in sys_type:
        up_condition   = jet_cut.replace('n_bjets', f'n_bjets_{sys_type}_up')
        down_condition = jet_cut.replace('n_bjets', f'n_bjets_{sys_type}_down')
    else:
        up_condition   = jet_cut.replace('n_jets', f'n_jets_{sys_type}_up')
        up_condition   = up_condition.replace('n_bjets', f'n_bjets_{sys_type}_up')

        down_condition = jet_cut.replace('n_jets', f'n_jets_{sys_type}_down')
        down_condition = down_condition.replace('n_bjets', f'n_bjets_{sys_type}_down')

    h_nominal, _ = np.histogram(df.query(jet_cut)[feature], 
                                bins=bins, 
                                weights=df.query(jet_cut).weight
                                )

    h_up, _   = np.histogram(df.query(up_condition)[feature], 
                             bins=bins, 
                             weights=df.query(up_condition).weight
                             )

    h_down, _ = np.histogram(df.query(down_condition)[feature],
                             bins=bins, 
                             weights=df.query(down_condition).weight
                             )

    # average over bin-by-bin variations for now
    h_up   = (h_up.sum()/h_nominal.sum()) * h_nominal
    h_down = (h_down.sum()/h_nominal.sum()) * h_nominal
    
    return h_up, h_down

class SystematicTemplateGenerator():
    def __init__(self, selection, feature, binning, h_nominal):
        self._selection = selection
        self._feature   = feature
        self._binning   = binning
        self._h         = h_nominal
        self._df_sys    = pd.DataFrame(dict(bins=binning[:-1]))

    def get_syst_dataframe(self):
        return self._df_sys.set_index('bins')

    def jes_systematics(self, df, jet_cut):
        '''
        Generates morpthing templates for: 
           * jet energy scale/resolution

        Parameters:
        ===========
        df: dataframe for target dataset without the jet cuts applied
        '''

        jet_syst_list = [f'jes_{n}' for n in jec_source_names]
        jet_syst_list += ['jer']
        for syst_type in jet_syst_list:
            h_up, h_down = jet_scale(df, self._feature, self._binning, syst_type, jet_cut)
            self._df_sys[f'{syst_type}_up'], self._df_sys[f'{syst_type}_down'] = h_up, h_down

    def btag_systematics(self, df, jet_cut):
        '''
        Generates morpthing templates for: 
           * b tag efficiency systematics

        Parameters:
        ===========
        df: dataframe for target dataset without the jet cuts applied
        '''

        btag_syst_list = [f'btag_{n}' for n in btag_source_names]
        btag_syst_list += ['ctag', 'mistag']
        for syst_type in btag_syst_list:
            h_up, h_down = jet_scale(df, self._feature, self._binning, syst_type, jet_cut)
            self._df_sys[f'{syst_type}_up'], self._df_sys[f'{syst_type}_down'] = h_up, h_down


    def electron_systematics(self, df):
        '''
        Generates shape templates for electron id and reco efficiency scale factors.
        '''
        bins = self._binning
        feature = self._feature

        if self._selection == 'ee':

            ## reco scale factor
            w_nominal = df.weight/(df['lepton1_reco_weight']*df['lepton2_reco_weight'])
            w_up      = w_nominal*(df['lepton1_reco_weight'] + np.sqrt(df['lepton1_reco_var']))*(df['lepton2_reco_weight'] + np.sqrt(df['lepton2_reco_var']))
            w_down    = w_nominal*(df['lepton1_reco_weight'] - np.sqrt(df['lepton1_reco_var']))*(df['lepton2_reco_weight'] - np.sqrt(df['lepton2_reco_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)

            #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
            #self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = y_up, y_down
            self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = h_up, h_down

            ## id/iso scale factor
            w_nominal = df.weight/(df['lepton1_id_weight']*df['lepton2_id_weight'])
            w_up      = w_nominal*(df['lepton1_id_weight'] + np.sqrt(df['lepton1_id_var']))*(df['lepton2_id_weight'] + np.sqrt(df['lepton2_id_var']))
            w_down    = w_nominal*(df['lepton1_id_weight'] - np.sqrt(df['lepton1_id_var']))*(df['lepton2_id_weight'] - np.sqrt(df['lepton2_id_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)

            #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
            #self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = y_up, y_down
            self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = h_up, h_down

        elif self._selection in ['etau', 'e4j']:

            ## reco scale factor
            w_nominal = df.weight/df['lepton1_reco_weight']
            w_up      = w_nominal*(df['lepton1_reco_weight'] + np.sqrt(df['lepton1_reco_var']))
            w_down    = w_nominal*(df['lepton1_reco_weight'] - np.sqrt(df['lepton1_reco_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)

            #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
            #self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = y_up, y_down
            self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = h_up, h_down

            ## id/iso scale factor
            w_nominal = df.weight/df['lepton1_id_weight']
            w_up      = w_nominal*(df['lepton1_id_weight'] + np.sqrt(df['lepton1_id_var']))
            w_down    = w_nominal*(df['lepton1_id_weight'] - np.sqrt(df['lepton1_id_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)

            #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
            #self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = y_up, y_down
            self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = h_up, h_down

        elif self._selection == 'emu':

            ## reco scale factor
            w_nominal = df.weight/df['lepton2_reco_weight']
            w_up      = w_nominal*(df['lepton2_reco_weight'] + np.sqrt(df['lepton2_reco_var']))
            w_down    = w_nominal*(df['lepton2_reco_weight'] - np.sqrt(df['lepton2_reco_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)

            #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
            #self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = y_up, y_down
            self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = h_up, h_down

            ## id/iso scale factor
            w_nominal = df.weight/df['lepton2_id_weight']
            w_up      = w_nominal*(df['lepton2_id_weight'] + np.sqrt(df['lepton2_id_var']))
            w_down    = w_nominal*(df['lepton2_id_weight'] - np.sqrt(df['lepton2_id_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)

            #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
            #self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = y_up, y_down
            self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = h_up, h_down

        ## electron energy scale
        scale = 0.002 # need reference
        if self._selection in ['ee', 'e4j']:
            h_up, _      = np.histogram((1+scale)*df[feature], bins=bins, weights=df.weight)
            h_down, _    = np.histogram((1-scale)*df[feature], bins=bins, weights=df.weight)

        if self._selection == 'emu':
            mask = abs(df.trailing_lepton_flavor) == 11
            df.loc[mask, 'trailing_lepton_pt'] *= 1 + scale
            h_up, _   = np.histogram(df.trailing_lepton_pt, bins=bins, weights=df.weight)
            df.loc[mask, 'trailing_lepton_pt'] *= (1 - scale)/(1 + scale)
            h_down, _ = np.histogram(df.trailing_lepton_pt, bins=bins, weights=df.weight)
            df.loc[mask, 'trailing_lepton_pt'] /= (1 - scale)

        self._df_sys['escale_e_up'], self._df_sys['escale_e_down'] = h_up, h_down

        return

    def muon_systematics(self, df):
        '''
        Generates shape templates for muon id and reco efficiency scale factors.
        '''

        bins = self._binning
        feature = self._feature

        ## id scale factor
        if self._selection == 'mumu':
            w_nominal = df.weight/(df['lepton1_id_weight']*df['lepton2_id_weight'])
            w_up      = w_nominal*(df['lepton1_id_weight'] + np.sqrt(df['lepton1_id_var']))*(df['lepton2_id_weight'] + np.sqrt(df['lepton2_id_var']))
            w_down    = w_nominal*(df['lepton1_id_weight'] - np.sqrt(df['lepton1_id_var']))*(df['lepton2_id_weight'] - np.sqrt(df['lepton2_id_var']))
        elif self._selection in ['mutau', 'mu4j', 'emu']:
            w_nominal = df.weight/df['lepton1_id_weight']
            w_up      = w_nominal*(df['lepton1_id_weight'] + np.sqrt(df['lepton1_id_var']))
            w_down    = w_nominal*(df['lepton1_id_weight'] - np.sqrt(df['lepton1_id_var']))

        h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
        h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)

        #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
        self._df_sys['eff_id_mu_up'], self._df_sys['eff_id_mu_down'] = h_up, h_down

        ## iso scale factor (called "reco" in ntuples)
        if self._selection == 'mumu':
            w_nominal = df.weight/(df['lepton1_reco_weight']*df['lepton2_reco_weight'])
            w_up      = w_nominal*(df['lepton1_reco_weight'] + np.sqrt(df['lepton1_reco_var']))*(df['lepton2_reco_weight'] + np.sqrt(df['lepton2_reco_var']))
            w_down    = w_nominal*(df['lepton1_reco_weight'] - np.sqrt(df['lepton1_reco_var']))*(df['lepton2_reco_weight'] - np.sqrt(df['lepton2_reco_var']))
        elif self._selection in ['mutau', 'mu4j', 'emu']:
            w_nominal = df.weight/df['lepton1_reco_weight']
            w_up      = w_nominal*(df['lepton1_reco_weight'] + np.sqrt(df['lepton1_reco_var']))
            w_down    = w_nominal*(df['lepton1_reco_weight'] - np.sqrt(df['lepton1_reco_var']))

        h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
        h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)

        #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
        self._df_sys['eff_iso_mu_up'], self._df_sys['eff_iso_mu_down'] = h_up, h_down

        ## muon energy scale
        scale = 0.002 # need reference
        if self._selection in ['mumu', 'mu4j']:
            h_up, _      = np.histogram((1+scale)*df[feature], bins=bins, weights=df.weight)
            h_down, _    = np.histogram((1-scale)*df[feature], bins=bins, weights=df.weight)

        elif self._selection == 'emu':
            mask = abs(df.trailing_lepton_flavor) == 13 # only apply when muon is the trailing pt lepton
            df.loc[mask, 'trailing_lepton_pt'] *= 1 + scale
            h_up, _   = np.histogram(df.trailing_lepton_pt, bins=bins, weights=df.weight)
            df.loc[mask, 'trailing_lepton_pt'] *= (1 - scale)/(1 + scale)
            h_down, _ = np.histogram(df.trailing_lepton_pt, bins=bins, weights=df.weight)
            df.loc[mask, 'trailing_lepton_pt'] /= (1 - scale)

        self._df_sys['escale_mu_up'], self._df_sys['escale_mu_down'] = h_up, h_down

        return

    def tau_j_misid_systematics(self, df):
        '''
        Generates morphing templates for tau mis ID as a function of tau pt.
        Binning is {20, 25, 25, 30, 40, 70, inf}.

        Parameters:
        ===========
        df: dataframe for target dataset with
        '''
        
        # jet->tau misid systematic
        pt_bins = [20, 25, 30, 40, 50, 65, np.inf]
        sigma   = [0.055, 0.046, 0.0434, 0.041, 0.0448, 0.0418] # statistical only
        for ipt, pt_bin in enumerate(pt_bins[:-1]):
            mask = (df.lepton2_pt > pt_bin) & (df.lepton2_pt < pt_bins[ipt+1])
            h_up, h_down = conditional_scaling(df, self._binning, sigma[ipt], mask, 'lepton2_pt', type='weight')
            self._df_sys[f'misid_tau_{ipt}_up'], self._df_sys[f'misid_tau_{ipt}_down'] = h_up, h_down

        # tau misid flavor systematic
        tau_misid_err = 0.05
        h_up, _   = np.histogram(df.lepton2_pt, bins=self._binning, weights=df.weight*(1 + tau_misid_err))
        h_down, _ = np.histogram(df.lepton2_pt, bins=self._binning, weights=df.weight*(1 - tau_misid_err))
        self._df_sys[f'misid_tau_h_up'], self._df_sys[f'misid_tau_h_down'] = h_up, h_down

        return

    def tau_e_misid_systematics(self, df):
        '''
        Generates morphing templates for an electron misID'd as a hadronic tau. 
        Parameters:
        ===========
        df: dataframe for target dataset with
        '''
        
        # tau misid flavor systematic
        tau_misid_err = 0.10
        h_up, _   = np.histogram(df.lepton2_pt, bins=self._binning, weights=df.weight*(1 + tau_misid_err))
        h_down, _ = np.histogram(df.lepton2_pt, bins=self._binning, weights=df.weight*(1 - tau_misid_err))
        self._df_sys[f'misid_tau_e_up'], self._df_sys[f'misid_tau_e_down'] = h_up, h_down

        return

    def tau_systematics(self, df):
        '''
        Systematics for correctly identified taus.  Just energy scale currently.

        Parameters:
        ===========
        df: dataframe for target dataset
        '''

        # tau id efficiency systematic
        pt_bins = [20, 25, 30, 40, 50, 65, np.inf]
        sigma   = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05] # statistical only
        for ipt, pt_bin in enumerate(pt_bins[:-1]):
            mask = (df.lepton2_pt > pt_bin) & (df.lepton2_pt < pt_bins[ipt+1])
            h_up, h_down = conditional_scaling(df, self._binning, sigma[ipt], mask, 'lepton2_pt', type='weight')
            self._df_sys[f'eff_tau_{ipt}_up'], self._df_sys[f'eff_tau_{ipt}_down'] = h_up, h_down

        # tau misid systematic
        tau_id_err = 0.05
        h_up, _   = np.histogram(df.lepton2_pt, bins=self._binning, weights=df.weight*(1 + tau_id_err))
        h_down, _ = np.histogram(df.lepton2_pt, bins=self._binning, weights=df.weight*(1 - tau_id_err))
        self._df_sys[f'eff_tau_up'], self._df_sys[f'eff_tau_down'] = h_up, h_down

        ## tau energy scale
        for decay_mode in [0, 1, 10]:
            h_up, h_down = conditional_scaling(df, self._binning, 0.012, df.tau_decay_mode == decay_mode, 'lepton2_pt')

            # a little hack to deal with missing data with pt < 20 GeV
            h_up[0] = 1.0025*self._h[0]
            h_down[0] = .9975*self._h[0]

            #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
            #self._df_sys[f'escale_tau_{decay_mode}_up'], self._df_sys[f'escale_tau_{decay_mode}_down'] = y_up, y_down
            self._df_sys[f'escale_tau_{decay_mode}_up'], self._df_sys[f'escale_tau_{decay_mode}_down'] = h_up, h_down

        return

    def misc_systematics(self, df):
        '''
        Generates templates for:
           * pileup

        Parameters:
        ===========
        df: dataframe for target dataset with
       '''
        bins = self._binning
        feature = self._feature

        # pileup
        h_up, h_down = pileup_morph(df, feature, bins)
        self._df_sys['pileup_up'], self._df_sys['pileup_down'] = h_up, h_down

        return

    def theory_systematics(self, df, label, njets, dm_syst=None):
        '''
        Generates templates for theory systematics:
           * QCD scale
           * alpha_s
           * UE
           * ME-PS
           * PDF

        Parameters:
        ===========
        df: dataframe for target dataset 
        label: name of dataset being run over
        njets: number of jets required
        dm_syst: data manager with dedicated ttbar samples for non-weight based systematics
        '''

        bins = self._binning
        feature = self._feature

        # PS variations (need to revisit this)
        #for sys_type in ['isr', 'fsr', 'hdamp', 'tune']:
        #    df_up     = dm.get_dataframe(f'ttbar_{sys_type}up', cut)
        #    df_down   = dm.get_dataframe(f'ttbar_{sys_type}down', cut)

        #    h_up, _   = np.histogram(df_up[feature], bins=bins, weights=df_up.weight)
        #    h_down, _ = np.histogram(df_down[feature], bins=bins, weights=df_down.weight)

        # pdf variations
        pdf_err = 0.01
        h_up, _   = np.histogram(df[feature], bins=bins, weights=df.weight*(1 + pdf_err))
        h_down, _ = np.histogram(df[feature], bins=bins, weights=df.weight*(1 - pdf_err))
        self._df_sys[f'xs_{label}_pdf_up'], self._df_sys[f'xs_{label}_pdf_down'] = h_up, h_down

        # alpha_s variations
        alpha_s_err = (df['alpha_s_err'] - df['qcd_weight_nominal'])/df['qcd_weight_nominal']
        h_up, _   = np.histogram(df[feature], bins=bins, weights=df.weight*(1 + alpha_s_err))
        h_down, _ = np.histogram(df[feature], bins=bins, weights=df.weight*(1 - alpha_s_err))
        self._df_sys[f'xs_{label}_alpha_s_up'], self._df_sys[f'xs_{label}_alpha_s_down'] = h_up, h_down

        # qcd scale variations
        qcd_vars = [
                    'qcd_weight_nom_up', 'qcd_weight_nom_down', 'qcd_weight_up_nom', 
                    'qcd_weight_down_nom', 'qcd_weight_up_up', 'qcd_weight_down_down'
                    ]
        h_qcd_vars = []
        for qcd_var in qcd_vars:
            dqcd = (df[qcd_var] - df['qcd_weight_nominal'])/df['qcd_weight_nominal']
            h_var, _ = np.histogram(df[feature], bins=bins, weights=df.weight*(1 + dqcd))
            h_qcd_vars.append(h_var)
        
        h_qcd_vars = np.array(h_qcd_vars)
        h_up, h_down = np.max(h_qcd_vars, axis=0), np.min(h_qcd_vars, axis=0)
        if label == 'ttbar': 
            self._df_sys[f'xs_{label}_qcd_scale_up'], self._df_sys[f'xs_{label}_qcd_scale_down'] = h_up, h_down
        else: # split processes where ME influences jet multiplicity (Z, W, etc.)
            self._df_sys[f'xs_{label}_qcd_scale_{njets}_up'], self._df_sys[f'xs_{label}_qcd_scale_{njets}_down'] = h_up, h_down

        return

    def top_pt_systematics(self, df):
        '''
        Variation from reweighting the top quark pt spectrum in ttbar events
        (https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting).  The
        nominal case is with the weights, down variation is no weight, up
        variation is twice the nominal weight.

        Parameters:
        ===========
        df: dataframe for dataset
        cut: jet category
        '''

        w_up      = (df.weight/df.top_pt_weight)*(1 + 2*(df.top_pt_weight - 1))
        w_down    = (df.weight/df.top_pt_weight)
        h_up, _   = np.histogram(df[self._feature], bins=self._binning, weights=w_up*(df.weight.sum()/w_up.sum()))
        h_down, _ = np.histogram(df[self._feature], bins=self._binning, weights=w_down*(df.weight.sum()/w_down.sum()))
        self._df_sys['top_pt_up'], self._df_sys['top_pt_down'] = h_up, h_down

