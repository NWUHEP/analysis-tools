

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import interp1d
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
#kernel_regression = sm.nonparametric.kernel_regression

import scripts.plot_tools as pt
import scripts.fit_helpers as fh
from scripts.blt_reader import jec_source_names, btag_source_names

def template_smoothing(x, h_nom, h_up, h_down, **kwargs):
    '''
    Smoothing to reduce impact from limited statistics in determining variation
    templates.  By default this will use the statsmodel implementation of
    Lowess.
    '''

    mask = h_nom > 0
    dh_up, dh_down = h_up - h_nom, h_down - h_nom
    dh_up   = lowess(dh_up, x, frac=0.5, return_sorted=False)
    dh_down = lowess(dh_down, x, frac=0.5, return_sorted=False)

    return h_nom + dh_up, h_nom + dh_down

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

    print(f'--{sys_type}--')
    print(f'--"{jet_cut}"--')
    print('up', h_up.sum())
    print('nominal', h_nominal.sum())
    print('down', h_down.sum())

    # average over bin-by-bin variations for now
    #h_up   = (h_up.sum()/h_nominal.sum()) * h_nominal
    #h_down = (h_down.sum()/h_nominal.sum()) * h_nominal

    print('up (averaged)', h_up.sum())
    print('down (averaged)', h_down.sum())
    
    return h_up, h_down

def ttbar_systematics(dm, df_syst, cut, decay_mode, feature, binning, smooth=None):
    '''
    Account for systematics due to modeling of ttbar.
    '''

    #print(f'--{decay_mode}--')
    syst_names = ['isr', 'fsr', 'hdamp', 'tune']
    for syst in syst_names:

        df_up     = dm.get_dataframe(f'ttbar_{syst}up', cut)
        df_down   = dm.get_dataframe(f'ttbar_{syst}down', cut)
        h_nominal, var_nominal = df_syst['val'].values, df_syst['var'].values

        h_up, _     = np.histogram(df_up[feature], bins=binning, weights=df_up.weight)
        var_up, _   = np.histogram(df_up[feature], bins=binning, weights=df_up.weight**2)
        h_down, _   = np.histogram(df_down[feature], bins=binning, weights=df_down.weight)
        var_down, _ = np.histogram(df_down[feature], bins=binning, weights=df_down.weight**2)

        if h_up.sum() == 0. and h_down.sum() == 0.:
            continue

        if syst == 'fsr':
            h_up   = h_nominal + (h_up - h_nominal)/np.sqrt(2)
            h_down = h_nominal + (h_down - h_nominal)/np.sqrt(2)

            if dm._selection in ['etau', 'mutau']:
                # corrections for FSR sample
                k_down, k_up = 1., 1.
                if decay_mode in [7, 8, 12, 15]: #real taus
                    k_down, k_up = 1.02, 0.96
                elif decay_mode in [16, 17, 18, 19]: #fake taus
                    k_down, k_up = 1.27, 0.72

                h_up /= k_up
                h_down /= k_down

        # smoothing: do LOWESS smoothing on the difference of histograms
        x = (binning[:-1] + binning[1:])/2
        h_up, h_down = template_smoothing(x, h_nominal, h_up, h_down)

        # symmetrizations (*shrugs*)
        h_diff = h_nominal - (h_up + h_down)/2
        h_up += h_diff
        h_down += h_diff

        df_syst[f'{syst}_up'], df_syst[f'{syst}_down'] = h_up, h_down

def energy_scale_morphing(bins, hist, scale):
    '''
    Generates morphing templates based on histogram input.

    Parameters:
    ===========
    bins: histogram bin edges
    hist: histogram bin content
    scale: energy scale variation in percent.  (Include pt-dependent values in the future)
    '''

    #convert data (drop last bin edge)
    x = bins[:-1]
    y = hist
    dx = bins[1:] - bins[:-1]

    # shift values up/down
    x_up, x_down    = x/(1 - scale), x/(1 + scale)
    dy_up, dy_down  = np.zeros(y.size), np.zeros(y.size)
    dy_up[1:]      += y[:-1]*abs(x_down[1:] - x[1:])/dx[:-1]
    dy_up[:-1]     -= y[:-1]*abs(x_down[1:] - x[1:])/dx[:-1]
    dy_down[:-1]   += y[1:]*abs(x_up[1:] - x[1:])/dx[1:]
    dy_down        -= y*abs(x_up - x)/dx

    # fix for first bin: symmetrize the uncertainties to account for lack of
    # knowledge of values below lower values
    dy_up[0] = -dy_down[0]

    y_up, y_down = y + dy_up/np.sqrt(2), y + dy_down/np.sqrt(2)

    return y_up, y_down


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
            w_up      = df['weight']*(1 + np.sqrt(df['lepton1_reco_var'])/df['lepton1_reco_weight'])*(1 + np.sqrt(df['lepton2_reco_var'])/df['lepton2_reco_weight'])
            w_down    = df['weight']*(1 - np.sqrt(df['lepton1_reco_var'])/df['lepton1_reco_weight'])*(1 - np.sqrt(df['lepton2_reco_var'])/df['lepton2_reco_weight'])
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up.values)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down.values)
            self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = h_up, h_down

            ## id/iso scale factor
            w_up      = df['weight']*(1 + np.sqrt(df['lepton1_id_var'])/df['lepton1_id_weight'])
            w_down    = df['weight']*(1 - np.sqrt(df['lepton1_id_var'])/df['lepton1_id_weight'])
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up.values)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down.values)
            self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = h_up, h_down

            # electron pt-dependent efficiency systematic
            pt_bins = [20, 25, 30, 40, 50, 60, np.inf]
            for ipt, pt_bin in enumerate(pt_bins[:-1]):
                mask = (df[feature] > pt_bin) & (df[feature] < pt_bins[ipt+1])
                scale = np.sqrt(df.loc[mask, 'lepton2_id_var'])/df.loc[mask, 'lepton2_id_weight']
                h_up, h_down = conditional_scaling(df, self._binning, scale, mask, feature, type='weight')
                self._df_sys[f'eff_e_{ipt}_up'], self._df_sys[f'eff_e_{ipt}_down'] = h_up, h_down

        
        elif self._selection in ['etau', 'e4j']:

            ## reco scale factor
            w_nominal = df.weight/df['lepton1_reco_weight']
            w_up      = w_nominal*(df['lepton1_reco_weight'] + np.sqrt(df['lepton1_reco_var']))
            w_down    = w_nominal*(df['lepton1_reco_weight'] - np.sqrt(df['lepton1_reco_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)
            self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = h_up, h_down

            if self._selection == 'e4j':
                # electron pt-dependent efficiency systematic
                pt_bins = [30, 35, 40, 45, 50, 60, np.inf]
                for ipt, pt_bin in enumerate(pt_bins[:-1]):
                    mask = (df[feature] > pt_bin) & (df[feature] < pt_bins[ipt+1])

                    scale = np.sqrt(df.loc[mask, 'lepton1_id_var'])/df.loc[mask, 'lepton1_id_weight']
                    h_up, h_down = conditional_scaling(df, self._binning, scale, mask, feature, type='weight')
                    self._df_sys[f'eff_e_{ipt}_up'], self._df_sys[f'eff_e_{ipt}_down'] = h_up, h_down

                    trigger_var = df.loc[mask, 'trigger_var'] + df.loc[mask, 'el_trigger_syst_tag']**2 + df.loc[mask, 'el_trigger_syst_probe']**2
                    scale = np.sqrt(trigger_var)/df.loc[mask, 'trigger_weight']
                    h_up, h_down = conditional_scaling(df, self._binning, scale, mask, feature, type='weight')
                    self._df_sys[f'trigger_e_{ipt}_up'], self._df_sys[f'trigger_e_{ipt}_down'] = h_up, h_down

            else:
                ## id/iso scale factor
                w_nominal = df.weight/df['lepton1_id_weight']
                w_up      = w_nominal*(df['lepton1_id_weight'] + np.sqrt(df['lepton1_id_var']))
                w_down    = w_nominal*(df['lepton1_id_weight'] - np.sqrt(df['lepton1_id_var']))
                h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
                h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)
                self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = h_up, h_down


        elif self._selection == 'emu':

            ## reco scale factor
            w_nominal = df.weight/df['lepton2_reco_weight']
            w_up      = w_nominal*(df['lepton2_reco_weight'] + np.sqrt(df['lepton2_reco_var']))
            w_down    = w_nominal*(df['lepton2_reco_weight'] - np.sqrt(df['lepton2_reco_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)
            self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = h_up, h_down

            ## id/iso scale factor 
            mask = (abs(df.lead_lepton_flavor) == 11)
            df.loc[mask, 'weight'] *= (df.loc[mask, 'lepton2_id_weight'] + np.sqrt(df.loc[mask, 'lepton2_id_var']))/df.loc[mask, 'lepton2_id_weight']
            h_up, _ = np.histogram(df[feature], bins=bins, weights=df['weight'])
            df.loc[mask, 'weight'] *= (df.loc[mask, 'lepton2_id_weight'] - np.sqrt(df.loc[mask, 'lepton2_id_var']))/(df.loc[mask, 'lepton2_id_weight'] + np.sqrt(df.loc[mask, 'lepton2_id_var']))
            h_down, _ = np.histogram(df[feature], bins=bins, weights=df['weight'])
            df.loc[mask, 'weight'] *= df.loc[mask, 'lepton2_id_weight']/(df.loc[mask, 'lepton2_id_weight'] - np.sqrt(df.loc[mask, 'lepton2_id_var']))
            self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = h_up, h_down

            # electron pt-dependent efficiency systematic
            pt_bins = [30, 35, 40, 45, 50, 60, np.inf]
            for ipt, pt_bin in enumerate(pt_bins[:-1]):
                mask = (df[feature] > pt_bin) & (df[feature] < pt_bins[ipt+1]) & (abs(df.trailing_lepton_flavor) == 11)
                scale = np.sqrt(df.loc[mask, 'lepton2_id_var'])/df.loc[mask, 'lepton2_id_weight']
                h_up, h_down = conditional_scaling(df, self._binning, scale, mask, feature, type='weight')
                self._df_sys[f'eff_e_{ipt}_up'], self._df_sys[f'eff_e_{ipt}_down'] = h_up, h_down

        if self._selection != 'e4j':
            ## tag systematic
            w_nominal = df.weight/df['trigger_weight']
            w_up      = w_nominal*(df['trigger_weight'] + df['el_trigger_syst_tag'])
            w_down    = w_nominal*(df['trigger_weight'] - df['el_trigger_syst_tag'])
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)
            self._df_sys['trigger_e_tag_up'], self._df_sys['trigger_e_tag_down'] = h_up, h_down

            ## probe systematic
            w_nominal = df.weight/df['trigger_weight']
            w_up      = w_nominal*(df['trigger_weight'] + df['el_trigger_syst_probe'])
            w_down    = w_nominal*(df['trigger_weight'] - df['el_trigger_syst_probe'])
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)
            self._df_sys['trigger_e_probe_up'], self._df_sys['trigger_e_probe_down'] = h_up, h_down

        ## electron energy scale
        scale = 0.001 # need reference
        if self._selection in ['ee', 'e4j']:
            h_up, h_down = energy_scale_morphing(bins, self._h, scale)

        elif self._selection == 'emu':
            e_mask          = abs(df.trailing_lepton_flavor) == 11
            pt_masked       = df.loc[e_mask, 'trailing_lepton_pt'].values
            pt_antimasked   = df.loc[~e_mask, 'trailing_lepton_pt'].values
            h_masked, _     = np.histogram(pt_masked, bins=bins, weights=df.weight[e_mask])
            h_antimasked, _ = np.histogram(pt_antimasked, bins=bins, weights=df.weight[~e_mask])

            h_up, h_down = energy_scale_morphing(bins, h_masked, scale)
            #print(h_masked, h_antimasked, h_up, h_down, sep='\n')
            #print('---------------------------')
            h_up += h_antimasked
            h_down += h_antimasked

        self._df_sys['escale_e_up'], self._df_sys['escale_e_down'] = h_up, h_down

    def muon_systematics(self, df):
        '''
        Generates shape templates for muon id and reco efficiency scale factors.
        '''

        bins = self._binning
        feature = self._feature
        pt_bins = [10, 15, 20, 25, 30, 40, 50, 65, np.inf]

        if self._selection == 'mumu':

            ## iso scale factor (called "reco" in ntuples)
            w_nominal = df.weight/(df['lepton1_reco_weight']*df['lepton2_reco_weight'])
            w_up      = w_nominal*(df['lepton1_reco_weight'] + np.sqrt(df['lepton1_reco_var']))*(df['lepton2_reco_weight'] + np.sqrt(df['lepton2_reco_var']))
            w_down    = w_nominal*(df['lepton1_reco_weight'] - np.sqrt(df['lepton1_reco_var']))*(df['lepton2_reco_weight'] - np.sqrt(df['lepton2_reco_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)
            self._df_sys['eff_iso_mu_up'], self._df_sys['eff_iso_mu_down'] = h_up, h_down

            ## id scale factor (lead muon)
            w_nominal = df.weight/df['lepton1_id_weight']
            w_up      = w_nominal*(df['lepton1_id_weight'] + np.sqrt(df['lepton1_id_var']))
            w_down    = w_nominal*(df['lepton1_id_weight'] - np.sqrt(df['lepton1_id_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)
            self._df_sys['eff_id_mu_up'], self._df_sys['eff_id_mu_down'] = h_up, h_down

            # muon pt-dependent efficiency systematic
            for ipt, pt_bin in enumerate(pt_bins[:-1]):
                mask = (df[feature] > pt_bin) & (df[feature] < pt_bins[ipt+1]) 
                scale = np.sqrt(df.loc[mask, 'lepton2_id_var'])/df.loc[mask, 'lepton2_id_weight']
                h_up, h_down = conditional_scaling(df, self._binning, scale, mask, feature, type='weight')
                self._df_sys[f'eff_mu_{ipt}_up'], self._df_sys[f'eff_mu_{ipt}_down'] = h_up, h_down


        elif self._selection in ['mutau', 'mu4j']:
            ## iso scale factor (called "reco" in ntuples)
            w_up      = df['weight']*(1 + np.sqrt(df['lepton1_reco_var'])/df['lepton1_reco_weight'])
            w_down    = df['weight']*(1 - np.sqrt(df['lepton1_reco_var'])/df['lepton1_reco_weight'])
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)
            self._df_sys['eff_iso_mu_up'], self._df_sys['eff_iso_mu_down'] = h_up, h_down

            ## id scale factor (lead muon)
            w_up      = df['weight']*(1 + np.sqrt(df['lepton1_id_var'])/df['lepton1_id_weight'])
            w_down    = df['weight']*(1 - np.sqrt(df['lepton1_id_var'])/df['lepton1_id_weight'])
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)
            self._df_sys['eff_id_mu_up'], self._df_sys['eff_id_mu_down'] = h_up, h_down

        elif self._selection == 'emu':
            ## reco scale factor
            w_nominal = df.weight/df['lepton1_reco_weight']
            w_up      = w_nominal*(df['lepton1_reco_weight'] + np.sqrt(df['lepton1_reco_var']))
            w_down    = w_nominal*(df['lepton1_reco_weight'] - np.sqrt(df['lepton1_reco_var']))
            h_up, _   = np.histogram(df[feature], bins=bins, weights=w_up)
            h_down, _ = np.histogram(df[feature], bins=bins, weights=w_down)

            #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
            #self._df_sys['eff_reco_e_up'], self._df_sys['eff_reco_e_down'] = y_up, y_down
            self._df_sys['eff_reco_mu_up'], self._df_sys['eff_reco_mu_down'] = h_up, h_down

            ## id/iso scale factor 
            mask      = abs(df.lead_lepton_flavor) == 13
            df.loc[mask, 'weight'] *= (df.loc[mask, 'lepton1_id_weight'] + np.sqrt(df.loc[mask, 'lepton1_id_var']))/df.loc[mask, 'lepton1_id_weight']
            h_up, _   = np.histogram(df[feature], bins=bins, weights=df['weight'])
            df.loc[mask, 'weight'] *= (df.loc[mask, 'lepton1_id_weight'] - np.sqrt(df.loc[mask, 'lepton1_id_var']))/(df.loc[mask, 'lepton1_id_weight'] + np.sqrt(df.loc[mask, 'lepton1_id_var']))
            h_down, _   = np.histogram(df[feature], bins=bins, weights=df['weight'])
            df.loc[mask, 'weight'] *= df.loc[mask, 'lepton1_id_weight']/(df.loc[mask, 'lepton1_id_weight'] - np.sqrt(df.loc[mask, 'lepton1_id_var']))

            #y_up, y_down = variation_template_smoothing(self._binning, self._h, h_up, h_down)
            #self._df_sys['eff_id_e_up'], self._df_sys['eff_id_e_down'] = y_up, y_down
            self._df_sys['eff_id_mu_up'], self._df_sys['eff_id_mu_down'] = h_up, h_down

            # muon pt-dependent efficiency systematic
            for ipt, pt_bin in enumerate(pt_bins[:-1]):
                mask = (df[feature] > pt_bin) & (df[feature] < pt_bins[ipt+1]) & (abs(df.trailing_lepton_flavor) == 13)
                scale = np.sqrt(df.loc[mask, 'lepton1_id_var'])/df.loc[mask, 'lepton1_id_weight']
                h_up, h_down = conditional_scaling(df, self._binning, scale, mask, feature, type='weight')
                self._df_sys[f'eff_mu_{ipt}_up'], self._df_sys[f'eff_mu_{ipt}_down'] = h_up, h_down


        ## muon energy scale
        scale = 0.001 # need reference
        if self._selection in ['mumu', 'mu4j']:
            h_up, h_down = energy_scale_morphing(bins, self._h, scale)

        elif self._selection == 'emu':
            mu_mask         = abs(df.trailing_lepton_flavor) == 13
            pt_masked       = df.loc[mu_mask, 'trailing_lepton_pt'].values
            pt_antimasked   = df.loc[~mu_mask, 'trailing_lepton_pt'].values
            h_masked, _     = np.histogram(pt_masked, bins=bins, weights=df.weight[mu_mask])
            h_antimasked, _ = np.histogram(pt_antimasked, bins=bins, weights=df.weight[~mu_mask])

            h_up, h_down = energy_scale_morphing(bins, h_masked, scale)
            h_up += h_antimasked
            h_down += h_antimasked

        self._df_sys['escale_mu_up'], self._df_sys['escale_mu_down'] = h_up, h_down

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

    def tau_systematics(self, df):
        '''
        Systematics for correctly identified taus.  Just energy scale currently.

        Parameters:
        ===========
        df: dataframe for target dataset
        '''

        # tau id efficiency systematic
        pt_bins = [20, 25, 30, 40, 50, 65, np.inf]
        sigma   = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05] 
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
            # varies individual taus
            scale = 0.012
            #h_up, h_down = conditional_scaling(df, self._binning, 0.012, df.tau_decay_mode == decay_mode, 'lepton2_pt')

            dm_mask      = df.tau_decay_mode == decay_mode
            pt_masked    = df.loc[dm_mask, 'lepton2_pt'].values
            h_masked, _  = np.histogram(pt_masked, bins=self._binning, weights=df.weight[dm_mask])
            h_up, h_down = energy_scale_morphing(self._binning, h_masked, scale)

            pt_antimasked   = df.loc[~dm_mask, 'lepton2_pt'].values
            h_antimasked, _ = np.histogram(pt_antimasked, bins=self._binning, weights=df.weight[~dm_mask])
            h_up += h_antimasked
            h_down += h_antimasked

            self._df_sys[f'escale_tau_{decay_mode}_up'], self._df_sys[f'escale_tau_{decay_mode}_down'] = h_up, h_down

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

    def top_pt_systematics(self, df):
        '''
        Variation from reweighting the top quark pt spectrum in ttbar events
        (https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting).  Not sure
        what exactly I'm supposed to do here.  My understanding is there are
        two cases: no weight and with weight.  Effectively, this will mean
        there is an up variation (weighted) and nominal case (no weight), but
        no downward variation. \shrug 

        Parameters:
        ===========
        df: dataframe for dataset
        cut: jet category
        '''

        w_up      = df.weight*df.top_pt_weight
        h_up, _   = np.histogram(df[self._feature], bins=self._binning, weights=w_up*(df.weight.sum()/w_up.sum()))
        mask = self._h > 0
        h_down = np.zeros_like(self._h)
        h_down[mask] = self._h[mask] - 0.33*(h_up[mask] - self._h[mask])
        self._df_sys['top_pt_up'], self._df_sys['top_pt_down'] = h_up, h_down

    def ww_pt_systematics(self, df):
        '''
        Applies resummatiion and scale variation uncertainties to qq->WW
        sample.  (Be careful to only apply the variation to the qq production
        and not gg!)
        '''

        weights = df['weight'].values.copy()
        mask = weights != 1
       
        # scale variation
        k_up, k_down = 0.993, 1.001
        h_up, _   = np.histogram(df[self._feature], bins=self._binning, weights=weights*df['ww_pt_scale_up']/k_up)
        h_down, _ = np.histogram(df[self._feature], bins=self._binning, weights=weights*df['ww_pt_scale_down']/k_down)

        self._df_sys['ww_scale_up'], self._df_sys['ww_scale_down'] = h_up, h_down

        # resum variation
        k_up, k_down = 1.012, 0.9493
        h_up, _   = np.histogram(df[self._feature], bins=self._binning, weights=weights*df['ww_pt_resum_up']/k_up)
        h_down, _ = np.histogram(df[self._feature], bins=self._binning, weights=weights*df['ww_pt_resum_down']/k_down)

        #print('--------')
        #print(self._h)
        #print(h_up, h_down, sep='\n')
        #print(k_up*h_up, k_down*h_down, sep='\n')

        self._df_sys['ww_resum_up'], self._df_sys['ww_resum_down'] = h_up, h_down

