

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scripts.plot_tools as pt
import scripts.fit_helpers as fh

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

def jet_scale(df, feature, bins, sys_type, jet_condition):
    '''
    Jet systematics are treated as shape systematics, but mostly vary depending
    on the jet/b tag multiplicity.  Nonetheless, it's easier to account for
    them as a shape systematic.
    '''

    # systematic up/down
    up_condition   = jet_condition.replace('n_bjets', f'n_bjets_{sys_type}_up')
    down_condition = jet_condition.replace('n_bjets', f'n_bjets_{sys_type}_down')
    if sys_type not in ['btag', 'mistag']:
        up_condition   = up_condition.replace('n_jets', f'n_jets_{sys_type}_up')
        down_condition = down_condition.replace('n_jets', f'n_jets_{sys_type}_down')

    h_up, _   = np.histogram(df.query(up_condition)[feature], 
                             bins=bins, 
                             weights=df.query(up_condition).weight
                             )
    h_down, _ = np.histogram(df.query(down_condition)[feature],
                             bins=bins, 
                             weights=df.query(down_condition).weight
                             )

    return h_up, h_down

def theory_systematics(df_nominal, dm, feature, bins, sys_type, cut):
    '''
    Theory systematics are handled in two different ways: a subset of the
    systematics are estimated from dedicated samples where a particular
    generator parameter has been scale +/- 1 sigma from the nominal value.
    These indclude,
       * isr
       * fsr
       * ME+PS (hdamp)
       * UE (tune)
    Other systematics are calculated based on event level weights.  These include,
       * PDF
       * alpha_s
       * QCD scale (mu_R and mu_F)
    The variation due to normalization is divided out so that only the slope changes are present.
    '''

    if sys_type in ['isr', 'fsr', 'hdamp', 'tune']:
        df_up     = dm.get_dataframe(f'ttbar_{sys_type}up', cut)
        df_down   = dm.get_dataframe(f'ttbar_{sys_type}down', cut)

        h_up, _   = np.histogram(df_up[feature], bins=bins, weights=df_up.weight)
        h_down, _ = np.histogram(df_down[feature], bins=bins, weights=df_down.weight)
    elif sys_type == 'pdf':
        h_up, _   = np.histogram(df_nominal[feature], bins=bins, weights=df_nominal.weight*(1 + np.sqrt(df_nominal.pdf_var)/np.sqrt(100)))
        h_down, _ = np.histogram(df_nominal[feature], bins=bins, weights=df_nominal.weight*(1 - np.sqrt(df_nominal.pdf_var)/np.sqrt(100)))
    elif sys_type == 'mur':
        h_up, _   = np.histogram(df_nominal[feature], bins=bins, weights=df_nominal.weight*df_nominal.qcd_weight_up_nom)
        h_down, _ = np.histogram(df_nominal[feature], bins=bins, weights=df_nominal.weight*df_nominal.qcd_weight_down_nom)
    elif sys_type == 'muf':
        h_up, _   = np.histogram(df_nominal[feature], bins=bins, weights=df_nominal.weight*df_nominal.qcd_weight_nom_up)
        h_down, _ = np.histogram(df_nominal[feature], bins=bins, weights=df_nominal.weight*df_nominal.qcd_weight_nom_down)
    elif sys_type == 'mur_muf':
        h_up, _   = np.histogram(df_nominal[feature], bins=bins, weights=df_nominal.weight*df_nominal.qcd_weight_up_up)
        h_down, _ = np.histogram(df_nominal[feature], bins=bins, weights=df_nominal.weight*df_nominal.qcd_weight_down_down)

    return h_up, h_down

class SystematicTemplateGenerator():
    def __init__(self, selection, label, feature, binning, h_nominal, cut, cut_name):
        self._selection = selection
        self._label     = label
        self._feature   = feature
        self._binning   = binning
        self._h         = h_nominal
        self._cut       = cut
        self._cut_name  = cut_name
        self._df_sys    = pd.DataFrame(dict(bins=binning[:-1]))

    def get_syst_dataframe(self):
        return self._df_sys.set_index('bins')

    def jet_shape_systematics(self, df):
        '''
        Generates morpthing templates for: 
           * jet energy scale/resolution
           * b tag/mistag efficiency

        Parameters:
        ===========
        df: dataframe for target dataset without the jet cuts applied
        '''

        for syst_type in ['jes', 'jer', 'btag', 'mistag']:
            h_up, h_down = jet_scale(df, self._feature, self._binning, syst_type, self._cut)
            self._df_sys[f'{syst_type}_up'], self._df_sys[f'{syst_type}_down'] = h_up, h_down
            self.template_overlays(h_up, h_down, syst_type)

    def reco_shape_systematics(self, df):
        '''
        Generates templates for:
           * pileup
           * lepton energy scale

        Parameters:
        ===========
        df: dataframe for target dataset with
       '''
        bins = self._binning
        feature = self._feature

        # pileup
        h_up, h_down = pileup_morph(df, feature, bins)
        self.template_overlays(h_up, h_down, 'pileup')
        self._df_sys['pileup_up'], self._df_sys['pileup_down'] = h_up, h_down

        # lepton energy scale
        ## muon scale
        if self._selection in ['mumu', 'mu4j']:
            scale = 0.002 # need reference
            h_up, _      = np.histogram((1+scale)*df[feature], bins=bins, weights=df.weight)
            h_down, _    = np.histogram((1-scale)*df[feature], bins=bins, weights=df.weight)
            self.template_overlays(h_up, h_down, 'escale_mu')
            self._df_sys['escale_mu_up'], self._df_sys['escale_mu_down'] = h_up, h_down

        ## electron scale
        if self._selection in ['ee', 'e4j']:
            scale = 0.005 # need reference
            h_up, _      = np.histogram((1+scale)*df[feature], bins=bins, weights=df.weight)
            h_down, _    = np.histogram((1-scale)*df[feature], bins=bins, weights=df.weight)
            self.template_overlays(h_up, h_down, 'escale_e')
            self._df_sys['escale_e_up'], self._df_sys['escale_e_down'] = h_up, h_down

        ## tau scale
        if self._selection in ['etau', 'mutau']:
            scale = 0.012 # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendation13TeV#Tau_energy_scale
            h_up, _      = np.histogram((1+scale)*df[feature], bins=bins, weights=df.weight)
            h_down, _    = np.histogram((1-scale)*df[feature], bins=bins, weights=df.weight)
            self.template_overlays(h_up, h_down, 'escale_tau')
            self._df_sys['escale_tau_up'], self._df_sys['escale_tau_down'] = h_up, h_down

        ## emu channel needs to be treated separately
        if self._selection == 'emu':
            ### not great... scale up, then down, then down, then up

            ### electron scale
            scale = 0.005 # need reference
            mask = abs(df.trailing_lepton_flavor) == 11
            df.loc[mask, 'trailing_lepton_pt'] *= 1 + scale
            h_up, _   = np.histogram(df.trailing_lepton_pt, bins=bins, weights=df.weight)
            df.loc[mask, 'trailing_lepton_pt'] *= (1 - scale)/(1 + scale)
            h_down, _ = np.histogram(df.trailing_lepton_pt, bins=bins, weights=df.weight)
            df.loc[mask, 'trailing_lepton_pt'] /= (1 - scale)

            self.template_overlays(h_up, h_down, 'escale_e')
            self._df_sys['escale_e_up'], self._df_sys['escale_e_down'] = h_up, h_down

            ### muon scale
            scale = 0.002 # need reference
            mask = abs(df.trailing_lepton_flavor) == 13
            df.loc[mask, 'trailing_lepton_pt'] *= 1 + scale
            h_up, _   = np.histogram(df.trailing_lepton_pt, bins=bins, weights=df.weight)
            df.loc[mask, 'trailing_lepton_pt'] *= (1 - scale)/(1 + scale)
            h_down, _ = np.histogram(df.trailing_lepton_pt, bins=bins, weights=df.weight)
            df.loc[mask, 'trailing_lepton_pt'] /= (1 - scale)

            self.template_overlays(h_up, h_down, 'escale_mu')
            self._df_sys['escale_mu_up'], self._df_sys['escale_mu_down'] = h_up, h_down

        return

    def theory_shape_systematics(self, df, dm_syst, cut, renorm):
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
        dm_syst: data manager with dedicated ttbar samples for non-weight based systematics
        cut: jet and W decay category cut
        renorm: dict of renormalization factors to remove normalization changes due to pdf and QCD changes
        '''

        for syst_type in ['mur', 'muf', 'mur_muf', 'pdf', 'isr', 'fsr']:#, 'hdamp', 'tune']:
            h_up, h_down = theory_systematics(df, dm_syst, self._feature, self._binning, syst_type, cut)
            if syst_type in ['mur', 'muf', 'mur_muf', 'pdf']:
                h_up   /= renorm[f'{syst_type}_up']
                h_down /= renorm[f'{syst_type}_down']
            self._df_sys[f'{syst_type}_up'], self._df_sys[f'{syst_type}_down'] = h_up, h_down
            self.template_overlays(h_up, h_down, syst_type)

        return


    def template_overlays(self, h_up, h_down, systematic):
        '''
        Overlay nominal, variation up, and variation down templates.
        '''

        output_path = f'plots/systematics/{self._selection}/{self._label}'
        pt.make_directory(output_path, clear=False)
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), facecolor='white', sharex=False, gridspec_kw={'height_ratios':[3,1]})
        fig.subplots_adjust(hspace=0)

        # get the histogram templates
        h_nominal = self._h
        h_nominal[h_nominal == 0] = 1e-9
        h_up[h_up == 0] = 1e-9
        h_down[h_down == 0] = 1e-9

        # define the bins
        bins = self._binning
        dx = (bins[1:] - bins[:-1])/2
        x = bins[:-1] + dx

        ax = axes[0]
        ax.plot(x, h_nominal/dx, drawstyle='steps-post', c='C1', linestyle='-', linewidth=1.)
        ax.plot(x, h_up/dx,      drawstyle='steps-post', c='C0', linestyle='-', linewidth=1.)
        ax.plot(x, h_down/dx,    drawstyle='steps-post', c='C2', linestyle='-', linewidth=1.)
        ax.fill_between(x, h_up/dx, h_down/dx, color = 'C1', alpha=0.5, step='post')

        ax.set_xlim(bins[0], bins[-2])
        ax.set_ylim(0., 1.25*np.max(h_nominal/dx))
        ax.legend(['nominal', r'$+\sigma$', r'$-\sigma$'])
        ax.set_ylabel('Entries / GeV')
        ax.set_title(fh.fancy_labels[self._selection][1])
        ax.grid()

        ax = axes[1]
        y_up = h_up/h_nominal
        y_down = h_down/h_nominal
        ax.plot(x, y_up,   'C0', drawstyle='steps-post')
        ax.plot(x, y_down, 'C2', drawstyle='steps-post')
        ax.fill_between(x, y_up, y_down, color = 'C1', alpha=0.5, step='post')
        ax.plot([bins[0], bins[-2]], [1, 1], 'C1--')

        ax.set_xlim(bins[0], bins[-2])
        ax.set_ylim(0.95*np.min([y_up.min(), y_down.min()]), 1.05*np.max([y_up.max(), y_down.max()]))
        ax.set_xlabel(fh.fancy_labels[self._selection][0])
        ax.set_ylabel(r'$\sf \frac{N^{\pm}}{N^{0}}$', fontsize=14)
        ax.grid()
        #ax.set_yscale('linear')

        plt.tight_layout()
        plt.savefig(f'{output_path}/{systematic}_{self._cut_name}.pdf')
        plt.savefig(f'{output_path}/{systematic}_{self._cut_name}.png')
        plt.close()

