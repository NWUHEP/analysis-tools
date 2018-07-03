import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from functools import partial
import numdifftools as nd
#from scipy.integrate import quad
#from lmfit import Parameters

#import nllfit.fit_tools as ft
import scripts.plot_tools as pt

np.set_printoptions(precision=2)

def ebar_wrapper(data, ax, bins, limits, style):
    x, y, err = pt.hist_to_errorbar(data, bins, limits)
    mask = y > 0.
    x, y, err = x[mask], y[mask], err[mask]
    ax.errorbar(x, y, yerr=err,
                capsize = 0,
                fmt = style,
                elinewidth = 2,
                markersize = 5
                )

def shape_morphing(f, up_template, down_template, order='quadratic'):
    '''
    Efficiency shape morphing for nuisance parameters.  
    '''

    if order == 'linear':
        r_eff = 1 + f*(up_template - down_template)/2
    elif order == 'quadratic':
        r_eff = (f*(f - 1)/2)*down_template - (f - 1)*(f + 1) + (f*(f + 1)/2)*up_template

    return r_eff

class FitData(object):
    def __init__(self, path, selections, feature_map, bins=[0]):
        self._selections     = selections
        self._n_selections   = len(selections)
        self._bins           = bins
        self._decay_map      = pd.read_csv('data/decay_map.csv').set_index('id')
        self._selection_data = {s: self._initialize_template_data(path, feature_map[s], s) for s in selections}

        # parameters
        self._beta   = [0.108, 0.108, 0.108, 1 - 3*0.108]  # e, mu, tau, h
        self._tau_br = [0.1783, 0.1741, 0.6476]  # e, mu, h
        #self._initialize_nuisance_parameters(selections)

    def _initialize_template_data(self, location, target, selection):
        '''
        Retrieves data, bg, and signal templates as well as their variances.
        '''

        out_data = dict()
        for b in self._bins:
            bin_data = dict()
            # get our data
            df_templates = pd.read_csv(f'{location}/{selection}/{target}_bin-{b}_val.csv').set_index('bins')
            df_syst      = pd.read_csv(f'{location}/{selection}/{target}_bin-{b}_syst.csv').set_index('bins')
            df_vars      = pd.read_csv(f'{location}/{selection}/{target}_bin-{b}_var.csv').set_index('bins')

            # replace NaN and negative entries with 0
            df_templates = df_templates.fillna(0)
            df_vars      = df_vars.fillna(0)
            df_syst      = df_syst.fillna(0)
            df_templates[df_templates < 0.] = 0.

            # split template dataframe into data, bg, and signal, and then convert to numpy arrays
            decay_map = self._decay_map['decay'].values
            bin_data['bins'] = df_templates.index.values
            bin_data['data'] = (df_templates['data'].values, df_vars['data'].values)

            # get background components
            bin_data['zjets'] = (df_templates['zjets'].values, df_vars['zjets'].values)
            bin_data['wjets'] = (df_templates['wjets'].values, df_vars['wjets'].values)

            if selection in ['mu4j', 'mutau']:
                bin_data['fakes'] = (df_templates['fakes'].values, df_vars['fakes'].values)

            # get signal components
            bin_data['signal'] = (df_templates[decay_map].values, df_vars[decay_map].values)

            # get shape variation templates 
            bin_data['syst'] = df_syst

            out_data[b] = bin_data

        return out_data

    def _initialize_nuisance_parameters(self, selections):
        '''
        Retrieves nuisance parameters (needs development)
        '''
        pass
        #self._nuisance_params = pd.read_csv('data/nuisance_parameters.csv')

    def get_selection_data(self, selection):
        return self._selection_data[selection]

    def get_params_init(self):
        return self._beta

    def objective(self, params, data, cost_type='poisson'):
        '''
        Cost function for MC data model.  This version has no background
        compononent and is intended for fitting toy data generated from the signal
        MC.

        Parameters:
        ===========
        params : numpy array of parameters.  The first four are the W branching
                 fractions, all successive ones are nuisance parameters.
        data : dataset to be fitted
        cost_type : either 'chi2' or 'poisson'
        mask : an array with same size as the input parameters for indicating parameters to fix
        '''

        # unpack parameters here
        # branching fractions first
        beta = params[:4]

        # nuisance parameters
        # normalization
        lumi       = params[4]
        xs_top     = params[5]
        xs_zjets   = params[6]
        xs_wjets   = params[7]
        norm_fakes = params[8]
        eff_e      = params[9]
        eff_mu     = params[10]
        eff_tau    = params[11]

        # morphing 
        pileup     = 1. - params[12]

        # lepton energy scale
        escale_e   = 1. - params[13]
        escale_mu  = 1. - params[14]
        escale_tau = 1. - params[15]

        # jet systematics
        jes        = 1. - params[16]
        jer        = 1. - params[17]
        btag       = 1. - params[18]
        mistag     = 1. - params[19]

        # theory systematics
        fsr        = 1. - params[20]
        isr        = 1. - params[21]
        tune       = 1. - params[22]
        hdamp      = 1. - params[23]
        qcd        = 1. - params[24]
        pdf        = 1. - params[25]

        # calculate per category, per bin costs
        cost = 0
        for sel in self._selections:
            s_data = self.get_selection_data(sel)

            for b in self._bins:
                df_syst          = s_data[b]['syst']
                f_data, var_data = data[b][sel], data[b][sel]
                f_sig, var_sig   = signal_mixture_model(beta,
                                                        br_tau = self._tau_br,
                                                        h_temp = s_data[b]['signal']
                                                        )

                # prepare mixture
                f_model   = xs_top*f_sig
                var_model = var_sig

                # get background components and apply cross-section nuisance parameters
                f_zjets, var_zjets = s_data[b]['zjets']
                f_wjets, var_wjets = s_data[b]['wjets']
                f_model   += xs_zjets*f_zjets + xs_wjets*f_wjets
                var_model += var_zjets + var_wjets

                # lepton efficiencies as normalization nuisance parameters
                # lepton energy scale as morphing parameters
                if sel in 'ee':
                    f_model *= eff_e**2
                    f_model *= shape_morphing(escale_e, df_syst['el_es_up'], df_syst['el_es_down'])
                elif sel in 'emu':
                    f_model *= eff_e*eff_mu
                    f_model *= shape_morphing(escale_e, df_syst['el_es_up'], df_syst['el_es_down'])
                    f_model *= shape_morphing(escale_mu, df_syst['mu_es_up'], df_syst['mu_es_down'])
                elif sel in 'mumu':
                    f_model *= eff_mu**2
                    f_model *= shape_morphing(escale_mu, df_syst['mu_es_up'], df_syst['mu_es_down'])
                elif sel == 'etau':
                    f_model *= eff_tau*eff_e
                    f_model *= shape_morphing(escale_tau, df_syst['tau_es_up'], df_syst['tau_es_down'])
                elif sel == 'mutau':
                    f_model *= eff_tau*eff_mu
                    f_model *= shape_morphing(escale_tau, df_syst['tau_es_up'], df_syst['tau_es_down'])
                elif sel == 'e4j':
                    f_model *= eff_e
                    f_model *= shape_morphing(escale_e, df_syst['el_es_up'], df_syst['el_es_down'])
                elif sel == 'mu4j':
                    f_model *= eff_mu
                    f_model *= shape_morphing(escale_mu, df_syst['mu_es_up'], df_syst['mu_es_down'])

                # jet energy scale/resolution and b tag/mistag systematics
                # (these are more like normalization systematics, but it's
                # easier to apply them as shape systematics)
                # jes
                f_model *= shape_morphing(jes, df_syst['jes_up'], df_syst['jes_down'])

                # jer
                f_model *= shape_morphing(jer, df_syst['jer_up'], df_syst['jer_down'])

                # btag
                f_model *= shape_morphing(btag, df_syst['btag_up'], df_syst['btag_down'])

                # mistag
                f_model *= shape_morphing(mistag, df_syst['mistag_up'], df_syst['mistag_down'])

                # shape systematic from pileup
                f_model *= shape_morphing(pileup, df_syst['pileup_up'], df_syst['pileup_down'])

                # theory systematics #
                # fsr
                f_model *= shape_morphing(fsr, df_syst['fsr_up'], df_syst['fsr_down'])

                # isr
                f_model *= shape_morphing(isr, df_syst['isr_up'], df_syst['isr_down'])

                # UE tune
                f_model *= shape_morphing(tune, df_syst['tune_up'], df_syst['tune_down'])

                # ME-PS matching
                f_model *= shape_morphing(hdamp, df_syst['hdamp_up'], df_syst['hdamp_down'])

                # QCD scale (mu_R/mu_F variation)
                f_model *= shape_morphing(qcd, df_syst['qcd_up'], df_syst['qcd_down'])

                # PDF variation
                f_model *= shape_morphing(pdf, df_syst['pdf_up'], df_syst['pdf_down'])

                # apply overall lumi nuisance parameter
                f_model *= lumi

                # get fake background and include normalization nuisance parameters
                if sel == 'mu4j': 
                    f_fakes, var_fakes = s_data[b]['fakes']
                    f_model   += norm_fakes*f_fakes
                    var_model += var_fakes

                # add removing shape information as an argument
                #f_data = np.sum(f_data)
                #f_model = np.sum(f_model)
                #print(sel, f_data, f_model)

                # calculate the cost
                if cost_type == 'chi2':
                    mask = var_data + var_model > 0
                    nll = (f_data - f_model)**2 / (var_data + var_model)
                    nll = nll[mask]
                elif cost_type == 'poisson':
                    mask = f_model > 0
                    nll = -f_data[mask]*np.log(f_model[mask]) + f_model[mask]
                cost += np.sum(nll)

        # require that the branching fractions sum to 1
        cost += (1 - np.sum(beta))**2/(2*0.000001**2)  

        # constrain branching fractions (optional)
        #beta_init = np.array(3*[0.108, ] + [1. - 3*0.108])
        #beta_var = np.array(3*[0.001**2, ] + [0.0027**2])
        #cost += np.sum((beta_init - beta)**2/(2*beta_var))

        # Add prior terms for nuisance parameters correlated across channels (lumi, cross-sections)
        # luminosity
        lumi_var = 0.025**2
        cost += (lumi - 1.)**2 / (2*lumi_var)

        # pileup
        pileup_var = 1.**2
        cost += (pileup - 1.)**2 / (2*pileup_var)

        ## top
        xs_top_var = 0.05**2
        cost += (xs_top - 1.)**2 / (2*xs_top_var)

        ## zjets
        xs_zjets_var = 0.3**2
        cost += (xs_zjets - 1.)**2 / (2*xs_zjets_var)

        ## wjets
        xs_wjets_var = 0.3**2
        cost += (xs_wjets - 1.)**2 / (2*xs_wjets_var)

        ## fakes
        norm_fakes_var = 0.25**2
        cost += (norm_fakes - 1.)**2 / (2*norm_fakes_var)

        ## lepton effs
        eff_e_var = 0.01**2
        cost += (eff_e - 1.)**2 / (2*eff_e_var)

        eff_mu_var = 0.01**2
        cost += (eff_mu - 1.)**2 / (2*eff_mu_var)

        eff_tau_var = 0.05**2
        cost += (eff_tau - 1.)**2 / (2*eff_tau_var)

        ## lepton energy scales
        escale_e_var = 0.5**2
        cost += (escale_e - 1.)**2 / (2*escale_e_var)

        escale_mu_var = 0.2**2
        cost += (escale_mu - 1.)**2 / (2*escale_mu_var)

        escale_tau_var = 1.**2
        cost += (escale_tau - 1.)**2 / (2*escale_tau_var)

        ## jes
        jes_var = 1.**2
        cost += (jes - 1.)**2 / (2*jes_var)

        # jer
        jer_var = 1.**2
        cost += (jer - 1.)**2 / (2*jer_var)

        # btag
        btag_var = 1.**2
        cost += (btag - 1.)**2 / (2*btag_var)

        # mistag
        mistag_var = 1.**2
        cost += (mistag - 1.)**2 / (2*mistag_var)

        # fsr
        fsr_var = 0.5**2
        cost += (fsr - 1.)**2 / (2*fsr_var)

        # isr
        isr_var = 1.**2
        cost += (isr - 1.)**2 / (2*isr_var)

        # tune
        tune_var = 1.**2
        cost += (tune - 1.)**2 / (2*tune_var)

        # hdamp
        hdamp_var = 1.**2
        cost += (hdamp - 1.)**2 / (2*hdamp_var)

        # qcd
        qcd_var = 1.**2
        cost += (qcd - 1.)**2 / (2*qcd_var)

        # pdf
        pdf_var = 1.**2
        cost += (pdf - 1.)**2 / (2*pdf_var)

        ###########################################

        return cost


def signal_amplitudes(beta, br_tau):
    '''
    returns an array of branching fractions for each signal channel.

    parameters:
    ===========
    beta : W branching fractions [beta_e, beta_mu, beta_tau, beta_h]
    br_tau : tau branching fractions [br_e, br_mu, br_h]
    '''
    amplitudes = np.array([beta[0]*beta[0],  # e, e
                           beta[1]*beta[1],  # mu, mu
                           2*beta[0]*beta[1],  # e, mu
                           beta[2]*beta[2]*br_tau[0]**2,  # tau_e, tau_e
                           beta[2]*beta[2]*br_tau[1]**2,  # tau_mu, tau_mu
                           2*beta[2]*beta[2]*br_tau[0]*br_tau[1],  # tau_e, tau_m
                           2*beta[2]*beta[2]*br_tau[0]*br_tau[2],  # tau_e, tau_
                           2*beta[2]*beta[2]*br_tau[1]*br_tau[2],  # tau_mu, tau_h
                           2*beta[0]*beta[2]*br_tau[0],  # e, tau_e
                           beta[2]*beta[2]*br_tau[2]*br_tau[2],  # tau_h, tau_h
                           2*beta[0]*beta[2]*br_tau[1],  # e, tau_mu
                           2*beta[0]*beta[2]*br_tau[2],  # e, tau_h
                           2*beta[1]*beta[2]*br_tau[0],  # mu, tau_e
                           2*beta[1]*beta[2]*br_tau[1],  # mu, tau_mu
                           2*beta[1]*beta[2]*br_tau[2],  # mu, tau_h
                           2*beta[0]*beta[3],  # e, h
                           2*beta[1]*beta[3],  # mu, h
                           2*beta[2]*beta[3]*br_tau[0],  # tau_e, h
                           2*beta[2]*beta[3]*br_tau[1],  # tau_mu, h
                           2*beta[2]*beta[3]*br_tau[2],  # tau_h, h
                           beta[3]*beta[3],  # tau_h, h
                           ])

    return amplitudes

def signal_mixture_model(beta, br_tau, h_temp, mask=None, sample=False):
    '''
    Mixture model for the ttbar/tW signal model.  The output will be an array
    corresponding to a sum over the input template histograms scaled by their
    respective efficiencies and branching fraction products.

    parameters:
    ==========
    beta : branching fractions for the W decay
    br_tau : branching fractions for the tau decay
    h_temp : a tuple with the template histograms and their errors
    mask : a mask that selects a subset of mixture components
    sample : if True, the input templates will be sampled before returning
    '''

    beta_init  = signal_amplitudes([0.108, 0.108, 0.108, 0.676], [0.1783, 0.1741, 0.6476])
    beta_fit   = signal_amplitudes(beta, br_tau)
    beta_ratio = beta_fit/beta_init

    if not isinstance(mask, type(None)):
        beta_ratio = mask*beta_ratio

    if sample:
        f = np.dot(np.random.poisson(h_temp[0]), beta_ratio)
    else:
        f = np.dot(h_temp[0], beta_ratio)
    var = np.dot(h_temp[1], beta_ratio**2)

    return f, var

def calculate_covariance(f, x0):
    '''
    calculates covariance for input function.
    '''

    hcalc = nd.Hessian(f,
                       step        = 1e-2,
                       method      = 'central',
                       full_output = True
                       )

    hobj = hcalc(x0)[0]
    if np.linalg.det(hobj) != 0:
        # calculate the full covariance matrix in the case that the H
        hinv        = np.linalg.pinv(hobj)
        sig         = np.sqrt(hinv.diagonal())
        corr_matrix = hinv/np.outer(sig, sig)

        return sig, corr_matrix
    else:
        return False

def fit_plot(fit_data, selection, xlabel, log_scale=False):

    # unpack fit_data
    results  = fit_data['results']
    ix       = fit_data['selections'].index(selection)
    n_sel    = fit_data['n_selections']
    br_tau   = fit_data['br_tau']

    sel_data = fit_data[selection]
    data     = sel_data['data']
    bg       = sel_data['bg']
    signal   = sel_data['signal']

    #print(data[0].sum(), bg[0].sum(), signal[0].sum())

    # starting amplitudes
    p_init     = fit_data['p_init']['vals']
    beta_init  = p_init[n_sel+1:]
    lumi_scale = results.x[0]
    alpha_fit  = results.x[ix+1]
    beta_fit   = results.x[n_sel+1:]

    # initialize the canvas
    fig, axes = plt.subplots(2, 1,
                             figsize     = (8, 9),
                             facecolor   = 'white',
                             sharex      = True,
                             gridspec_kw = {'height_ratios': [3, 1]}
                             )
    fig.subplots_adjust(hspace=0)

    # initialize bins
    bins = fit_data[selection]['bins']
    xmin, xmax = bins.min(), bins.max()
    dx = (bins[1:] - bins[:-1])
    dx = np.append(dx, dx[-1])
    x = bins + dx/2

    # plot the data
    y_data, yerr_data = data[0], np.sqrt(data[1])
    data_plot = axes[0].errorbar(x, y_data/dx, yerr_data/dx,
                                 fmt        = 'ko',
                                 capsize    = 0,
                                 elinewidth = 2
                                 )

    # plot signal and bg (prefit)
    y_bg, yerr_bg = bg[0], np.sqrt(bg[1])
    axes[0].errorbar(x, y_bg/dx, yerr_bg/dx,
                     label='_nolegend_',
                     fmt        = 'C1.',
                     markersize = 0,
                     capsize    = 0,
                     elinewidth = 5,
                     alpha = 0.5
                     )
    axes[0].plot(bins, y_bg/dx, drawstyle='steps-post', c='C1', alpha=0.5)

    y_sig, yvar_sig = signal_mixture_model(beta_init, br_tau, signal)
    y_combined, yerr_combined = y_bg + y_sig, np.sqrt(yerr_bg**2 + yvar_sig)
    axes[0].errorbar(x, y_combined/dx, yerr_combined/dx,
                     label='_nolegend_',
                     fmt        = 'C0.',
                     markersize = 0,
                     capsize    = 0,
                     elinewidth = 5,
                     alpha = 0.5
                     )
    axes[0].plot(bins, y_combined/dx, drawstyle='steps-post', c='C0', alpha=0.5)

    ratio_pre = y_data/y_combined
    ratio_pre_err = (1/y_combined**2)*np.sqrt(y_data**2*yerr_combined**2 + y_combined**2*yerr_data**2)

    y_bg, yerr_bg = lumi_scale*alpha_fit*y_bg, lumi_scale*alpha_fit*yerr_bg
    axes[0].errorbar(x, y_bg/dx, yerr_bg/dx,
                     label = '_nolegend_',
                     fmt        = 'C3.',
                     markersize = 0,
                     capsize    = 0,
                     elinewidth = 5,
                     alpha = 0.5,
                     )
    axes[0].plot(bins, y_bg/dx, drawstyle='steps-post', linestyle='--', label='_nolegend_', c='C3')

    y_sig, yvar_sig = signal_mixture_model(beta_fit, br_tau, signal)
    y_combined      = y_bg + lumi_scale*y_sig
    yerr_combined   = np.sqrt(yerr_bg**2 + yvar_sig*lumi_scale**2)
    axes[0].errorbar(x, y_combined/dx, yerr_combined/dx,
                     fmt        = 'C9.',
                     capsize    = 0,
                     markersize = 0,
                     elinewidth = 5,
                     alpha = 0.5,
                     label = '_nolegend_'
                     )
    axes[0].plot(bins, y_combined/dx, drawstyle='steps-post', linestyle='--', label='_nolegend_', c='C9')

    ratio_post = y_data/y_combined
    ratio_post_err = (1/y_combined**2)*np.sqrt(y_data**2*yerr_combined**2 + y_combined**2*yerr_data**2)

    axes[0].grid()
    axes[0].set_ylabel(r'Events / 1 GeV')
    axes[0].set_xlim(xmin, xmax)
    if log_scale:
        axes[0].set_yscale('log')
        axes[0].set_ylim(0.05, 10*np.max(y_data/dx))
    else:
        axes[0].set_ylim(0., 1.2*np.max(y_data/dx))

    # custom legend handles
    from matplotlib.legend_handler import HandlerBase

    class AnyObjectHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0, y0+width],
                            [0.7*height, 0.7*height],
                            linestyle='--',
                            color=orig_handle[1]
                            )
            l2 = plt.Line2D([x0, y0+width],
                            [0.3*height, 0.3*height],
                            color=orig_handle[0]
                            )
            return [l1, l2]

    axes[0].legend([('C1', 'C3'), ('C0', 'C9'), data_plot],
                   ['background', r'$\sf t\bar{t}+tW$', 'data'],
                   handler_map={tuple: AnyObjectHandler()}
                   )

    #axes[0].legend([
    #                r'BG',
    #                r'$\sf t\bar{t}/tW$',
    #                'Data',
    #                ])

    #axes[0].text(80, 2200, r'$\alpha = $' + f' {results.x[0]:3.4} +/- {sig[0]:2.2}', {'size':20})

    ### calculate ratios
    axes[1].errorbar(x, ratio_pre, ratio_pre_err,
                     fmt        = 'C0o',
                     ecolor     = 'C0',
                     capsize    = 0,
                     elinewidth = 3,
                     alpha = 1.
                     )
    axes[1].errorbar(x, ratio_post, ratio_post_err,
                     fmt        = 'C1o',
                     ecolor     = 'C1',
                     capsize    = 0,
                     elinewidth = 3,
                     alpha = 1.
                     )

    axes[1].grid()
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Data / MC')
    axes[1].set_ylim(0.8, 1.19)
    #axes[1].legend(['prefit', 'postfit'], loc=1, fontsize=16)
    axes[1].plot([xmin, xmax], [1, 1], 'k--', alpha=0.5)

    plt.savefig(f'plots/fits/{selection}_channel.pdf')
    plt.savefig(f'plots/fits/{selection}_channel.png')
    plt.show()
