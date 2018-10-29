import pickle
from multiprocessing import Process, Queue, Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numdifftools as nd

#from functools import partial
#from scipy.integrate import quad
#from lmfit import Parameters

#import nllfit.fit_tools as ft
import scripts.plot_tools as pt

np.set_printoptions(precision=2)

features = dict()
features['mumu']  = 'lepton2_pt' # trailing muon pt
features['ee']    = 'lepton2_pt' # trailing electron pt
features['emu']   = 'trailing_lepton_pt' # like the name says
features['mutau'] = 'lepton2_pt' # tau pt
features['etau']  = 'lepton2_pt' # tau pt
features['mu4j']  = 'lepton1_pt' # muon pt
features['e4j']   = 'lepton1_pt' # electron pt

fancy_labels = dict()
fancy_labels['mumu']  = (r'$\sf p_{T,\mu}$', r'$\sf \mu\mu$')
fancy_labels['ee']    = (r'$\sf p_{T,e}$', r'$\sf ee$')
fancy_labels['emu']   = (r'$\sf p_{T,trailing}$', r'$\sf e\mu$')
fancy_labels['mutau'] = (r'$\sf p_{T,\tau}$', r'$\sf \mu\tau$')
fancy_labels['etau']  = (r'$\sf p_{T,\tau}$', r'$\sf e\tau$')
fancy_labels['mu4j']  = (r'$\sf p_{T,\mu}$', r'$\sf \mu+jets$')
fancy_labels['e4j']   = (r'$\sf p_{T,e}$', r'$\sf e+jets$')

def reduced_objective(p, mask, p_init):
    masked_p = p_init.copy()
    masked_p[mask] = p
    return fit_data.objective(masked_p, data=toy_data, cost_type=cost_type)

def shape_morphing(f, templates, order='quadratic'):
    '''
    Efficiency shape morphing for nuisance parameters.  

    Parameters:
    ===========
    f: value of nuisance parameter
    templates: triplet of (nominal, up, down) template variations
    order: choose either a linear or quadratic variation of templates with nuisance parameter f
    '''
    t_nom  = templates[0]
    t_up   = templates[1]
    t_down = templates[2]

    if order == 'linear':
        t_eff = t_nom + f*(t_up - t_down)/2
    elif order == 'quadratic':
        t_eff = (f*(f - 1)/2)*t_down - (f - 1)*(f + 1)*t_nom + (f*(f + 1)/2)*t_up

    return t_eff


def signal_amplitudes(beta, br_tau, single_w = False):
    '''
    Returns an array of branching fractions for each signal channel.

    parameters:
    ===========
    beta : W branching fractions [beta_e, beta_mu, beta_tau, beta_h]
    br_tau : tau branching fractions [br_e, br_mu, br_h]
    single_w : if process contains a single w decay
    '''
    if single_w:
        amplitudes = np.array([beta[0],  # e 
                               beta[1],  # mu
                               beta[2]*br_tau[0],  # tau_e
                               beta[2]*br_tau[1],  # tau_mu
                               beta[2]*br_tau[2],  # tau_h
                               beta[3],  # h
                               ])
    else:
        amplitudes = np.array([beta[0]*beta[0],  # e, e
                               beta[1]*beta[1],  # mu, mu
                               2*beta[0]*beta[1],  # e, mu
                               beta[2]*beta[2]*br_tau[0]**2,  # tau_e, tau_e
                               beta[2]*beta[2]*br_tau[1]**2,  # tau_mu, tau_mu
                               2*beta[2]*beta[2]*br_tau[0]*br_tau[1],  # tau_e, tau_m
                               2*beta[2]*beta[2]*br_tau[0]*br_tau[2],  # tau_e, tau_h
                               2*beta[2]*beta[2]*br_tau[1]*br_tau[2],  # tau_mu, tau_h
                               beta[2]*beta[2]*br_tau[2]*br_tau[2],  # tau_h, tau_h
                               2*beta[0]*beta[2]*br_tau[0],  # e, tau_e
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
                               beta[3]*beta[3],  # h, h
                               ])

    return amplitudes

def signal_mixture_model(beta, br_tau, h_temp, mask=None, sample=False, single_w=False):
    '''
    Mixture model for the ttbar/tW signal model.  The output will be an array
    corresponding to a sum over the input template histograms scaled by their
    respective efficiencies and branching fraction products.

    parameters:
    ==========
    beta : branching fractions for the W decay
    br_tau : branching fractions for the tau decay
    h_temp : dataframe with template histograms for each signal component
    mask : a mask that selects a subset of mixture components
    sample : if True, the input templates will be sampled before returning
    single_w : if process contains a single w decay
    '''

    beta_init  = signal_amplitudes([0.108, 0.108, 0.108, 0.676], [0.1783, 0.1741, 0.6476], single_w)
    beta_fit   = signal_amplitudes(beta, br_tau, single_w)
    beta_ratio = beta_fit/beta_init

    if not isinstance(mask, type(None)):
        beta_ratio = mask*beta_ratio

    if sample:
        f = np.dot(np.random.poisson(h_temp), beta_ratio)
    else:
        f = np.dot(h_temp, beta_ratio)

    return f

def calculate_variance(f, x0):
    '''
    calculates variance for input function.
    '''

    hcalc = nd.Hessdiag(f)
    hobj = hcalc(x0)[0]
    var = 1./hobj

    return var

def calculate_covariance(f, x0):
    '''
    calculates covariance for input function.
    '''

    hcalc = nd.Hessian(f,
                       step        = 1e-3,
                       method      = 'forward',
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


class FitData(object):
    def __init__(self, path, selections, feature_map, nprocesses=8):
        self._selections     = selections
        self._n_selections   = len(selections)
        self._decay_map      = pd.read_csv('data/decay_map.csv').set_index('id')
        self._selection_data = {s: self._initialize_template_data(path, feature_map[s], s) for s in selections}

        # retrieve parameter configurations
        #self._pool = Pool(processes = min(16, nprocesses))
        self._initialize_parameters()

    def _initialize_template_data(self, location, target, selection):
        '''
        Gets data for given selection including:
        * data templates
        * signal templates
        * background templates
        * morphing templates for shape systematics
        * binning
        '''
        infile = open(f'{location}/{selection}_templates.pkl', 'rb')
        data = pickle.load(infile)
        infile.close()
        return data

    def _initialize_parameters(self):
        '''
        Gets parameter configuration from a file.
        '''
        df_params = pd.read_csv('data/model_parameters_partial.csv')
        df_params = df_params.set_index('name')
        #df_params = df_params.iloc[:4,]
        self._parameters = df_params

        # make a map of each shape n.p. to be considered for each selection and
        # dataset (and maybe jet bin later, if needed)
        df_shape = self._parameters.query(f'type == "shape"') 
        np_dict = dict()
        for s in self._selections:
            np_dict[s] = dict()
            for ds in pt.selection_dataset_dict[s]:
                np_dict[s][ds] = df_shape.query(f'{ds} == 1 and {s} == 1').index.values 
        self._np_dict = np_dict

    def get_selection_data(self, selection):
        return self._selection_data[selection]

    def get_params_init(self, as_array=False):
        if as_array:
            return self._parameters['val_init'].values
        else:
            return self._parameters['val_init']

    def modify_template(self, templates, pdict, dataset_name, selection):
        '''
        Modifies a single template based on all shape nuisance parameters save
        in templates dataframe.  Only applies variation in the case that there
        are a sufficient number fo events.
        '''
        t_nominal = templates['val']
        if templates.shape[1] == 2: # no systematics generated
            return t_nominal
        else:
            t_new = np.zeros(t_nominal.shape)
            #print(df_np.index.values)
            for pname in self._np_dict[selection][dataset_name]:
                t_up, t_down = templates[f'{pname}_up'], templates[f'{pname}_down']
                dt = shape_morphing(pdict[pname], (t_nominal, t_up, t_down)) - t_nominal
                t_new += dt
            t_new += t_nominal

            return t_new

    def sub_objective(self, pdict, selection, category, cat_data, data,
                      cost_type='poisson',
                      no_shape=False
                      ):
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
        no_shape : sums over all bins in input templates
        
        '''

        # unpack W and tau branching fractions
        beta   = np.array([pdict['beta_e'], pdict['beta_mu'], pdict['beta_tau'], pdict['beta_h']])
        br_tau = np.array([pdict['br_tau_e'], pdict['br_tau_mu'], pdict['br_tau_h']])

        # get the data
        templates = cat_data['templates']
        #f_data, var_data = templates['data']['val'], templates['data']['var']
        f_data, var_data = data[selection][category], data[selection][category]

        # get simulated background components and apply cross-section nuisance parameters
        f_model, var_model = np.zeros(f_data.shape), np.zeros(f_data.shape)

        # Drell-Yan
        f_model   += pdict['xs_zjets']*self.modify_template(templates['zjets_alt'], pdict, 'zjets_alt', selection)
        var_model += pdict['xs_zjets']*templates['zjets_alt']['var']

        # Diboson
        f_model   += pdict['xs_diboson']*templates['diboson']['val']
        var_model += pdict['xs_diboson']*templates['diboson']['var']

        if selection in ['etau', 'mutau']:
            f_model *= pdict['eff_tau']

        # get the signal components and apply mixing of W decay modes according to beta
        for label in ['ttbar', 't', 'wjets']:
            template_collection = templates[label]
            signal_template     = pd.DataFrame.from_dict({dm: self.modify_template(t, pdict, label, selection) for dm, t in template_collection.items()})
            #signal_template     = pd.DataFrame.from_dict({dm: t['val'] for dm, t in template_collection.items()})

            if selection in ['etau', 'mutau'] and label != 'wjets': # split real and misID taus
                mask = np.zeros(21).astype(bool)

                # real tau component (indices taken from decay_map.csv)
                mask[[6,7,8,11,14]] = True
                f_real = signal_mixture_model(beta, br_tau,
                                              h_temp   = signal_template,
                                              mask     = mask,
                                              single_w = (label == 'wjets'),
                                             )

                # apply misID nuisance parameter for "fake" taus
                mask = np.invert(mask)
                f_fake = signal_mixture_model(beta, br_tau,
                                              h_temp   = signal_template,
                                              mask     = mask,
                                              single_w = (label == 'wjets')
                                             )

                f_sig = pdict['eff_tau']*f_real + pdict['misid_tau_h']*f_fake
            else:
                f_sig = signal_mixture_model(beta, br_tau,
                                             h_temp   = signal_template,
                                             single_w = (label == 'wjets')
                                            )

                if selection in ['etau', 'mutau'] and label == 'wjets': 
                    f_sig *= pdict['misid_tau_h']

            # prepare mixture
            #f_model   += f_sig
            #var_model += var_sig # figure this out
            f_model += pdict[f'xs_{label}']*f_sig
            #var_model += pdict[f'xs_{label}']*templates[label]['var']

        # lepton efficiencies as normalization nuisance parameters
        # lepton energy scale as morphing parameters
        if selection == 'ee':
            f_model *= pdict['trigger_e']**2
            f_model *= pdict['eff_e']**2
        elif selection == 'emu':
            f_model *= pdict['trigger_mu']*pdict['trigger_e']
            f_model *= pdict['eff_e']*pdict['eff_mu']
        elif selection == 'mumu':
            f_model *= pdict['trigger_mu']**2
            f_model *= pdict['eff_mu']**2
        elif selection == 'etau':
            f_model *= pdict['trigger_e']
            f_model *= pdict['eff_e']
        elif selection == 'mutau':
            f_model *= pdict['trigger_mu']
            f_model *= pdict['eff_mu']
        elif selection == 'e4j':
            f_model *= pdict['trigger_e']
            f_model *= pdict['eff_e']
        elif selection == 'mu4j':
            f_model *= pdict['trigger_mu']
            f_model *= pdict['eff_mu']

        # apply overall lumi nuisance parameter
        f_model *= pdict['lumi']

        # get fake background and include normalization nuisance parameters
        if selection in ['etau', 'e4j']:
            #f_model   += templates['fakes']['val']
            f_model   += pdict['e_fakes']*templates['fakes']['val']
            var_model += templates['fakes']['var']

        if selection in ['mutau', 'mu4j']:
            #f_model   += templates['fakes']['val']
            f_model   += pdict['mu_fakes']*templates['fakes']['val']
            var_model += templates['fakes']['var']

        # for testing parameter estimation without estimating kinematic fit
        if no_shape:
            f_data    = np.sum(f_data)
            var_data  = np.sum(var_data)
            f_model   = np.sum(f_model)
            var_model = np.sum(var_model)

        # calculate the cost
        if cost_type == 'chi2':
            mask = var_data > 0 # + var_model > 0
            nll = (f_data[mask] - f_model[mask])**2 / (2*var_data[mask])# + var_model)
        elif cost_type == 'poisson':
            mask = f_model > 0
            nll = -f_data[mask]*np.log(f_model[mask]) + f_model[mask]

            #print(-f_data[mask].sum()*np.log(f_model[mask].sum()) + f_model[mask].sum())
            #print(nll.sum())

        cost = nll.sum()

        return cost

    def objective(self, params, data, cost_type='poisson', no_shape=False):
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
        no_shape : sums over all bins in input templates
        '''
        
        # unpack parameters here
        pdict = dict(zip(self._parameters.index.values, params))

        # calculate per category, per selection costs
        cost = 0
        for selection in self._selections:
            sdata = self.get_selection_data(selection)
            for category, cat_data in sdata.items():
                cost += self.sub_objective(pdict, selection, category, cat_data, data, cost_type, no_shape)

        # Add prior terms for nuisance parameters 
        for pname, p in pdict.items():
            #if pname in ['beta_e', 'beta_mu', 'beta_tau', 'beta_h']:
            #    continue

            p_chi2 = (p - self._parameters.loc[pname, 'val_init'])**2 / (2*self._parameters.loc[pname, 'err_init']**2)
            cost += p_chi2

        # require that the branching fractions sum to 1
        beta  = params[:4]
        cost += (1 - np.sum(beta))**2/(2e-9)  

        #print(params)
        #print(cost)

        return cost

