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

    np.set_printoptions(precision=2)
    #print((f*(f - 1)/2), (f - 1)*(f + 1), (f*(f + 1)/2))

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

# covariance approximators
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

# GOF statistics
def chi2_test(y1, y2, var1, var2):
    chi2 = 0.5*(y1 - y2)**2/(var1 + var2)
    return chi2


class FitData(object):
    def __init__(self, path, selections, processes):
        self._selections     = selections
        self._processes      = processes
        self._n_selections   = len(selections)
        self._n_processess   = len(processes)
        self._selection_data = {s: self._initialize_data(path, s) for s in selections}

        # retrieve parameter configurations
        self._decay_map = pd.read_csv('data/decay_map.csv').set_index('id')
        self._initialize_parameters()

        # initialize fit data
        self._initialize_fit_tensor()
        self._cost_init = 0

    # initialization functions
    def _initialize_data(self, location, selection):
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
        df_params = pd.read_csv('data/model_parameters.csv')
        df_params = df_params.set_index('name')
        self._parameters = df_params

        return

    def _initialize_fit_tensor(self):
        '''
        This converts the data stored in the input dataframes into a numpy tensor of
        dimensions (n_selections*n_categories*n_bins, n_processes, n_nuisances).
        '''

        params = self._parameters.query(f'type != "poi"')
        self._category_data = dict()
        for sel in self._selections:
            category_tensor = []
            for category, templates in self.get_selection_data(sel).items():
                templates = templates['templates']
                data_val, data_var = templates['data']['val'], templates['data']['var']
                
                process_mask = []
                data_tensor = []
                for ds in self._processes:

                    # mask out missing processes
                    if ds not in templates.keys():
                        if ds in ['ttbar', 't', 'ww']:
                            process_mask.extend(21*[0,])
                        elif ds == 'wjets':
                            process_mask.extend(6*[0,])
                        else:
                            process_mask.append(0)
                        continue
                    else:
                        template = templates[ds]

                    if ds in ['zjets_alt', 'diboson']: # processes that are not subdivided
                        val, var = template['val'].values, template['var'].values
                        #print(ds, val/np.sqrt(data_var))

                        # determine whether process contribution is significant
                        # or should be masked (this should be studied for
                        # impact on poi to determine proper threshold)
                        if val.sum()/np.sqrt(data_var.sum()) <= 0.1:
                            process_mask.append(0)
                            continue
                        else:
                            process_mask.append(1)

                        delta_plus, delta_minus = [], []
                        for pname, param in params.iterrows():
                            if not params.loc[pname][sel]:
                                continue

                            if f'{pname}_up' in template.columns and param.type == 'shape':
                                deff_plus = template[f'{pname}_up'].values - val
                                deff_minus = template[f'{pname}_down'].values - val
                            elif param.type == 'norm' and param[sel] and param[ds]:
                                deff_plus = val*(1 + param['err_init'])
                                deff_minus = val*(1 - param['err_init'])
                            else:
                                deff_plus  = np.zeros_like(val)
                                deff_minus = np.zeros_like(val)

                            delta_plus.append(deff_plus + deff_minus)
                            delta_minus.append(deff_plus - deff_minus)

                        process_array = np.vstack([val.reshape(1, val.size), var.reshape(1, var.size), delta_plus, delta_minus])
                        data_tensor.append(process_array.T)

                    elif ds in ['ttbar', 't', 'ww', 'wjets']: # datasets with sub-templates
                        full_sum, reduced_sum = 0, 0
                        for sub_ds, sub_template in template.items():
                            val, var = sub_template['val'].values, sub_template['var'].values
                            full_sum += val.sum()

                            # determine wheter process should be masked
                            if val.sum()/np.sqrt(data_var.sum()) <= 0.1:
                                process_mask.append(0)
                                continue
                            else:
                                process_mask.append(1)

                            delta_plus, delta_minus = [], []
                            for pname, param in params.iterrows():
                                if not params.loc[pname][sel]:
                                    continue

                                if f'{pname}_up' in sub_template.columns and param.type == 'shape':
                                    deff_plus = sub_template[f'{pname}_up'].values - val
                                    deff_minus = sub_template[f'{pname}_down'].values - val
                                elif param.type == 'norm' and param[sel] and param[ds]:
                                    deff_plus = val*param['err_init']
                                    deff_minus = -val*param['err_init']
                                else:
                                    deff_plus  = np.zeros_like(val)
                                    deff_minus = np.zeros_like(val)

                                delta_plus.append(deff_plus + deff_minus)
                                delta_minus.append(deff_plus - deff_minus)
                            process_array = np.vstack([val.reshape(1, val.size), var.reshape(1, var.size), delta_plus, delta_minus])
                            data_tensor.append(process_array.T)
                
                category_tensor = np.stack(data_tensor)
                self._category_data[f'{sel}_{category}'] = (category_tensor, np.array(process_mask), params[sel].values)

        return
 
    # getter functions
    def get_selection_data(self, selection):
        return self._selection_data[selection]

    def get_params_init(self, as_array=False):
        if as_array:
            return self._parameters['val_init'].values
        else:
            return self._parameters['val_init']

    def get_errs_init(self, as_array=False):
        if as_array:
            return self._parameters['err_init'].values
        else:
            return self._parameters['err_init']

    def get_fit_tensor(self, category):
        return self._category_data[category]

    # model building
    def model_sums(self, selection, category):
        '''
        This sums overall datasets/sub_datasets in selection_data for the given category.
        '''

	templates = self._selection_data[selection][category]['templates'] 
	outdata = np.zeros_like(templates['data']['val'], dtype=float)
        for ds, template in templates.items():
            if ds == 'data':
                continue 

            if ds in ['ttbar', 't', 'ww', 'wjets']:
                for sub_ds, sub_template in template.items():
                    outdata += sub_template['val'].values
            else:
                outdata += template['val'].values

        return outdata

    def mixture_model(self, selection, category):
        '''
        produces full mixture model
        '''

        # things that could possibly be available globally
        br_tau_init = [0.1783, 0.1741, 0.6476]
        beta_init = [0.108, 0.108, 0.108, 1 - 3*0.108] 


	# build the process array
	w_amp_init  = fh.signal_amplitudes(beta_init, br_tau, single_w=True)
	ww_amp_init = fh.signal_amplitudes(beta_init, br_tau, single_w=False)
	process_amplitudes = []
	for process in processes:
	    if process in ['zjets_alt', 'fakes', 'diboson']:
		process_amplitudes.append(1)
	    elif process in ['ttbar', 't', 'ww']:
		process_amplitudes.extend(fh.signal_amplitudes(beta_init, br_tau_init)/ww_amp_init)
	    elif process == 'wjets':
		process_amplitudes.extend(fh.signal_amplitudes(beta_init, br_tau_init, single_w=True)/w_amp_init)
		
	process_amplitudes = np.array(process_amplitudes) 

        model_tensor, process_mask, param_mask = fit_data.get_fit_tensor(f'{selection}_{category}')
        param_array_masked = param_array[param_mask.astype(bool)]
        param_array_masked = np.concatenate([[1, 0], 0.5*param_array_masked**2, 0.5*param_array_masked])
        process_amplitudes_masked = process_amplitudes[process_mask.astype(bool)]

        # build expectation from model_tensor
        expected_pre = np.tensordot(model_tensor[:,:,0].T, process_amplitudes_masked, axes=1) # mixture model
        expected_var = np.tensordot(model_tensor[:,:,1].T, process_amplitudes_masked, axes=1) # mixture model
        expected_post = np.tensordot(model_tensor, param_array_masked, axes=1) # n.p. modification
        expected_post = np.tensordot(expected_post.T, process_amplitudes_masked, axes=1) # mixture model


	templates = self._selection_data[selection][category]['templates'] 

        return outdata
        

    def modify_template(self, templates, pdict, dataset, selection, category, sub_ds=None):
        '''
        Updates 
        '''
        pass

    # evaluation of objective functions
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
        f_model   += self.modify_template(templates['zjets_alt'], pdict, 'zjets_alt', selection, category)
        var_model += templates['zjets_alt']['var']

        # non-WW diboson
        f_model   += pdict['xs_diboson']*templates['diboson']['val']
        var_model += pdict['xs_diboson']*templates['diboson']['var']

        if selection in ['etau', 'mutau']:
            f_model *= pdict['eff_tau']

        # get the signal components and apply mixing of W decay modes according to beta
        for label in ['ttbar', 't', 'ww', 'wjets']:
            template_collection = templates[label]
            signal_template = pd.DataFrame.from_dict({dm: self.modify_template(t, pdict, label, selection, category, dm) for dm, t in template_collection.items()})
            #signal_template     = pd.DataFrame.from_dict({dm: t['val'] for dm, t in template_collection.items()})

            if selection in ['etau', 'mutau']: # split real and misID taus
                if label != 'wjets': 

                    # real tau component (indices taken from decay_map.csv)
                    mask = np.zeros(21).astype(bool)
                    mask[[6,7,8,11,14]] = True
                    f_real = signal_mixture_model(beta, br_tau,
                                                  h_temp   = signal_template,
                                                  mask     = mask,
                                                 )
                    f_real *= pdict['eff_tau']

                    # apply misID nuisance parameter for jets faking taus
                    mask = np.zeros(21).astype(bool)
                    mask[[15, 16, 17, 18, 19, 20]] = True
                    f_fake_h = signal_mixture_model(beta, br_tau,
                                                    h_temp   = signal_template,
                                                    mask     = mask,
                                                   )
                    f_fake_h *= pdict['misid_tau_h']

                    # e faking tau
                    mask = np.zeros(21).astype(bool)
                    if selection == 'etau':
                        mask[[0, 3, 9]] = True
                        f_fake_e = signal_mixture_model(beta, br_tau,
                                                      h_temp   = signal_template,
                                                      mask     = mask,
                                                     )
                        f_fake_e *= pdict['misid_tau_e']

                    elif selection == 'mutau':
                        mask[[2, 5, 12]] = True
                        f_fake_e = signal_mixture_model(beta, br_tau,
                                                      h_temp   = signal_template,
                                                      mask     = mask,
                                                     )
                        f_fake_e *= pdict['misid_tau_e']

                    f_sig = f_real + f_fake_h + f_fake_e
                else:
                    f_sig = signal_mixture_model(beta, br_tau,
                                                 h_temp   = signal_template,
                                                 single_w = True
                                                )
                    f_sig *= pdict['misid_tau_h']
            else:
                f_sig = signal_mixture_model(beta, br_tau,
                                             h_temp   = signal_template,
                                             single_w = (label == 'wjets')
                                            )

            # prepare mixture (ttbar normalization is a shape n.p.)
            if label != 'ttbar':
                f_model += pdict[f'xs_{label}']*f_sig
                #var_model += pdict[f'xs_{label}']*templates[label]['var']
            else:
                f_model   += f_sig
                #var_model += var_sig # figure this out

        # lepton trigger efficiencies (move these to shape nuisances)
        if selection == 'ee':
            f_model *= pdict['trigger_e']**2
            #f_model *= pdict['eff_e']**2
        elif selection == 'emu':
            f_model *= pdict['trigger_mu']*pdict['trigger_e']
        elif selection == 'mumu':
            f_model *= pdict['trigger_mu']**2
        elif selection == 'etau':
            f_model *= pdict['trigger_e']
            #f_model *= pdict['eff_e']
        elif selection == 'mutau':
            f_model *= pdict['trigger_mu']
        elif selection == 'e4j':
            f_model *= pdict['trigger_e']
            #f_model *= pdict['eff_e']
        elif selection == 'mu4j':
            f_model *= pdict['trigger_mu']

        # apply overall lumi nuisance parameter
        f_model *= pdict['lumi']

        # data-driven estimates here (do not apply simulation/theory uncertainties)
        # get fake background and include normalization nuisance parameters
        if selection == 'e4j':
            #f_model   += templates['fakes']['val']
            f_model   += pdict['e_fakes']*templates['fakes']['val']
            var_model += templates['fakes']['var']

        if selection == 'mu4j':
            #f_model   += templates['fakes']['val']
            f_model   += pdict['mu_fakes']*templates['fakes']['val']
            var_model += templates['fakes']['var']

        if selection == 'etau':
            #f_model   += templates['fakes']['val']
            f_model   += pdict['e_fakes_ss']*templates['fakes']['val']
            var_model += templates['fakes']['var']

        if selection == 'mutau':
            #f_model   += templates['fakes']['val']
            f_model   += pdict['mu_fakes_ss']*templates['fakes']['val']
            var_model += templates['fakes']['var']

        # for testing parameter estimation while excluding kinematic shape information
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
        for category, template_data in self._category_data.items():

            # remove 0 b tag category for ee and mumu channels#
            if category in ['ee_cat_gt2_eq0', 'ee_cat_gt2_eq0']:
                continue

            # remove additional (WW & Z+jets) categories for emu channels#
            if selection == 'emu' and category in ['cat_eq0_eq0_a', 'cat_eq1_eq0_a', 'cat_eq1_eq1_a']:
                continue

            cost += self.sub_objective(pdict, selection, category, cat_data, data, cost_type, no_shape)

        # Add prior terms for nuisance parameters 
        for pname, p in pdict.items():
            p_chi2 = (p - self._parameters.loc[pname, 'val_init'])**2 / (2*self._parameters.loc[pname, 'err_init']**2)
            cost += p_chi2

        # require that the branching fractions sum to 1
        beta  = params[:4]
        cost += (1 - np.sum(beta))**2/(2e-9)  

        # testing if this helps with precision
        cost -= self._cost_init

        #print(params)
        #print(cost)

        return cost
