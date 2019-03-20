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

        # initialize branching fraction parameters
        self._beta_init   = np.array([0.108, 0.108, 0.108, 1 - 3*0.108])
        self._br_tau_init = np.array([0.1783, 0.1741, 0.6476])
        self._ww_amp_init = signal_amplitudes(self._beta_init, self._br_tau_init, single_w=False)
        self._w_amp_init  = signal_amplitudes(self._beta_init, self._br_tau_init, single_w=True)

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
        df_params = df_params.astype({'err_init':float, 'val_init':float})

        self._parameters = df_params
        self._npoi   = df_params.query('type == "poi"').shape[0]
        self._nnorm  = df_params.query('type == "norm"').shape[0]
        self._nshape = df_params.query('type == "shape"').shape[0]

        return

    def _initialize_fit_tensor(self):
        '''
        This converts the data stored in the input dataframes into a numpy tensor of
        dimensions (n_selections*n_categories*n_bins, n_processes, n_nuisances).
        '''

        params = self._parameters.query('type != "poi"')
        self._model_data = dict()
        for sel in self._selections:
            category_tensor = []
            for category, templates in self.get_selection_data(sel).items():
                templates = templates['templates']
                data_val, data_var = templates['data']['val'], templates['data']['var']
                
                norm_matrix = []  
                process_mask = []
                data_tensor = []
                for ds in self._processes:

                    # initialize mask for removing irrelevant processes
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
                        norm_vector = []
                        for pname, param in params.iterrows():
                            if param.type == 'shape' and param[sel]:
                                if f'{pname}_up' in template.columns:
                                    deff_plus  = template[f'{pname}_up'].values - val
                                    deff_minus = template[f'{pname}_down'].values - val
                                else:
                                    deff_plus  = np.zeros_like(val)
                                    deff_minus = np.zeros_like(val)
                                delta_plus.append(deff_plus + deff_minus)
                                delta_minus.append(deff_plus - deff_minus)

                            elif param.type == 'norm':
                                if param[sel] and param[ds]:
                                    norm_vector.append(1)
                                else:
                                    norm_vector.append(0)

                        process_array = np.vstack([val.reshape(1, val.size), var.reshape(1, var.size), delta_plus, delta_minus])
                        data_tensor.append(process_array.T)
                        norm_matrix.append(norm_vector)

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
                            norm_vector = []
                            for pname, param in params.iterrows():
                                if param.type == 'shape' and param[sel]:
                                    if f'{pname}_up' in sub_template.columns:
                                        deff_plus  = sub_template[f'{pname}_up'].values - val
                                        deff_minus = sub_template[f'{pname}_down'].values - val
                                    else:
                                        deff_plus  = np.zeros_like(val)
                                        deff_minus = np.zeros_like(val)
                                    delta_plus.append(deff_plus + deff_minus)
                                    delta_minus.append(deff_plus - deff_minus)
                                elif param.type == 'norm':
                                    if param[sel] and param[ds]:
                                        norm_vector.append(1)
                                    else:
                                        norm_vector.append(0)

                            process_array = np.vstack([val.reshape(1, val.size), var.reshape(1, var.size), delta_plus, delta_minus])
                            data_tensor.append(process_array.T)
                            norm_matrix.append(norm_vector)


                model_tensor = np.stack(data_tensor)
                self._model_data[f'{sel}_{category}'] = dict(
                                                             data             = (data_val, data_var),
                                                             model            = model_tensor,
                                                             process_mask     = np.array(process_mask, dtype=bool),
                                                             shape_param_mask = params.query('type == "shape"')[sel].values.astype(bool),
                                                             norm_matrix      = np.stack(norm_matrix)
                                                             )

        return
 
    # getter functions
    def get_selection_data(self, selection):
        return self._selection_data[selection]

    def get_model_data(self, category):
        return self._model_data[category]

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

    def mixture_model(self, params, category):
        '''
        produces full mixture model
        '''

        # get the model data
        model_data = self.get_model_data(category)

        # split parameters into poi, normalization, and shape
        beta, br_tau = params[:4]*self._beta_init, params[4:7]*self._br_tau_init
        norm_params  = params[self._npoi:self._npoi + self._nnorm]
        shape_params = params[self._npoi + self._nnorm:]

        # update norm parameter array
        norm_params = model_data['norm_matrix']*norm_params
        norm_params[norm_params == 0] = 1 

        # apply shape parameter mask and build array for morphing
        shape_params_masked = shape_params[model_data['shape_param_mask']]
        shape_params_masked = np.concatenate([[1, 0], 0.5*shape_params_masked**2, 0.5*shape_params_masked])

        # build the process array and apply mask (this could be improved)
        process_amplitudes = [1, 1, 1] # zjets, fakes, diboson
        process_amplitudes.extend(signal_amplitudes(beta, br_tau)/self._ww_amp_init) # ttbar, 
        process_amplitudes.extend(signal_amplitudes(beta, br_tau)/self._ww_amp_init) # t
        process_amplitudes.extend(signal_amplitudes(beta, br_tau)/self._ww_amp_init) # ww
        process_amplitudes.extend(signal_amplitudes(beta, br_tau, single_w=True)/self._w_amp_init) # wjets
        process_amplitudes = np.array(process_amplitudes)

        process_amplitudes_masked = process_amplitudes[model_data['process_mask']]
        process_amplitudes_masked = norm_params.T*process_amplitudes_masked
        process_amplitudes_masked = np.product(process_amplitudes_masked, axis=0)

        # build expectation from model_tensor
        model_tensor = model_data['model']
        model_val    = np.tensordot(model_tensor, shape_params_masked, axes=1) # n.p. modification
        model_val    = np.tensordot(model_val.T, process_amplitudes_masked, axes=1)
        model_var    = None #np.tensordot(model_tensor[:,:,1].T, process_amplitudes_masked, axes=1)

        return model_val, model_var
        
    def objective(self, params, cost_type='poisson', no_shape=False):
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

            # omit categories from calculation of cost
            if category in [
                            'ee_cat_gt2_eq0', 'ee_cat_gt2_eq0', 
                            'emu_cat_eq0_eq0_a', 'emu_cat_eq1_eq0_a', 
                            'emu_cat_eq1_eq1_a'
                            ]:
                continue
            
            # get the model and data templates
            data_val, data_var   = template_data['data']
            model_val, model_var = fh.mixture_model(params, category)

            # for testing parameter estimation while excluding kinematic shape information
            if no_shape:
                data_val    = np.sum(data_val)
                data_var  = np.sum(data_var)
                model_val   = np.sum(model_val)
                model_var = np.sum(model_var)

            # calculate the cost
            if cost_type == 'chi2':
                mask = data_var > 0 # + model_var > 0
                nll = (data_val[mask] - model_val[mask])**2 / (2*data_var[mask])# + model_var)
            elif cost_type == 'poisson':
                mask = model_val > 0
                nll = -data_val[mask]*np.log(model_val[mask]) + model_val[mask]

            cost += nll.sum()

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
