import pickle
from multiprocessing import Process, Queue, Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numdifftools as nd
from scipy.stats import lognorm, norm

import scripts.plot_tools as pt

np.set_printoptions(precision=2)

fancy_labels = dict(
                    mumu  = [r'$\sf p_{T,\mu}$', r'$\sf \mu\mu$'],
                    ee    = [r'$\sf p_{T,e}$', r'$\sf ee$'],
                    emu   = [r'$\sf p_{T,trailing}$', r'$\sf e\mu$'],
                    mutau = [r'$\sf p_{T,\tau}$', r'$\sf \mu\tau$'],
                    etau  = [r'$\sf p_{T,\tau}$', r'$\sf e\tau$'],
                    mujet = [r'$\sf p_{T,\mu}$', r'$\sf \mu+jets$'],
                    ejet  = [r'$\sf p_{T,e}$', r'$\sf e+jets$'],
                    )
features = dict(
                mumu  = 'lepton2_pt', # trailing muon pt
                ee    = 'lepton2_pt', # trailing electron pt
                emu   = 'trailing_lepton_pt', # like the name says
                mutau = 'lepton2_pt', # tau pt
                etau  = 'lepton2_pt', # tau pt
                mujet = 'lepton1_pt', # muon pt
                ejet  = 'lepton1_pt', # electron pt
                )

def signal_amplitudes(beta, br_tau, single_w=False):
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

def signal_amplitudes_jacobian(beta, br_tau, npadding, single_w=False):
    '''
    Derivatives of signal component amplitudes.
    '''

    if single_w:
        amplitudes_jac = np.array([
                                  [1, 0, 0,         0,         0,         0],
                                  [0, 1, 0,         0,         0,         0],
                                  [0, 0, br_tau[0], br_tau[1], br_tau[2], 0],
                                  [0, 0, 0,         0,         0,         1],
                                  [0, 0, beta[2],   0,         0,         0],
                                  [0, 0, 0,         beta[2],   0,         0],
                                  [0, 0, 0,         0,         beta[2],   0],
                                  ]
                                 )
        amplitudes_jac = np.vstack([amplitudes_jac, np.zeros((npadding, 6))])
    else:
        amplitudes_jac = np.array([
                                  [2*beta[0], 0, 2*beta[1], 0, 
                                      0, 0, 0, 0, 
                                      0, 2*beta[2]*br_tau[0], 2*beta[2]*br_tau[1], 2*beta[2]*br_tau[2], 
                                      0, 0, 0, 2*beta[3], 
                                      0, 0, 0, 0, 0],
                                  [0, 2*beta[1], 2*beta[0], 0, 
                                      0, 0, 0, 0, 
                                      0, 0, 0, 0, 
                                      2*beta[2]*br_tau[0], 2*beta[2]*br_tau[1], 2*beta[2]*br_tau[2], 0, 
                                      2*beta[3], 0, 0, 0, 0],
                                  [0, 0, 0, 2*beta[2]*br_tau[0]**2, 
                                      2*beta[2]*br_tau[1]**2, 4*beta[2]*br_tau[0]*br_tau[1], 4*beta[2]*br_tau[0]*br_tau[2], 4*beta[2]*br_tau[1]*br_tau[2], 
                                      2*beta[2]*br_tau[2]**2, 2*beta[0]*br_tau[0], 2*beta[0]*br_tau[1], 2*beta[0]*br_tau[2],
                                      2*beta[1]*br_tau[0], 2*beta[1]*br_tau[1], 2*beta[1]*br_tau[2], 0, 
                                      0, 2*beta[3]*br_tau[0], 2*beta[3]*br_tau[1], 2*beta[3]*br_tau[2], 0],
                                  [0, 0, 0, 0, 
                                      0, 0, 0, 0, 
                                      0, 0, 0, 0, 
                                      0, 0, 0, 2*beta[0], 
                                      2*beta[1], 2*beta[2]*br_tau[0], 2*beta[2]*br_tau[1], 2*beta[2]*br_tau[2], 2*beta[3]],
                                  [0, 0, 0, 2*beta[2]*beta[2]*br_tau[0], 
                                      0, 2*beta[2]*beta[2]*br_tau[1], 2*beta[2]*beta[2]*br_tau[2], 0, 
                                      0, 2*beta[0]*beta[2], 0, 0, 
                                      2*beta[1]*beta[2], 0, 0, 0,
                                      0, 2*beta[2]*beta[3], 0, 0, 0],
                                  [0, 0, 0, 0,
                                      2*beta[2]*beta[2]*br_tau[1], 2*beta[2]*beta[2]*br_tau[0], 0, 2*beta[2]*beta[2]*br_tau[2],
                                      0, 0, 2*beta[0]*beta[2], 0,
                                      0, 2*beta[1]*beta[2], 0, 0,
                                      0, 0, 2*beta[2]*beta[3], 0, 0],
                                  [0, 0, 0, 0,
                                      0, 0, 2*beta[2]*beta[2]*br_tau[0], 2*beta[2]*beta[2]*br_tau[1],
                                      2*beta[2]*beta[2]*br_tau[2], 0, 0, 2*beta[0]*beta[2],
                                      0, 0, 2*beta[1]*beta[2], 0,
                                      0, 0, 0, 2*beta[2]*beta[3], 0]
                                  ]
                                 )
        amplitudes_jac = np.vstack([amplitudes_jac, np.zeros((npadding,21))])

    return amplitudes_jac

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
                       method      = 'central',
                       full_output = True
                       )

    hobj = hcalc(x0)[0]
    if np.linalg.det(hobj) != 0:
        # calculate the full covariance matrix
        hinv        = np.linalg.pinv(hobj)
        sig         = np.sqrt(hinv.diagonal())
        corr_matrix = hinv/np.outer(sig, sig)

        return sig, corr_matrix
    else:
        return False, False 

# GOF statistics
def chi2_test(y1, y2, var1, var2):
    chi2 = 0.5*(y1 - y2)**2/(var1 + var2)
    return chi2

# Barlow-Beeston method for limited MC statistics
def bb_objective_aux(data_val, exp_val, exp_var):
    a = 1
    b = exp_var/exp_val - 1
    c = -data_val*exp_var/exp_val**2
    if np.any(b*b - 4*a*c < 0.):
        print(b*b - 4*a*c)

    beta_plus  = (-b + np.sqrt(b*b - 4*a*c))/2
    beta_minus = (-b - np.sqrt(b*b - 4*a*c))/2

    return beta_plus, beta_minus

class FitData(object):
    def __init__(self, path, selections, processes, 
                 param_file  = 'data/model_parameters_default.csv',
                 use_prefit  = False,
                 process_cut = 0.01,
                 veto_list   = ['ee_cat_gt2_eq0', 'mumu_cat_gt2_eq0', 
                                'ejet_cat_eq3_gt2', 'mujet_cat_eq3_gt2'
                                ],
                 debug_mode = False

                 ):
        self._selections   = selections
        self._n_selections = len(selections)
        self._processes    = processes
        self._n_processes  = len(processes)
        self._selection_data = {s: self._initialize_data(path, s) for s in selections}

        # retrieve parameter configurations
        self._decay_map = pd.read_csv('data/decay_map.csv').set_index('id')
        self._initialize_parameters(param_file, use_prefit)

        # initialize branching fraction parameters
        self._beta_init   = self._pval_init[:4]
        self._br_tau_init = self._pval_init[4:7]
        self._ww_amp_init = signal_amplitudes(self._beta_init, self._br_tau_init)
        self._w_amp_init  = signal_amplitudes(self._beta_init, self._br_tau_init, single_w=True)

        # initialize fit data
        self.veto_list = veto_list # used to remove categories from fit
        self._initialize_fit_tensor(process_cut, debug=debug_mode)

        # initialize cost (do this last)
        #self._cost_init = 0
        #self._cost_init = self.objective(self.get_params_init(as_array=True))

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

    def _initialize_parameters(self, param_file, use_prefit):
        '''
        Gets parameter configuration from a file.
        '''
        df_params = pd.read_csv(param_file)
        df_params = df_params.set_index('name')

        if use_prefit:
            df_params = df_params.astype({'err_init':float, 'val_init':float, 'err_fit':float, 'val_fit':float})
            df_params.loc[:,'val_init']  = df_params['val_fit'].values
            df_params.loc[:,'err_init']  = df_params['err_fit'].values
        else:
            df_params = df_params.astype({'err_init':float, 'val_init':float})

        self._nparams    = df_params.shape[0]
        self._npoi       = df_params.query('type == "poi"').shape[0]
        self._nnorm      = df_params.query('type == "norm"').shape[0]
        self._nshape     = df_params.query('type == "shape"').shape[0]
        self._pval_init  = df_params['val_init'].values.copy()
        self._pval_fit   = df_params['val_init'].values.copy()
        self._perr_init  = df_params['err_init'].values
        self._pmask      = df_params['active'].values.astype(bool)
        self._parameters = df_params

        # temporary handling of top pt systematic (one-sided Gaussian)
        self._pi_mask = self._pmask.copy()
        #self._pi_mask[:4] = False

        # define priors here
        self._priors = []
        for pname, pdata in df_params.iterrows():
            mu, sigma= pdata['val_init'], pdata['err_init']
            if pdata['pdf'] == 'none':
                self._priors.append(0)
            elif pdata['pdf'] == 'lognorm':
                self._priors.append(lognorm(s=np.log(1 + sigma/mu), scale=mu))
            elif pdata['pdf'] == 'gaussian':
                self._priors.append(norm(loc=mu, scale=sigma))

        return

    def _initialize_fit_tensor(self, process_cut, debug=False):
        '''
        This converts the data stored in the input dataframes into a numpy tensor of
        dimensions (n_selections*n_categories*n_bins, n_processes, n_nuisances).
        '''
        
        params = self._parameters.query('type != "poi"')
        self._model_data = dict()
        self._rnum_cache = dict()
        self._bb_np      = dict()
        self._bb_penalty = dict()
        self._categories = []
        for sel in self._selections:

            # use for older template data
            #if sel == 'ejet':
            #    sel = 'e4j'
            #if sel == 'mujet':
            #    sel = 'mu4j'

            for category, templates in self.get_selection_data(sel).items():

                # omit categories 
                if f'{sel}_{category}' in self.veto_list:
                    continue

                self._categories.append(f'{sel}_{category}') 
                templates                             = templates['templates']
                data_val, data_var                    = templates['data']['val'], templates['data']['var']
                self._rnum_cache[f'{sel}_{category}'] = np.random.randn(data_val.size)
                self._bb_np[f'{sel}_{category}']      = np.ones(data_val.size)

                if debug:
                    print('\n', sel, category)
                    print(data_val, np.sqrt(data_val.sum()), '\n')

                norm_mask    = []
                process_mask = []
                data_tensor  = []
                for ds in self._processes:

                    if sel in ['etau', 'mutau', 'emu'] and ds == 'fakes':
                        ds = 'fakes_ss'

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

                        if sel in ['etau', 'mutau', 'emu'] and ds == 'fakes_ss':
                            ds = 'fakes'
                
                    if ds in ['zjets_alt', 'diboson', 'gjets', 'fakes']: # processes that are not subdivided

                        val, var = template['val'].values, template['var'].values

                        if debug and val.sum() > 0.:
                            print(ds, val)

                        # determine whether process contribution is significant
                        # or should be masked (this should be studied for
                        # impact on poi to determine proper threshold)

                        if val.sum() == 0. or val.sum()/np.sqrt(data_var.sum()) <= process_cut:
                            process_mask.append(0)
                            continue
                        else:
                            process_mask.append(1)

                        delta_plus, delta_minus = [], []
                        norm_vector = []
                        for pname, param in params.iterrows():
                            #if debug:
                            #    print(pname, f'type : {param.type}, active : {param['active']}, {sel} : {param[sel]}, {ds} : {param[ds]}')

                            if param.type == 'shape' and param[sel]:
                                if f'{pname}_up' in template.columns and param['active'] and param[ds]:
                                    diff_plus  = template[f'{pname}_up'].values - val
                                    diff_minus = template[f'{pname}_down'].values - val

                                    if debug:
                                        print(template[['val', f'{pname}_up', f'{pname}_down']])

                                    #print(diff_plus + diff_minus, diff_plus - diff_minus)
                                else:
                                    diff_plus  = np.zeros_like(val)
                                    diff_minus = np.zeros_like(val)
                                delta_plus.append(diff_plus + diff_minus)
                                delta_minus.append(diff_plus - diff_minus)


                            elif param.type == 'norm':
                                if param[sel] and param[ds]:
                                    norm_vector.append(1)
                                else:
                                    norm_vector.append(0)

                        process_array = np.vstack([val.reshape(1, val.size), var.reshape(1, var.size), delta_plus, delta_minus])
                        data_tensor.append(process_array.T)
                        norm_mask.append(norm_vector)

                    elif ds in ['ttbar', 't', 'ww', 'wjets']: # datasets with sub-templates
                        full_sum, reduced_sum = 0, 0
                        for sub_ds, sub_template in template.items():
                            val, var = sub_template['val'].values, sub_template['var'].values
                            full_sum += val.sum()

                            # determine wheter process should be masked
                            if val.sum() == 0. or val.sum()/np.sqrt(data_var.sum()) <= process_cut:
                                process_mask.append(0)
                                continue
                            else:
                                process_mask.append(1)

                            if debug and val.sum() > 0.:
                                print(ds, sub_ds, val)

                            delta_plus, delta_minus = [], []
                            norm_vector = []
                            for pname, param in params.iterrows():
                                if param.type == 'shape' and param[sel]:
                                    if f'{pname}_up' in sub_template.columns and param[ds]: 
                                        if debug:
                                            print(sub_template[['val', f'{pname}_up', f'{pname}_down']])

                                        ## temporary modifcation to top pt morphing for ttbar templates
                                        #if ds == 'ttbar' and pname == 'top_pt':
                                        #    sub_template.loc[:,'top_pt_down'] = val

                                        diff_plus  = sub_template[f'{pname}_up'].values - val
                                        diff_minus = sub_template[f'{pname}_down'].values - val
                                    else:
                                        diff_plus  = np.zeros_like(val)
                                        diff_minus = np.zeros_like(val)
                                    delta_plus.append(diff_plus + diff_minus)
                                    delta_minus.append(diff_plus - diff_minus)

                                    #print(pname, diff_plus + diff_minus, diff_plus - diff_minus, sep='\n')

                                elif param.type == 'norm':
                                    if param[sel] and param[ds]:
                                        norm_vector.append(1)
                                    else:
                                        norm_vector.append(0)

                            process_array = np.vstack([val.reshape(1, val.size), var.reshape(1, var.size), delta_plus, delta_minus])
                            data_tensor.append(process_array.T)
                            norm_mask.append(norm_vector)

                        if debug: 
                            print(full_sum)

                self._model_data[f'{sel}_{category}'] = dict(
                                                             data             = (data_val, data_var),
                                                             model            = np.stack(data_tensor),
                                                             process_mask     = np.array(process_mask, dtype=bool),
                                                             shape_param_mask = params.query('type == "shape"')[sel].values.astype(bool),
                                                             norm_mask        = np.stack(norm_mask)
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
    def model_sums(self, selection, category, syst=None):
        '''
        This sums over all datasets/sub_datasets in selection_data for the given category.
        '''

        templates = self._selection_data[selection][category]['templates']
        outdata = np.zeros_like(templates['data']['val'], dtype=float)
        for ds, template in templates.items():
            if ds == 'data':
                continue

            if ds in ['ttbar', 't', 'ww', 'wjets']:
                for sub_ds, sub_template in template.items():
                    if syst is not None and syst in sub_template.columns:
                        outdata += sub_template[syst].values
                    else:
                        outdata += sub_template['val'].values
            else:
                if syst is not None and syst in template.columns:
                    outdata += template[syst].values
                else:
                    outdata += template['val'].values

        return outdata

    def mixture_model(self, params, category, 
                      process_amplitudes = None,
                      no_sum             = False,
                      no_var             = False,
                      debug              = False
                      ):
        '''
        Outputs mixture and associated variance for a given category.

        Parameters:
        ===========
        params: parameter values for model
        category: description of lepton/jet/b tag category
        process_amplitudes: if signal process amplitudes have been calculated
            they can be passed in, otherwise calculates values based on input
            parameters
        no_sum: (default False) if set to True, will not sum across the process dimension
        '''

        # get the model data
        model_data   = self.get_model_data(category)

        # update norm parameter array
        norm_params  = params[self._npoi:self._npoi + self._nnorm]
        norm_mask       = model_data['norm_mask'].astype(bool)
        norm_param_prod = np.product(np.ones_like(norm_mask)*norm_params, axis=1, where=norm_mask)

        if debug:
            print(norm_param_prod)

        # apply shape parameter mask and build array for morphing.  When shape
        # parameter values are in the range [-1, 1] there is a quadratic
        # interpolation between those values.  Beyond that range the morphing
        # is linear.  
        shape_params = params[self._npoi + self._nnorm:]
        shape_params = shape_params[model_data['shape_param_mask']]
        sp_positive = 0.5*shape_params**2 # values in [-1, 1]
        sp_plus_mask, sp_minus_mask = (shape_params > 1), (shape_params < -1)
        sp_positive[sp_plus_mask]   = shape_params[sp_plus_mask] - 0.5 # params > 1
        sp_positive[sp_minus_mask]  = -shape_params[sp_minus_mask] - 0.5 # params < -1
        shape_params = np.concatenate([[1, 0], sp_positive, 0.5*shape_params])

        # get calculate process_amplitudes
        if process_amplitudes is None:
            beta, br_tau = params[:4], params[4:7]
            ww_amp = signal_amplitudes(beta, br_tau)/self._ww_amp_init
            w_amp  = signal_amplitudes(beta, br_tau, single_w=True)/self._w_amp_init
            process_amplitudes = np.concatenate([ww_amp, ww_amp, ww_amp, w_amp, [1, 1, 1, 1]])
            #self._process_amplitudes = process_amplitudes

        # mask the process amplitudes for this category and apply normalization parameters
        process_amplitudes = process_amplitudes[model_data['process_mask']]
        process_amplitudes = norm_param_prod.T*process_amplitudes

        # build expectation from model_tensor and propogate systematics
        model_tensor = model_data['model']
        model_val = np.tensordot(model_tensor, shape_params, axes=1) # n.p. modification
        if no_sum:
            model_val = model_val.T*process_amplitudes
            model_var = model_tensor[:,:,1].T*process_amplitudes 
        else:
            model_val = np.tensordot(model_val.T, process_amplitudes, axes=1)
            #model_var    = np.tensordot(model_tensor[:,:,1].T, process_amplitudes, axes=1)
            model_var = model_tensor[:,:,1].sum(axis=0)#*process_amplitudes 

        if debug:
            print(shape_params)
            print(process_amplitudes)
            for i, layer in enumerate(model_tensor):
                for j, sublayer in enumerate(layer):
                    print(i, j, sublayer)


        if no_var:
            return model_val
        else:
            return model_val, model_var

    def mixture_model_jacobian(self, params, category, process_amplitudes=None):
        '''
        Outputs mixture and associated variance for a given category.

        Parameters:
        ===========
        params: parameter values for model
        category: description of lepton/jet/b tag category
        process_amplitudes: if signal process amplitudes have been calculated
                            they can be passed in, otherwise calculates values based on input
                            parameters
        '''

        # get the model data
        model_data = self.get_model_data(category)

        # Calculate the normalization parameter products
        norm_params  = params[self._npoi:self._npoi + self._nnorm]
        norm_mask = model_data['norm_mask'].astype(bool)
        norm_params_prod = np.product(np.ones_like(norm_mask)*norm_params, axis=1, where=norm_mask)

        # norm parameter jacobian
        norm_params_jac = np.identity(norm_params.size) + (1 - np.identity(norm_params.size))*norm_params 
        norm_params_jac = np.array([np.product(norm_params_jac, axis=1, where=m)*m for m in norm_mask])
        norm_params_jac = np.hstack([np.zeros([norm_mask.shape[0], self._npoi]), 
                                     norm_params_jac, 
                                     np.zeros([norm_mask.shape[0], self._nshape])
                                     ])

        # apply shape parameter mask and build array for morphing
        shape_params = params[self._npoi + self._nnorm:]
        shape_params = shape_params[model_data['shape_param_mask']]
        sp_positive = 0.5*shape_params**2 # values in [-1, 1]
        sp_plus_mask, sp_minus_mask = (shape_params > 1), (shape_params < -1)
        sp_positive[sp_plus_mask]   = shape_params[sp_plus_mask] - 0.5 # params > 1
        sp_positive[sp_minus_mask]  = -shape_params[sp_minus_mask] - 0.5 # params < -1
        shape_params_arr = np.concatenate([[1, 0], sp_positive, 0.5*shape_params])

        # shape parameter jacobian
        sp_matrix = np.identity(self._nshape)[:, model_data['shape_param_mask']]
        sp_matrix = np.vstack([np.zeros([self._npoi+self._nnorm, shape_params.size]), sp_matrix])
        sp_positive_der = shape_params.copy()
        sp_positive_der[sp_plus_mask] = 1
        sp_positive_der[sp_minus_mask] = -1
        shape_params_jac = np.hstack([np.zeros([self._nparams, 2]), sp_positive_der*sp_matrix, 0.5*sp_matrix])

        # get the signal amplitudes and build process amplitudes
        beta, br_tau = params[:4], params[4:7]
        if process_amplitudes is None:
            ww_amp = signal_amplitudes(beta, br_tau)/self._ww_amp_init
            w_amp  = signal_amplitudes(beta, br_tau, single_w=True)/self._w_amp_init
            process_amplitudes = np.concatenate([ww_amp, ww_amp, ww_amp, w_amp, [1, 1, 1, 1]])

        # apply mask
        process_mask = model_data['process_mask'].astype(bool)
        process_amplitudes = process_amplitudes[process_mask]

        # do the same for the signal amplitude jacobians
        ww_amp_jac = signal_amplitudes_jacobian(beta, br_tau, params.size - 7)/self._ww_amp_init
        w_amp_jac  = signal_amplitudes_jacobian(beta, br_tau, params.size - 7, single_w=True)/self._w_amp_init
        process_amplitudes_jac = np.concatenate([ww_amp_jac, ww_amp_jac, ww_amp_jac, w_amp_jac, np.zeros((params.size, 4))], axis=1)
        process_amplitudes_jac = process_amplitudes_jac[:,process_mask]

        # combine everything together
        A1 = np.einsum('i,jk->ijk', norm_params_prod.T*process_amplitudes, shape_params_jac)
        A2 = np.einsum('i,jk->ijk', shape_params_arr, process_amplitudes_jac*norm_params_prod)
        A3 = np.einsum('i,jk->ijk', shape_params_arr, process_amplitudes*norm_params_jac.T)
        A = np.transpose(A1, (2, 1, 0)) + A2 + A3

        model_tensor = model_data['model']
        model_val_jac = np.einsum('ijk,kli->jl', model_tensor, A) # n.p. modification

        return model_val_jac
        
    def objective(self, params,
                  data                = None,
                  cost_type           = 'poisson',
                  do_bb_lite          = False,
                  no_shape            = False,
                  randomize_templates = False,
                  factorize_nll       = False,
                  lu_test             = None
                 ):
        '''
        Cost function for MC data model.  This version has no background
        compononent and is intended for fitting toy data generated from the signal
        MC.

        Parameters:
        ===========
        params: numpy array of parameters.  The first four are the W branching
                fractions, all successive ones are nuisance parameters.
        data: dataset to be fitted
        cost_type: either 'chi2' or 'poisson'
        no_shape: sums over all bins in input templates
        do_bb_lite: include bin-by-bin Barlow-Beeston parameters accounting for limited MC statistics
        randomize_templates: displaces the prediction in each bin by a fixed, random amount.
        lu_test: for testing of lepton universality
                * if 0 then all leptonic W branching fractions are equal
                * if 1 then e and mu leptonic W branching fractions are equal, tau is different
                * else all W branching fractions vary independently
        '''

        # apply mask to parameters
        params_reduced = self._pval_fit.copy()
        params_reduced[self._pmask] = params
        params = params_reduced

        # build the process amplitudes (once per evaluation) 
        beta, br_tau  = params[:4], params[4:7]
        ww_amp = signal_amplitudes(beta, br_tau)/self._ww_amp_init
        w_amp  = signal_amplitudes(beta, br_tau, single_w=True)/self._w_amp_init
        process_amplitudes = np.concatenate([ww_amp, ww_amp, ww_amp, w_amp, [1, 1, 1, 1]]) 

        # calculate per category, per selection costs
        cost = 0
        for category, template_data in self._model_data.items():

            if category in self.veto_list:
                continue

            # get the model and data templates
            model_val, model_var = self.mixture_model(params, category, process_amplitudes, no_sum=False)
            if randomize_templates:
                model_val += self._rnum_cache[category]*np.sqrt(model_var)

            if data is None:
                data_val, data_var = template_data['data']
            else:
                data_val, data_var = data[category]

            # for testing parameter estimation while excluding kinematic shape information
            if no_shape: 
                data_val  = np.sum(data_val)
                data_var  = np.sum(data_var)
                model_val = np.sum(model_val)
                model_var = np.sum(model_var)

            #print(category)
            #print(data_val)
            #print(model_val)

            # include effect of MC statisitcs (important that this is done
            # AFTER no_shape condition so inputs are integrated over)
            if do_bb_lite:

                # update bin-by-bin amplitudes
                bin_amp = bb_objective_aux(data_val, model_val, model_var)[0]
                model_val *= bin_amp
                self._bb_np[category] = bin_amp # save BB n.p.

                # add deviation of amplitudes to cost (assume Gaussian penalty)
                bb_penalty = (bin_amp - 1)**2/(2*model_var/model_val**2)
                cost += np.sum(bb_penalty)
                self._bb_penalty[category] = bb_penalty

            # calculate the cost
            if cost_type == 'poisson':
                mask = (model_val > 0) & (data_val > 0)
                nll = -data_val[mask]*np.log(model_val[mask]) + model_val[mask] \
                      + data_val[mask]*np.log(data_val[mask]) - data_val[mask]
            elif cost_type == 'chi2':
                mask = data_var + model_var > 0
                nll = 0.5*(data_val[mask] - model_val[mask])**2 / (data_var[mask] + model_var[mask])

            cost += nll.sum()

        # Add prior constraint terms for nuisance parameters 
        mask = self._pi_mask
        pi_param = (params[mask] - self._pval_init[mask])**2 / (2*self._perr_init[mask]**2)
        cost += pi_param.sum()
        self._np_cost = pi_param

        # require that the branching fractions sum to 1
        cost += (np.sum(beta) - 1)**2/1e-10

        # constraining branching fractions for lepton universality testing (maybe not the best way)
        if lu_test == 0:
            cost += (beta[0] - beta[1])**2/1e-10
            cost += (beta[0] - beta[2])**2/1e-10
        elif lu_test == 1:
            cost += (beta[0] - beta[1])**2/1e-10

        # do the same for the tau branching fraction
        #cost += (np.sum(br_tau) - 1)**2/1e-3

        return cost

    def objective_jacobian(self, params, 
                           data                = None,
                           do_bb_lite          = False,
                           randomize_templates = False,
                           no_shape            = False,
                           lu_test             = None,
                          ):
        '''
        Returns the jacobian of the objective.

        Parameters:
        ===========
        params : numpy array of parameters.  The first four are the W branching
                 fractions, all successive ones are nuisance parameters.
        data : dataset to be fitted
        randomize_templates: displaces the prediction in each bin by a fixed, random amount.
        lu_test: for testing of lepton universality
                * if 0 then all leptonic W branching fractions are equal
                * if 1 then e and mu leptonic W branching fractions are equal, tau is different
                * else all W branching fractions vary independently
        '''

        # apply mask to parameters
        params_reduced = self._pval_fit.copy()
        params_reduced[self._pmask] = params
        params = params_reduced

        # build the process amplitudes (once per evaluation, this should be
        # modified to infer the correct dimension and placement of values) 
        beta, br_tau = params[:4], params[4:7]
        ww_amp = signal_amplitudes(beta, br_tau)/self._ww_amp_init
        w_amp  = signal_amplitudes(beta, br_tau, single_w=True)/self._w_amp_init
        process_amplitudes = np.concatenate([ww_amp, ww_amp, ww_amp, w_amp, [1, 1, 1, 1]]) 

        # calculate per category, per selection costs
        dcost = np.zeros(params.size)
        for category, template_data in self._model_data.items():

            if category in self.veto_list:
                continue

            # get the model and data templates
            model_val, model_var = self.mixture_model(params, category, process_amplitudes)
            if randomize_templates:
                model_val += self._rnum_cache[category]*np.sqrt(model_var)

            if data is None:
                data_val, data_var = template_data['data']
            else:
                data_val, data_var = data[category]

            # get the jacobian of the model
            model_jac = self.mixture_model_jacobian(params, category, process_amplitudes)

            # for testing parameter estimation while excluding kinematic shape information
            if no_shape: 
                data_val  = np.sum(data_val)
                data_var  = np.sum(data_var)
                model_val = np.sum(model_val)
                model_var = np.sum(model_var)

                # just testing this part out, not sure if correct yet
                model_jac = model_jac.sum(axis=0)

            if do_bb_lite:
                # update bin-by-bin amplitudes
                bin_amp = bb_objective_aux(data_val, model_val, model_var)[0]
                model_val *= bin_amp
                if no_shape:
                    model_jac = model_jac*bin_amp
                else:
                    model_jac = model_jac*bin_amp.reshape(model_jac.shape[0], 1)

                # add deviation of amplitudes to cost (this is not needed as
                # long as bb amplitudes are calculated analytically)
                #bb_penalty_jac = (bin_amp - 1)/(model_var/model_val**2)
                #dcost += bb_penalty_jac

            # calculate the jacobian of the NLL
            mask = (model_val > 0) & (data_val > 0)
            nll_jac = np.dot(model_jac.T[:,mask], (1 - data_val[mask]/model_val[mask]))

            dcost += nll_jac

        # Add prior constraint terms for nuisance parameters 
        mask = self._pi_mask
        pi_param_jac = (params[mask] - self._pval_init[mask]) / self._perr_init[mask]**2
        dcost[mask] += pi_param_jac

        dcost[:4] += 2*(np.sum(beta) - 1)/1e-10

        # constraining branching fractions for lepton universality testing (maybe not the best way)
        if lu_test == 0:
            dcost[0] += 2*(beta[0] - beta[1])/1e-10
            dcost[0] += 2*(beta[0] - beta[2])/1e-10
            dcost[1] += -2*(beta[0] - beta[1])/1e-10
            dcost[2] += -2*(beta[0] - beta[2])/1e-10
        elif lu_test == 1:
            dcost[0] += 2*(beta[0] - beta[1])/1e-10
            dcost[1] += -2*(beta[0] - beta[1])/1e-10

        #dcost += 2*(np.sum(br_tau) - 1)/1e-4
        return dcost[self._pmask]

