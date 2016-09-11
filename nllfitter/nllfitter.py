from __future__ import division
from itertools import product

import numpy as np
import numdifftools as nd
from scipy.optimize import minimize
from lmfit import Parameter, Parameters, report_fit

class NLLFitter:
    '''
    Class for estimating PDFs using negative log likelihood minimization.  Fits
    a Model class to a dataset.    

    Parameters:
    ==========
    model    : a Model object or and array of Model objects
    data     : the dataset or datasets we wish to carry out the modelling on
    min_algo : algorith used for minimizing the nll (uses available scipy.optimize algorithms)
	verbose  : control verbosity of fit method
    fcons    : constraint function; should take arguments of the form (sig_pdf, params)
    '''
    def __init__(self, model, min_algo='SLSQP', verbose=True, lmult=(0., 0.), fcons=None):
       self.model     = model
       self.min_algo  = min_algo
       self.verbose   = verbose
       self._lmult    = lmult
       self._fcons    = fcons

    def _objective(self, params, data):
        '''
        Default objective function.  Perhaps it would make sense to make this
        easy to specify.  Includes L1 and L2 regularization terms which might
        be problematic...
        
        Parameters:
        ==========
        a: model parameters in an numpy array
        '''

        obj = 0.
        if self._fcons:
            obj += self._fcons(self.model._pdf, params)

        nll = self.model.calc_nll(data, params)
        if nll is not np.nan:
            obj += nll

        return nll + self._lmult[0] * np.sum(np.abs(params)) + self._lmult[1] * np.sum(params**2)

    def _get_corr(self, data, params):

        f_obj   = lambda a: self._objective(a, data)
        hcalc   = nd.Hessian(f_obj, step=0.01, method='central', full_output=True) 
        hobj    = hcalc(params)[0]
        hinv    = np.linalg.inv(hobj)

        # get uncertainties on parameters
        sig = np.sqrt(hinv.diagonal())

        # calculate correlation matrix
        mcorr = hinv/np.outer(sig, sig)

        return sig, mcorr

    def fit(self, data, min_algo='SLSQP', params_init=None, calculate_corr=True):
        '''
        Fits the model to the given dataset using scipy.optimize.minimize.
        Returns the fit result object.

        Parameter:
        ==========
        data           : dataset to be fit the model to
        min_algo       : minimization algorithm to be used (defaults to SLSQP
                         since it doesn't require gradient information and accepts bounds and
                         constraints).
        params_init    : initialization parameters; if not specified, current values are used
        calculate_corr : specify whether the covariance matrix should be
                         calculated.  If true, this will do a numerical calculation of the
                         covariance matrix based on the currenct objective function about the
                         minimum determined from the fit
        '''

        if params_init: 
            self.model.update_params(params_init)
        else:
            params_init = self.model.get_parameters(by_value=True)

        result = minimize(self._objective, 
                          params_init,
                          method = self.min_algo, 
                          bounds = self.model.get_bounds(),
                          #constraints = self.model.get_constraints(),
                          args   = (data)
                          )
        if self.verbose:
            print 'Fit finished with status: {0}'.format(result.status)

        if result.status == 0:
            if calculate_corr:
                sigma, corr = self._get_corr(data, result.x)
            else:
                sigma, corr = result.x, 0.

            self.model.update_parameters(result.x, (sigma, corr))

            if self.verbose:
                report_fit(self.model.get_parameters(), show_correl=False)
                print ''
                print '[[Correlation matrix]]'
                print corr, '\n'

        return result	

    def scan(self, scan_params, data, amps=None):
        '''
        Fits model to data while scanning over give parameters.

        Parameters:
        ===========
        scan_params : ScanParameters class object specifying parameters to be scanned over
        data        : dataset to fit the models to
        amps        : indices of signal amplitude parameters
        '''
        
        ### Save bounds for parameters to be scanned so that they can be reset
        ### when finished
        saved_bounds = {}
        params = self.model.get_parameters()
        for name in scan_params.names:
            saved_bounds[name] = (params[name].min, params[name].max)

        nllscan     = []
        dofs        = [] # The d.o.f. of the field will vary depending on the amplitudes
        best_params = 0.
        nll_min     = 1e9
        scan_vals, scan_div = scan_params.get_scan_vals()
        for i, scan in enumerate(scan_vals):
            ### set bounds of model parameters being scanned over
            for j, name in enumerate(scan_params.names):
                self.model.set_bounds(name, scan[j], scan[j]+scan_div[j])
                self.model.set_parameter_value(name, scan[j])

            ### Get initialization values
            params_init = [p.value for p in params.values()]
            result = minimize(self._objective, 
                              params_init,
                              method = self.min_algo, 
                              bounds = self.model.get_bounds(),
                              #constraints = self.model.get_constraints(),
                              args   = (data)
                              )

            if result.status == 0:
                nll = self.model.calc_nll(data, result.x)
                nllscan.append(nll)
                if nll < nll_min:
                    best_params = result.x
                    nll_min = nll

                if amps:
                    dofs.append(np.sum(result.x[amps] > 0.0001))
            else:
                continue

        ## Reset parameter bounds
        for name in scan_params.names:
            self.model.set_bounds(name, saved_bounds[name][0], saved_bounds[name][1])

        nllscan = np.array(nllscan)
        dofs = np.array(dofs)
        return nllscan, best_params, dofs

class ScanParameters:
    '''
    Class for defining parameters for scanning over fit parameters.
    Parameters
    ==========
    names: name of parameters to scan over
    bounds: values to scan between (should be an array with 2 values)
    nscans: number of scan points to consider
    '''
    def __init__(self, names, bounds, nscans, fixed=False):
        self.names  = names
        self.bounds = bounds
        self.nscans = nscans
        self.init_scan_params()

    def init_scan_params(self):
        scans = []
        div   = []
        for n, b, ns in zip(self.names, self.bounds, self.nscans):
            scans.append(np.linspace(b[0], b[1], ns))
            div.append(np.abs(b[1] - b[0])/ns)
        self.div   = div
        self.scans = scans

    def get_scan_vals(self, ):
        '''
        Return an array of tuples to be scanned over.
        '''
        scan_vals = list(product(*self.scans))
        return scan_vals, self.div

