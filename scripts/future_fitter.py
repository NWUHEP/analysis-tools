from timeit import default_timer as timer

import pandas as pd
import numpy as np
import numdifftools as nd

#from scipy import stats
from scipy.stats import chi2, norm 
from scipy.optimize import minimize

from collections import OrderedDict

from fitter import get_data, scale_data, fit_plot

# global options
np.set_printoptions(precision=3.)

class Model:
    '''
    Class for carrying model information.  Needs to be provided with function
    for the background and signal models in the format p(data, parameters)

    Parameters
    ==========
    model  : a function describing the model that takes argurements (params, data)
    params : names of parameters
    parinit : (optional) initial values for the parameters.  Useful when dealing with nonstandard pdfs.
    '''
    def __init__(self, model, params, parinit=None):
        
        self.model     = model
        self.parnames  = params
        self.params    = np.array(len(params)*[1.,])
        self.param_err = np.array(len(params)*[1.,])
        self.corr      = None

        self.bounds       = np.size(params)*[(None, None), ]
        self.constaints   = None
    
    def set_bounds(self, bounds):
        '''Returns bounds for fit'''
        self.bounds = bounds

    def set_constraints(self, cons):
        '''Returns constraints on fit'''
        self.contraints = cons

    def get_params(self, as_dict=True, include_errors=False):
        '''
        Returns a dictionary of parameters where the keys are the parameter
        names and values are tuples with the first entry being the parameter
        value and the second being the uncertainty on the parameter.
        '''
        if include_errors:
            params = zip(self.parnames, zip(self.params, self.param_err))
        else:
            params = zip(self.parnames, self.params)

        if as_dict:
            return OrderedDict(params)
        else:
            return params

    def get_pdf(self):
        '''
        Returns the pdf as a function with current values of parameters
        '''
        return lambda x: self.model(x, self.params)

    def update_params(self, params, param_err):
        self.params    = params
        self.param_err = param_err

    def pdf(self, X, a=None):
        '''
        Return the PDF of the model.  The same result can be achieved by accessing the model member directly.

        Parameters
        ==========
        X: data points where the PDF will be evaluated
        a: array of parameter values.  If not specified, the class member params will be used.
        '''
        if np.any(a) == None:
            return self.model(X, self.params)
        else:
            return self.model(X, a)


    def nll(self, X, a=None):
        '''
        Return the negative log likelihood of the model given some data.

        Parameters
        ==========
        X: data points where the PDF will be evaluated
        a: array of parameter values.  If not specified, the class member params will be used.
        '''
        pdf = self.pdf(X, a)
        nll = -np.sum(np.log(pdf))
        return nll

class CombinedModel():
    '''
    Multiple model class.  

    Parameters
    ==========
    models: an array of Model instances
    '''
    def __init__(self, models, parinit='None'):
        self.models = models
        self.initialize()
        self.corr = None

    def initialize(self):
        '''
        Returns a dictionary of parameters where the keys are the parameter
        names and values are tuples with the first entry being the parameter
        value and the second being the uncertainty on the parameter.
        '''
        params = OrderedDict() 
        bounds = OrderedDict() 
        for m in self.models:
            p = m.get_params()
            b = OrderedDict(zip(m.parnames, m.bounds))

            bounds.update(b)
            params.update(p)

        self.parnames  = params.keys()
        self.params    = [params[n] for n in self.parnames]
        self.param_err = np.zeros(np.size(self.params))
        self.bounds    = [bounds[n] for n in self.parnames]

    def get_params(self, as_dict=True, include_errors=False):
        '''
        Returns a dictionary of parameters where the keys are the parameter
        names and values are tuples with the first entry being the parameter
        value and the second being the uncertainty on the parameter.
        '''
        if include_errors:
            params = zip(self.parnames, zip(self.params, self.param_err))
        else:
            params = zip(self.parnames, self.params)

        if as_dict:
            return OrderedDict(params)
        else:
            return params
            
    def update_params(self, params, param_err):
        self.params     = params
        self.param_err  = param_err 

        pdict = self.get_params(include_errors=True)
        for m in self.models:
            m.params    = [pdict[n][0] for n in m.parnames]
            m.param_err = [pdict[n][1] for n in m.parnames]

    def nll(self, X, a=None):

        if np.any(a) == None:
            params = self.get_params()
        else:
            params = OrderedDict(zip(self.parnames, a)) 

        nll = 0.
        for m, x in zip(self.models, X):
            nll += m.nll(x, [params[name] for name in m.parnames])

        return nll


class NLLFitter: 
    '''
    Class for carrying out PDF estimation using unbinned negative log likelihood minimization.

    Parameters
    ==========
    model    : a Model object or and array of Model objects
    data     : the dataset or datasets we wish to carry out the modelling on
    min_algo : algorith used for minimizing the nll (uses available scipy.optimize algorithms)
    scaledict: (optional) a dictionary of functions for scaling parameters while pretty printing
    '''
    def __init__(self, model, data, min_algo='SLSQP', scaledict={}, verbose=True):
       self.model     = model
       self.data      = data
       self.min_algo  = min_algo
       self.scaledict = scaledict
       self.verbose   = verbose
       self.lmult     = (1., 1.)

    def regularization(self, a):
        nll = self.model.nll(self.data, a)
        return nll + self.lmult[0] * np.sum(np.abs(a)) + self.lmult[1] * np.sum(a**2)

    def get_corr(self, a):

        f_obj   = lambda params: self.model.nll(self.data, params)
        hcalc   = nd.Hessian(f_obj, step=0.01, method='central', full_output=True) 
        hobj    = hcalc(a)[0]
        hinv    = np.linalg.inv(hobj)

        # get uncertainties on parameters
        sig = np.sqrt(hinv.diagonal())

        # calculate correlation matrix
        mcorr = hinv/np.outer(sig, sig)

        return sig, mcorr

    def fit(self, init_params):

        self.model.update_params(init_params, init_params)
        if self.verbose:
            print 'Performing fit with initial parameters:'

            for n,p in self.model.get_params().iteritems():
                if n in self.scaledict.keys():
                    p = self.scaledict[n](p)
                print '{0}\t= {1:.3f}'.format(n, p)

        result = minimize(self.regularization, init_params,
                          method = self.min_algo, 
                          bounds = self.model.bounds,
                          #args   = (self.data, self.nll)
                          )
        if self.verbose:
            print 'Fit finished with status: {0}'.format(result.status)

        if result.status == 0:
            if self.verbose:
                print 'Calculating covariance of parameters...'

            sigma, corr = self.get_corr(result.x)
            self.model.update_params(result.x, sigma)
            self.model.corr = corr

            if self.verbose:
                self.print_results()

        return result

    def print_results(self):
        '''
        Pretty print model parameters
        '''
        print '\n'
        print 'RESULTS'
        print '-------'
        for n,p in self.model.get_params(as_dict=False, include_errors=True):
            if n in self.scaledict.keys():
                pct_err = p[1]/np.abs(p[0]) 
                pval    = self.scaledict[n](p[0])
                perr    = pval*pct_err
                print '{0}\t= {1[0]:.3f} +/- {1[1]:.3f}'.format(n, (pval, perr))
            else:
                print '{0}\t= {1[0]:.3f} +/- {1[1]:.3f}'.format(n, p)

        print '\n'
        print 'CORRELATION MATRIX'
        print '------------------'
        print self.model.corr
        print '\n'

if __name__ == '__main__':

    ### Start the timer
    start = timer()
    verbose = True

    ### get data and convert variables to be on the range [-1, 1]
    minalgo  = 'SLSQP'
    channels = ['1b1f', '1b1c']
    xlimits  = (12., 70.)
    sdict    = {'mu': lambda x: scale_data(x, invert = True),
                'sigma': lambda x: x*(xlimits[1] - xlimits[0])/2.,
               }

    bg_pdf  = lambda x, a:  0.5 + a[0]*x + 0.5*a[1]*(3*x**2 - 1)
    sig_pdf = lambda x, a: (1 - a[0])*bg_pdf(x, a[3:5]) + a[0]*norm.pdf(x, a[1], a[2])

    ### Fits for individual channels
    datas      = {}
    bg_models  = {} 
    sig_models = {} 
    for channel in channels:

        print 'Getting data for {0} channel and scaling to lie in range [-1, 1].'.format(channel)
        data, n_total  = get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
        datas[channel] = data
        print 'Analyzing {0} events...'.format(n_total)

        ### Define bg model and carry out fit ###
        bg_model = Model(bg_pdf, ['a1', 'a2'])
        bg_model.set_bounds([(0., 1.), (0., 1.)])
        bg_models[channel] = bg_model

        bg_fitter = NLLFitter(bg_model, data)
        bg_result = bg_fitter.fit([0.5, 0.05])

        ### Define bg+sig model and carry out fit ###
        sig_model = Model(sig_pdf, ['A', 'mu', 'sigma', 'a1', 'a2'])
        sig_model.set_bounds([(0., .5), 
                              (-0.8, -0.2), (0., 0.5),
                              (0., 1.), (0., 1.)])
        sig_models[channel] = sig_model

        sig_fitter = NLLFitter(sig_model, data, scaledict=sdict)
        sig_result = sig_fitter.fit((0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1]))

        ### Plots!!! ###
        print 'Making plot of fit results.'
        fit_plot(scale_data(data, invert=True), sig_pdf, sig_result.x, bg_pdf, bg_result.x, channel)


    ### Prepare data for combined fit
    ### Parameter naming is important.  If a parameter name is the same between
    ### multiple models it will be fixed between each model. 

    bg_models['1b1f'].parnames  = ['a1', 'a2']
    bg_models['1b1c'].parnames  = ['b1', 'b2']
    combined_bg_model   = CombinedModel([bg_models[ch] for ch in channels])

    sig_models['1b1f'].parnames = ['A1', 'mu', 'sigma', 'a1', 'a2']
    sig_models['1b1c'].parnames = ['A2', 'mu', 'sigma', 'b1', 'b2']
    combined_sig_model = CombinedModel([sig_models[ch] for ch in channels])

    ### Perform combined bg fit
    combination_bg_fitter = NLLFitter(combined_bg_model, [datas[ch] for ch in channels])
    bg_result = combination_bg_fitter.fit([0.5, 0.05, 0.5, 0.05])
    
    ### Perform combined signal+bg fit
    combination_sig_fitter = NLLFitter(combined_sig_model, [datas[ch] for ch in channels], scaledict=sdict)
    param_init = combined_sig_model.get_params().values()
    sig_result = combination_sig_fitter.fit(param_init)

    ### Plot results.  Overlay signal+bg fit, bg-only fit, and data
    fit_plot(scale_data(datas['1b1f'], invert=True), 
                        sig_pdf, sig_models['1b1f'].params,    
                        bg_pdf, bg_models['1b1f'].params, '1b1f_combined')
    fit_plot(scale_data(datas['1b1c'], invert=True), 
                        sig_pdf, sig_models['1b1c'].params,    
                        bg_pdf, bg_models['1b1c'].params, '1b1c_combined')
    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
