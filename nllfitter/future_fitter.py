from __future__ import division

from timeit import default_timer as timer
from collections import OrderedDict

import pandas as pd
import numpy as np
import numdifftools as nd
#from scipy import stats
from scipy.stats import chi2, norm 
from scipy.optimize import minimize

from nllfitter.fit_tools import get_data, scale_data, fit_plot, bg_pdf, sig_pdf
from lmfit import Parameters

# global options
np.set_printoptions(precision=3.)

def kolmogorov_smirinov(data, model_pdf, xlim=(-1, 1), npoints=10000):
    
    xvals = np.linspace(xlim[0], xlim[1], npoints)

    #Calculate CDFs 
    data_cdf = np.array([data[data < x].size for x in xvals]).astype(float)
    data_cdf = data_cdf/data.size

    model_pdf = model_pdf(xvals)
    model_cdf = np.cumsum(model_pdf)*(xlim[1] - xlim[0])/npoints

    return np.abs(model_cdf - data_cdf)

def calculate_likelihood_ratio(bg_model, s_model, data):
    '''
    '''
    bg_nll = bg_model.nll(data)
    s_nll = s_model.nll(data)

    return 2*(bg_nll - s_nll)


class Model:
    '''
    Model class that will be passed to NLLFitter to fit to a dataset.  Requires
    that the model pdf be specified and the model parameters.  The model
    parameters should be provided in a lmfit Parameters class object

    Parameters
    ==========
    model  : a function describing the model that takes argurements (params, data)
    parameters: lmfit Parameter object
    '''
    def __init__(self, model, params, parinit=None):
        
        self.model      = model
        self.parameters = params
        self.corr       = None

    def set_bounds(self, param_name, bounds):
        '''
        Set bounds for given parameter.

        Parameters:
        ===========
        param_name: name of parameter to be updated
        bounds: tuple with first entry being the min and second being the max
        '''
        self.parameters[param_name].set(min=bounds[0])
        self.parameters[param_name].set(max=bounds[1])

    def set_constraints(self, param_name, cons):
        '''
        Set constraint expression for given parameter.

        Parameters:
        ===========
        param_name: name of parameter to be updated
        cons: string expression specifying constraint on give parameter
        '''
        self.parameters[param_name].set(expr=cons)

    def get_params(self):
        '''
        Returns parameters object
        '''
        return self.parameters

    def get_pdf(self):
        '''
        Returns the pdf as a function with current values of parameters
        '''
        return lambda x: self.model(x, self.parameters)

    def update_params(self, params, param_err):
        '''
        Updates the values and errors of each of the parameters.
        '''
        for pname in self.parameters.keys():
            self.parameters[pname].set(value=params[pname])

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

    def set_data(self, data):
        self.data = data    

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

    def fit(self, init_params, calculate_corr=True):

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
            if calculate_corr:
                if self.verbose:
                    print 'Calculating covariance of parameters...'
                sigma, corr = self.get_corr(result.x)
            else:
                sigma, corr = result.x, 0.

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
    xlimits  = (100., 180.)
    sdict    = {'mu': lambda x: scale_data(x, invert = True),
                'sigma': lambda x: x*(xlimits[1] - xlimits[0])/2.
               }

    print 'Getting data and scaling to lie in range [-1, 1].'.format(channel)
    data, n_total  = get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    print 'Analyzing {0} events...'.format(n_total)

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
#                      (Name , Value , Vary , Min  , Max  , Expr)
    bg_params.add_many(('a1' , 0.    , True , None , None , None),
                       ('a2' , 0.    , True , None , None , None),
                       ('a3' , 0.    , True , None , None , None))
    bg_model = Model(bg_pdf, bg_params)
    bg_model.set_bounds([(None, None), (None, None), (None, None)])
    bg_fitter = NLLFitter(bg_model, data)
    bg_result = bg_fitter.fit([0.5, 0.05, 0.05])

    ### Define bg+sig model and carry out fit ###
    sig_model = Model(sig_pdf, ['A', 'mu', 'sigma', 'a1', 'a2'])
    sig_model.set_bounds([(0., .5), 
                          (-0.9, -0.2), (0., 1.),
                          (None, None), (None, None), (None, None)
                         ])
    sig_fitter = NLLFitter(sig_model, data, scaledict=sdict)
    sig_result = sig_fitter.fit((0.01, -0.3, 0.1, bg_result.x[0], bg_result.x[1], bg_result.x[2]))

    q = calculate_likelihood_ratio(bg_model, sig_model, data)
    print '{0}: q = {1:.2f}'.format(channel, q)

    ### Plots!!! ###
    print 'Making plot of fit results.'
    fit_plot(scale_data(data, invert=True), xlimits, sig_pdf, sig_result.x, bg_pdf, bg_result.x, channel)

    ### Plot results.  Overlay signal+bg fit, bg-only fit, and data
    fit_plot(scale_data(data, invert=True), xlimits,
                        sig_pdf, sig_model.params,    
                        bg_pdf, bg_model.params, '{0}'.format(channel))
    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
