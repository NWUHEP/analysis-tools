import pickle
from timeit import default_timer as timer
from multiprocessing import Process

import pandas as pd
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import numdifftools as nd

#from scipy import stats
from scipy import integrate
from scipy.stats import chi2, norm 
from scipy.optimize import minimize

from fitter import get_data

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

        self.bounds       = None
        self.constaints   = None
    
    def set_bounds(self, bounds):
        '''Returns bounds for fit'''
        self.bounds = bounds

    def set_constraints(self, cons):
        '''Returns constraints on fit'''
        self.contraints = cons

    def get_params(self):
        '''
        Returns a dictionary of parameters where the keys are the parameter
        names and values are tuples with the first entry being the parameter
        value and the second being the uncertainty on the parameter.
        '''
        return dict(zip(self.parnames, zip(self.params, self.param_err)))

    def pdf(self, X, a=None):
        '''
        Return the PDF of the model.  The same result can be achieved by accessing the model member directly.

        Parameters
        ==========
        X: data points where the PDF will be evaluated
        a: array of parameter values.  If not specified, the class member params will be used.
        '''
        if np.any(a) == None:
            return self.model(X)
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
        if np.any(a) == None:
            pdf = self.model(X)
        else:
            pdf = self.pdf(X, a)
        nll = -np.sum(np.log(pdf))
        return nll


class NLLFitter:
    '''
    Class for carrying out PDF estimation using unbinned log likelihood

    Parameters
    ==========
    model    : a Model object 
    data     : the dataset we wish to carry out the modelling on
    min_algo : algorith used for minimizing the nll (uses available scipy.optimize algorithms)
    '''
    def __init__(self, model, data, min_algo='SLSQP'):
       self.model    = model
       self.data     = data
       self.min_algo = min_algo
       self.lmult  = (1., 1.)

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
        print 'Performing fit...'
        print 'Using initial parameters: '

        self.model.params = init_params
        for n,p in self.model.get_params().iteritems():
            print '{0}\t= {1}'.format(n, p[0])

        result = minimize(self.regularization, 
                          init_params,
                          method = self.min_algo, 
                          bounds = self.model.bounds,
                          #args   = (self.data, self.nll)
                          )
        print 'Fit finished with status: {0}'.format(result.status)
        print 'Calculating covariance of parameters...'
        sigma, corr          = self.get_corr(result.x)
        self.model.params    = result.x
        self.model.param_err = sigma
        self.model.corr      = corr

        self.print_results()

        return result

    def print_results(self):
        print '\n'
        print 'RESULTS'
        print '-------'
        for n,p in self.model.get_params().iteritems():
            print '{0}\t= {1[0]:.3f} +/- {1[1]:.3f}'.format(n, p)

        print '\n'
        print 'CORRELATION MATRIX'
        print '------------------'
        print self.model.corr
        print '\n'

if __name__ == '__main__':

    # Start the timer
    start = timer()
    verbose = True

    # get data and convert variables to be on the range [-1, 1]
    print 'Getting data and scaling to lie in range [-1, 1].'
    minalgo = 'SLSQP'
    channel = '1b1f'
    xlimits = (12., 70.)
    data, n_total = get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    print 'Analyzing {0} events...'.format(n_total)

    ### Define bg fit model ###
    bg_pdf   = lambda x, a:  0.5 + a[0]*x + 0.5*a[1]*(3*x**2 - 1)
    bg_model = Model(bg_pdf, ['a1', 'a2'])
    bg_model.set_bounds([(0., 1.), (0., 1.)])

    bg_fitter = NLLFitter(bg_model, data)
    result = bg_fitter.fit([0.5, 0.05])

    #sig_model   = lambda x, a: stats.norm.pdf(x, a[0], a[1])
