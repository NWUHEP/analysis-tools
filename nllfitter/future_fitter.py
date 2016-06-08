from __future__ import division

from timeit import default_timer as timer
from collections import OrderedDict
from functools import partial

import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
#from scipy import stats
from scipy.stats import chi2, norm 
from scipy.optimize import minimize

from nllfitter.fit_tools import get_data, fit_plot, get_corr, scale_data
from lmfit import Parameter, Parameters, Minimizer, report_fit
import lmfit

# global options
np.set_printoptions(precision=3.)

### PDF definitions ###
def bg_pdf(x, a): 
    '''
    Third order Legendre Polynomial with constant term set to 0.5.

    Parameters:
    ===========
    x: data
    a: model parameters (a1 and a2)
    '''
    return 0.5 + a[0]*x + 0.5*a[1]*(3*x**2 - 1) + 0.5*a[2]*(5*x**3 - 3*x)

def sig_pdf(x, a):
    '''
    Second order Legendre Polynomial (normalized to unity) plus a Gaussian.

    Parameters:
    ===========
    x: data
    a: model parameters (a1, a2, mu, and sigma)
    '''
    return (1 - a[0])*bg_pdf(x, a[3:6]) + a[0]*norm.pdf(x, a[1], a[2])

class Model:
    '''
    Model class that will be passed to NLLFitter to fit to a dataset.  Requires
    that the model pdf be specified and the model parameters.  The model
    parameters should be provided in a lmfit Parameters class object

    Parameters
    ==========
    model      : a function describing the model that takes argurements (params, data)
    parameters : lmfit Parameter object
    '''
    def __init__(self, model, parameters, parinit=None):
        
        self.model      = model
        self.parameters = parameters
        self.corr       = None


    def get_parameters(self, by_val=False):
        '''
        Returns parameters object
        '''
        if by_val:
            return [p.value for p in self.parameters.values()]
        else:
            return self.parameters

    def get_pdf(self):
        '''
        Returns the pdf as a function with current values of parameters
        '''
        return lambda x: self.model(x, self.parameters)

    def update_parameters(self, params, covariance=None):
        '''
        Updates the values and errors of each of the parameters.

        Parameters:
        ===========
        params: new values of parameters
        covariance: result from get_corr(), i.e., the uncertainty on the
                    parameters and their correlations in a tuple (sigma, correlation_matrix)
        '''
        self.corr = covariance[1]
        for i, (pname, pobj) in enumerate(self.parameters.iteritems()):
            self.parameters[pname] = params[pname]
            self.parameters[pname].stderr = covariance[0][i]
            #if covariance:

    def pdf(self, X, a=None):
        '''
        Return the PDF of the model.  The same result can be achieved by accessing the model member directly.

        Parameters
        ==========
        X: data points where the PDF will be evaluated
        a: array of parameter values.  If not specified, the class member params will be used.
        '''
        if np.any(a) == None:
            return self.model(X, [p.value for p in bg_params.values()])
        else:
            return self.model(X, a)


    def calc_nll(self, X):
        '''
        Return the negative log likelihood of the model given some data.

        Parameters
        ==========
        X: data points where the PDF will be evaluated
        '''
        
        pvals = [p.value for p in self.parameters.values()]
        pdf = self.pdf(X, pvals)
        nll = -np.sum(np.log(pdf))
        return nll


def calc_nll(params, data, pdf):
    '''
    Return the negative log likelihood of the model given some data.

    Parameters
    ==========
    data   : data points where the PDF will be evaluated
    params : lmfit Parameters object or an numpy array
    pdf    : probability distribution function model
    '''
     
    if isinstance(params, Parameters):
        pdf = pdf(data, [p.value for p in params.values()])
    elif isinstance(params, np.ndarray):
        pdf = pdf(data, params)

    nll = -np.sum(np.log(pdf))
    return nll


if __name__ == '__main__':

    ### Start the timer
    start = timer()
    verbose = True

    ### get data and convert variables to be on the range [-1, 1]
    xlimits  = (100., 180.)

    print 'Getting data and scaling to lie in range [-1, 1].'
    data, n_total  = get_data('data/toy_hgammagamma.txt', 'dimuon_mass', xlimits)
    print 'Analyzing {0} events...\n'.format(n_total)

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None),
                       ('a3', 0., True, None, None, None)
                       )

    bg_model  = Model(bg_pdf, bg_params)
    bg_fitter = Minimizer(calc_nll, bg_params, fcn_args=(data, bg_pdf))
    bg_result = bg_fitter.minimize('SLSQP')
    bg_params = bg_result.params
    sigma, corr = get_corr(partial(calc_nll, data=data, pdf=bg_pdf), 
                           [p.value for p in bg_params.values()]) 
    bg_model.update_parameters(bg_params, (sigma, corr))
    report_fit(bg_params, show_correl=False)
    print ''
    print '[[Correlation matrix]]\n'
    print corr, '\n'

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    sig_params.add_many(
                        ('A'     , 0.01 , True , 0.   , 1.   , None),
                        ('mu'    , -0.3 , True , -0.8 , 0.8  , None),
                        ('sigma' , 0.01 , True , 0.01 , 1.   , None),
                        ('a1'    , 0.   , True , None , None , None),
                        ('a2'    , 0.   , True , None , None , None),
                        ('a3'    , 0.   , True , None , None , None)
                       )

    sig_model  = Model(sig_pdf, sig_params)
    sig_fitter = Minimizer(calc_nll, sig_params, fcn_args=(data, sig_pdf))
    sig_result = sig_fitter.minimize('SLSQP')
    sig_params = sig_result.params
    sigma, corr = get_corr(partial(calc_nll, data=data, pdf=sig_pdf), 
                           [p.value for p in sig_params.values()]) 
    sig_model.update_parameters(sig_params, (sigma, corr))
    report_fit(sig_model.get_parameters(), show_correl=False)
    print ''
    print '[[Correlation matrix]]\n'
    print corr, '\n'

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q = 2*(bg_model.nll(data) - sig_model.nll(data))
    print '{0}: q = {1:.2f}'.format('h->gg', q)

    ### Plots!!! ###
    print 'Making plot of fit results...'
    fit_plot(scale_data(data, xmin=100, xmax=180, invert=True), xlimits,
             sig_pdf, sig_model.get_parameters(by_val=True),    
             bg_pdf, bg_model.get_parameters(by_val=True), 'hgg')
    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
