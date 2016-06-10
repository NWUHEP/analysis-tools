from __future__ import division
from timeit import default_timer as timer

import numpy as np
from lmfit import Parameter, Parameters

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
    def __init__(self, pdf, parameters):
        
        self._pdf       = pdf
        self.parameters = parameters
        self.corr       = None

    def get_parameters(self, by_value=False):
        '''
        Returns parameters either as an lmfit Parameters object or a list of parameter values

        Parameters:
        ===========
        by_value: if True returns a list of parameter values, otherwise the
                  function will return the lmfit Parameters object
        '''
        if by_value:
            return [p.value for p in self.parameters.values()]
        else:
            return self.parameters

    def get_bounds(self):
        '''
        Return list of tuples with the bounds for each parameter.  
        '''
        return [(p.min, p.max) for n,p in self.parameters.iteritems()]

    def get_constranints(self):
        '''
        Return list of tuples with the bounds for each parameter.  
        '''
        return [p.expr for n,p in self.parameters.iteritems()]

    def pdf(self, data, params=None):
        '''
        Returns the pdf as a function with current values of parameters
        '''
        if isinstance(params, Parameters):
            return self._pdf(data, [params[n].value for n in self.parameters.keys()])
        if isinstance(params, np.ndarray):
            return self._pdf(data, params)
        else:
            return self._pdf(data, self.get_parameters(by_value=True))

    def update_parameters(self, params, covariance=None):
        '''
        Updates the parameters values and errors of each of the parameters if
        specified.  Parameters can be specified either as an lmfit Parameters
        object or an array.

        Parameters:
        ===========
        params: new values of parameters
        covariance: result from _get_corr(), i.e., the uncertainty on the
                    parameters and their correlations in a tuple (sigma, correlation_matrix)
        '''

        for i, (pname, pobj) in enumerate(self.parameters.iteritems()):
            if isinstance(params, np.ndarray):
                self.parameters[pname].value = params[i]
            else:
                self.parameters[pname] = params[pname]

            if covariance:
                self.parameters[pname].stderr = covariance[0][i]

        if covariance:
            self.corr = covariance[1]

    def calc_nll(self, X, params=None):
        '''
        Return the negative log likelihood of the model given some data.

        Parameters
        ==========
        a: model parameters specified as a numpy array or lmfit Parameters
           object.  If not specified, the current model parameters will be used
        X: data points where the PDF will be evaluated
        '''
        
        if isinstance(params, Parameters):
            params = [params[n].value for n in self.parameters.keys()]
        elif np.any(params) == None:
            params = [p.value for p in self.parameters.values()]

        pdf = self._pdf(X, params)
        nll = -np.sum(np.log(pdf))
        return nll

