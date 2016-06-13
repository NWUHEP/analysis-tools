''' 
NLLFitter is a tool for carrying out negative log-likelihood minimization
to estimate PDF parameters.  It is a wrapper for scipy.optimize.minimize and by
default use the SLSQP algorithm.  This framework will eventually be brough
closer to the lmfit framework.  For now, it only relies on the parameter
classes.
'''

from .model import Model, CombinedModel
from .nllfitter import NLLFitter, ScanParameters
from lmfit import Parameter, Parameters

