#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

def scale_data(x, xmin=12., xmax=70., invert=False):
    if not invert:
        return 2*(x - xmin)/(xmax - xmin) - 1
    else:
        return 0.5*(x + 1)*(xmax - xmin) + xmin

if __name__ == '__main__':

    # get data, scale, and define pdfs (with hard coded values)
    x = pd.read_csv('data/null_spectrum_1.txt', header=None).values
    x = scale_data(x)
    b = lambda x: 0.5 + 0.3603*x + 0.5*0.0152*(3*x**2 - 1)
    s = lambda x: norm.pdf(x, -0.4241, 0.0479)

    ## calculate coefficients
    c_0 = np.sum(np.log(b(x)))
    c_1 = np.sum((s(x) - b(x))/b(x))
    c_2 = -np.sum((s(x) - b(x))**2/b(x)**2)
    c_3 = 2*np.sum((s(x) - b(x))**3/b(x)**3)

    # define likelihood vs A
    A = np.linspace(-0.05, 0.1)
    nll_true = lambda A:np.sum(np.log((1 - A)*b(x) + A*s(x)), axis=0)
    nll_appr1 = lambda A: c_0 + c_1*A + 0.5*c_2*A**2 
    nll_appr2 = lambda A: c_0 + c_1*A + 0.5*c_2*A**2 + (1./6.)*c_3*A**3

    plt.plot(A, nll_true(A), '-b')
    plt.plot(A, nll_appr1(A), ':m')
    plt.plot(A, nll_appr2(A), '--r')
