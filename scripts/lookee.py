#!/usr/bin/env python

from fitter import *
from toyMC import *

if __name__ == '__main__':

    ### Carry out scan over signal fit parameters ###
    params = {'A':(0.88, 0.04), 'mu':(-0.42, 0.02), 'width':(0.054, 0.015), 'a1':(0.32, 0.076), 'a2':(0.133, 0.1)} 

    ### Generate scan values varying mu and width within th fit range
    xran = (12., 70.)
    
    ### Calculate LEE2D ###
    print 'Scanning over mu and sigma values of signal...'
    nscan = 100
    scan_vals = [(n1, n2) for n1 in np.linspace(-0.95, 0.95, nscan) for n2 in np.linspace(0.05, 0.5, nscan)]
    bnds = [(0., 1.04), # A
            2*(params['mu'], ), 2*(params['width'], ), # mean, sigma
            2*(params['a1'], ), 2*(params['a2'], )] # a1, a2
    llbg = bg_objective([0.208, 0.017], data_scaled)
    qscan = np.zeros((100, 100))
    #toys = 
    for i, scan in enumerate(scan_vals):
        bnds[1] = (scan[0], scan[0])
        bnds[2] = (scan[1], scan[1])
        scan_result = minimize(regularization, 
                               (1., scan[0], scan[1], result.x[3], result.x[4]), 
                               method = 'SLSQP',
                               #jac    = True,
                               args   = (data_scaled, bg_sig_objective, 1., 1.),
                               bounds = bnds
                               )
        qtest = np.sqrt(2*np.abs(bg_sig_objective(scan_result.x, data_scaled) - llbg))
        #qscan[i%nscan][]


    
