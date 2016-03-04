#!/usr/bin/env python

from fitter import *
from toy_MC import *
from lee2d import *

if __name__ == '__main__':
    start = timer()

    ### Config 
    channel = '1b1f'
    xlimits = (12., 70.)
    nscan   = 100
    nsims   = 20
    scan_vals = [(n1, n2) for n1 in np.linspace(-0.9, 0.9, nscan) for n2 in np.linspace(0.02, 0.08, nscan)]

    if channel == '1b1f':
        params  = {'A':(0.88, 0.04), 'mu':(-0.42, 0.02), 'width':(0.05, 0.015), 'a1':(0.32, 0.076), 'a2':(0.133, 0.1)} 
    elif channel == '1b1c':
        params  = {'A':(0.96, 0.02), 'mu':(-0.45, 0.03), 'width':(0.05, 0.015), 'a1':(0.21, 0.04), 'a2':(0.08, 0.05)} 

    ### Get data and scale
    ntuple  = pd.read_csv('data/ntuple_{0}.csv'.format(channel))
    data    = ntuple['dimuon_mass'].values
    data    = np.apply_along_axis(scale_data, 0, data, xlow=xlimits[0], xhigh=xlimits[1])
    n_total = data.size

    #######################
    ### Calculate LEE2D ###
    #######################

    ### scan over test data
    bnds = [(0., 1.05), # A
            2*(params['mu'][0], ), 2*(params['width'][0], ), # mean, sigma
            2*(params['a1'][0], ), 2*(params['a2'][0], )] # a1, a2
    llbg = bg_objective([params['a1'][0], params['a2'][0]], data)

    print 'Scanning ll ratio over data...'
    qscan = np.zeros((nscan, nscan))
    qmax  = 0.
    for i, scan in enumerate(scan_vals):
        # Remove edge effects
        if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
            qscan[i/nscan][i%nscan] = 0.
            continue

        bnds[1] = (scan[0], scan[0])
        bnds[2] = (scan[1], scan[1])
        scan_result = minimize(regularization, 
                               (1., scan[0], scan[1], params['a1'][0], params['a2'][0]),
                               method = 'SLSQP',
                               #jac    = True,
                               args   = (data, bg_sig_objective, 1., 1.),
                               bounds = bnds
                               )
        qtest = 2*(llbg - bg_sig_objective(scan_result.x, data))
        qscan[i/nscan][i%nscan] = qtest

        if qtest > qmax: qmax = qtest

    fig, ax = plt.subplots()
    im = ax.imshow(qscan, cmap='viridis', interpolation='none', origin='lower', vmin=0., vmax=20.)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    #plt.savefig('figures/qscan_data_{0}.png'.format(channel))
    plt.close()


    ### Make some pseudo-data ###
    print 'Scanning ll ratio over {0} pseudo-datasets...'.format(nsims)
    bg_pdf  = lambda x: 0.5 + params['a1'][0]*x + 0.5*params['a2'][0]*(3*x**2 -1)
    sims= mc_generator(bg_pdf, n_total, nsims)
    fig1, axes1 = plt.subplots(3, 3)
    fig2, axes2 = plt.subplots(3, 3)
    fig3, axes3 = plt.subplots(3, 3)
    phi1 = []
    phi2 = []
    u1, u2 = 1., 2.
    for i, sim in enumerate(sims):
        llbg = bg_objective([params['a1'][0], params['a2'][0]], sim)
        qscan = np.zeros((nscan, nscan))
        for j, scan in enumerate(scan_vals):
            if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
                qscan[i/nscan][i%nscan] = 0.
                continue
            bnds[1] = (scan[0], scan[0])
            bnds[2] = (scan[1], scan[1])
            scan_result = minimize(regularization, 
                                   (1., scan[0], scan[1], params['a1'][0], params['a2'][0]),
                                   method = 'SLSQP',
                                   #jac    = True,
                                   args   = (sim, bg_sig_objective, 1., 1.),
                                   bounds = bnds
                                   )
            qscan[j/nscan][j%nscan] = 2*(llbg - bg_sig_objective(scan_result.x, sim))

        ### Doing calculations
        phi1.append(calculate_euler_characteristic((qscan > u1) + 0.))
        phi2.append(calculate_euler_characteristic((qscan > u2) + 0.))
        
        a = (qscan > 1.) + 0.
        if i < 9: 
            im = axes1[i/3][i%3].imshow(qscan, cmap='viridis', interpolation='none', origin='lower', vmin=0., vmax=10.)
            axes2[i/3][i%3].imshow((qscan > u1) + 0., cmap='Greys', interpolation='none', origin='lower')
            axes3[i/3][i%3].imshow((qscan > u2) + 0., cmap='Greys', interpolation='none', origin='lower')

    fig1.subplots_adjust(right=0.8)
    cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
    fig1.colorbar(im, cax=cbar_ax)
    fig1.savefig('figures/qscan_toys_{0}.png'.format(channel))
    fig2.savefig('figures/qscan_u1_{0}.png'.format(channel))
    fig3.savefig('figures/qscan_u2_{0}.png'.format(channel))
    plt.close()

    ### Calculate LEE correction ###
    exp_phi1, exp_phi2 = np.mean(phi1), np.mean(phi2)
    print 'E[phi_1] = {0}'.format(exp_phi1)
    print 'E[phi_2] = {0}'.format(exp_phi2)
    do_LEE_correction(np.abs(norm.ppf(1 - chi2.cdf(qmax, 1))), u1, u2, exp_phi1, exp_phi2)

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))

