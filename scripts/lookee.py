#!/usr/bin/env python

from fitter import *
from toy_MC import *
from lee2d import *

def func(u, n1, n2):
    return 1-chi2.cdf(u,1) + np.exp(-u/2.)*(n1 + np.sqrt(u)*n2)

if __name__ == '__main__':
    start = timer()

    ### Config 
    channel = '1b1f'
    xlimits = (12., 70.)
    nscan   = (100, 1)
    nsims   = 200
    #scan_vals = np.array([(n1, n2) for n1 in np.linspace(-0.9, 0.9, nscan[0]) for n2 in np.linspace(0.02, 0.08, nscan[1])])

    if channel == '1b1f':
        params          = {'A':0.88, 'mu':-0.422, 'width':0.054, 'a1':0.319, 'a2':0.133} 
        params_err      = {'A':0.037, 'mu':0.02, 'width':0.015, 'a1':0.076, 'a2':0.1} 
        bg_params       = {'a1':0.208, 'a2':0.017}
        bg_params_err   = {'a1':0.065, 'a2':0.089}
    elif channel == '1b1c':
        params          = {'A':0.962, 'mu':-0.447, 'width':0.049, 'a1':0.208, 'a2':0.083} 
        params_err      = {'A':0.017, 'mu':-.026, 'width':0.02, 'a1':0.042, 'a2':0.051} 
        bg_params       = {'a1':0.173, 'a2':0.055}
        bg_params_err   = {'a1':0.039, 'a2':0.049}

    scan_vals = np.array([(n1, params['width']) for n1 in np.linspace(-0.9, 0.9, nscan[0])]) 

    ### Get data and scale
    ntuple  = pd.read_csv('data/ntuple_{0}.csv'.format(channel))
    data    = ntuple['dimuon_mass'].values
    data    = np.apply_along_axis(scale_data, 0, data, xlow=xlimits[0], xhigh=xlimits[1])
    n_total = data.size

    #######################
    ### Calculate LEE2D ###
    #######################

    ### scan over test data
    bnds    = [(0., 1.05), # A
               2*(params['mu'], ), 2*(params['width'], ), # mean, sigma
               (0.9*params['a1'], 1.1*params['a1']), # a1 
               (0.9*params['a2'], 1.1*params['a2'])  # a2 
               #(params['a1']-params_err['a1'], params['a1']+params_err['a1']), # a1 +- sig_a1
               #(params['a2']-params_err['a2'], params['a2']+params_err['a2'])  # a2 +- sig_a2
               ]
    bg_bnds = [(0.9*bg_params['a1'], 1.1*bg_params['a1']), # a1 
               (0.9*bg_params['a2'], 1.1*bg_params['a2'])  # a2 
            
               #(bg_params['a1']-bg_params_err['a1'], bg_params['a1']+bg_params_err['a1']), # a1 +- sig_a1
               #(bg_params['a2']-bg_params_err['a2'], bg_params['a2']+bg_params_err['a2'])  # a2 +- sig_a2
               ]
    nll_bg    = bg_objective(bg_params.values(), data)

    print 'Scanning ll ratio over data...'
    qscan = []
    qmax  = np.abs(2*(nll_bg - bg_sig_objective(params.values(), data)))
    for i, scan in enumerate(scan_vals):
        # Remove edge effects
        if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
            qscan.append(0.)
            continue

        bnds[1] = (scan[0], scan[0])
        bnds[2] = (scan[1], scan[1])
        result = minimize(regularization, 
                          (1., scan[0], scan[1], params['a1'], params['a2']),
                          method = 'SLSQP',
                          args   = (data, bg_sig_objective, 1., 1.),
                          bounds = bnds
                          )
        qtest = 2*np.abs(nll_bg - bg_sig_objective(result.x, data))
        qscan.append(qtest)
        if qtest > qmax: qmax = qtest

    print 'Scan over data done: qmax = {0}'.format(qmax)
    x       = scale_data(scan_vals[:,0].reshape(nscan), invert=True)
    y       = 0.5*(xlimits[1] - xlimits[0])*scan_vals[:,1].reshape(nscan)
    qscan   = np.array(qscan).reshape(nscan)
    phiscan_data = [calculate_euler_characteristic((qscan > u) + 0.) for u in np.linspace(0., 20., 100.)]

    fig, ax = plt.subplots()
    cmap = ax.pcolormesh(x, y, qscan, cmap='viridis', vmin=0.)#, vmax=25.)
    fig.colorbar(cmap)
    ax.set_xlabel('M_{mumu} [GeV]')
    ax.set_ylabel('width [GeV]')
    plt.savefig('figures/qscan_data_{0}.png'.format(channel))
    plt.close()

    ### Make some pseudo-data ###
    print 'Scanning ll ratio over {0} pseudo-datasets...'.format(nsims)
    bg_pdf  = lambda x: 0.5 + bg_params['a1']*x + 0.5*bg_params['a2']*(3*x**2 -1)
    sims = mc_generator(bg_pdf, n_total, nsims)

    fig1, axes1 = plt.subplots(3, 3)
    fig2, axes2 = plt.subplots(3, 3)
    fig3, axes3 = plt.subplots(3, 3)

    phiscan = []
    qmax_mc = []
    phi1 = []
    phi2 = []
    u1, u2 = 1., 2.
    for i, sim in enumerate(sims):
        bg_result = minimize(regularization, 
                             (bg_params['a1'], bg_params['a2']),
                             method = 'SLSQP',
                             args   = (sim, bg_objective, 1., 1.),
                             bounds = bg_bnds
                             )
        nll_bg = bg_objective(bg_result.x, sim)
        qmax_mc.append(0)
        qscan = []
        print 'Scan {0} started...'.format(i+1)
        for j, scan in enumerate(scan_vals):
            if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
                qscan.append(0.)
                continue

            bnds[1] = 2*(scan[0], )
            bnds[2] = (scan[1], scan[1])
            result = minimize(regularization, 
                                   (1., scan[0], scan[1], params['a1'], params['a2']),
                                   method = 'SLSQP',
                                   args   = (sim, bg_sig_objective, 1., 1.),
                                   bounds = bnds
                                   )

            qtest = 2*np.abs(nll_bg - bg_sig_objective(result.x, sim))
            qscan.append(qtest)
            if qtest > qmax_mc[-1]: qmax_mc[-1] = qtest

        ### Doing calculations
        qscan = np.array(qscan).reshape(nscan)
        phiscan.append([calculate_euler_characteristic((qscan > u) + 0.) for u in np.linspace(0., 20., 100.)])
        phi1.append(calculate_euler_characteristic((qscan > u1) + 0.))
        phi2.append(calculate_euler_characteristic((qscan > u2) + 0.))
        
        if i < 9: 
            cmap = axes1[i/3][i%3].pcolormesh(x, y, qscan, cmap='viridis', vmin=0.)
            axes2[i/3][i%3].imshow((qscan > u1) + 0., cmap='Greys', interpolation='none', origin='lower')
            axes3[i/3][i%3].imshow((qscan > u2) + 0., cmap='Greys', interpolation='none', origin='lower')

    phiscan = np.array(phiscan)

    '''
    #fig1.subplots_adjust(right=0.8)
    #fig1.colorbar(cmap)
    fig1.savefig('figures/qscan_toys_{0}.png'.format(channel))
    fig2.savefig('figures/qscan_u1_{0}.png'.format(channel))
    fig3.savefig('figures/qscan_u2_{0}.png'.format(channel))
    plt.close()
    '''

    ### Calculate LEE correction ###
    exp_phi1, exp_phi2 = np.mean(phi1), np.mean(phi2)
    print 'E[phi_1] = {0}'.format(exp_phi1)
    print 'E[phi_2] = {0}'.format(exp_phi2)
    do_LEE_correction(np.sqrt(qmax), u1, u2, exp_phi1, exp_phi2)
    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))

