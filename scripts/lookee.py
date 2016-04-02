#!/usr/bin/env python

from fitter import *
from toy_MC import *
from lee2d import *

import pickle

def exp_phi_u(u, n1, n2):
    return chi2.sf(u,1) + np.exp(-u/2.)*(n1 + n2*np.sqrt(u))

def _eq(p, exp_phi, u):
    return exp_phi - exp_phi_u(u, p, 0)

def lee_1D(max_local_sig, u, exp_phi):

   n1, = fsolve(_eq, 1, args=(exp_phi, u))
   global_p = exp_phi_u(max_local_sig**2, n1, 0)
   print ' n1 = {0}'.format(n1)
   print ' local p_value = {0:.7f},  local significance = {1:.2f}'.format(norm.cdf(-max_local_sig), max_local_sig)
   print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(global_p, -norm.ppf(global_p))
   return n1, global_p

def validation_plot(phi_scan, qmax, N1, N2, channel):
    '''Check that the GV tails look okay'''
    u = np.linspace(0., 20., 100.)
    phi_scan = np.array(phi_scan)
    exp_phi = np.apply_along_axis(np.mean, 0, phi_scan)
    qmax = np.array(qmax)

    hval, hbins, _ = plt.hist(qmax, bins=21, cumulative=True)
    hval = hval.max() - hval
    herr = np.sqrt(hval)
    pval = hval/hval.max()
    perr = pval*(herr/hval)
    plt.close()

    plt.plot(u, exp_phi, 'k-')
    plt.plot(u, exp_phi_u(u, N1, N2), 'r--')
    plt.plot(hbins[1:], pval, 'b-')
    plt.fill_between(hbins[1:], pval-perr, pval+perr, color='b', alpha=0.25, interpolate=True)

    plt.yscale('log')
    plt.ylim(1e-4, 10)
    plt.ylabel('P[max(q) > u]')
    plt.xlabel('u')
    plt.savefig('figures/GV_validate_{0}.png'.format(channel))
    plt.close()


if __name__ == '__main__':
    start = timer()

    ### Config 
    minalgo     = 'SLSQP'
    channel     = '1b1f'
    xlimits     = (12., 70.)
    nscan       = (30, 30)
    nsims       = 100
    make_plots  = True

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

    scan_vals = np.array([(n1, n2) for n1 in np.linspace(-0.9, 0.9, nscan[0]) for n2 in np.linspace(0.02, 0.08, nscan[1])])
    #scan_vals = np.array([(n1, params['width']) for n1 in np.linspace(-0.9, 0.9, nscan[0])]) 

    ### Get data and scale
    data, n_total = get_data('data/ntuple_{0}.csv'.format(channel), 'dimuon_mass', xlimits)

    #######################
    ### Calculate LEE2D ###
    #######################

    ### scan over test data
    bnds    = [(0., 1.0), # A
               2*(params['mu'], ), 2*(params['width'], ), # mean, sigma
               (0., 1.), (0., 1.) # a1, a2 
               ]
    nll_bg = bg_objective(bg_params.values(), data)

    print 'Scanning ll ratio over data...'
    qscan = []
    qmax  = np.abs(2*(nll_bg - bg_sig_objective(params.values(), data)))
    for i, scan in enumerate(scan_vals):
        # Remove edge effects
        if scan[0] - 3*scan[1] < -1 or scan[0] + 3*scan[1] > 1: 
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
        qtest = 2*(nll_bg - bg_sig_objective(result.x, data))
        qscan.append(qtest)
        if qtest > qmax: qmax = qtest

    print 'Scan over data done: qmax = {0:.1f}'.format(qmax)
    x       = scale_data(scan_vals[:,0].reshape(nscan), invert=True)
    y       = 0.5*(xlimits[1] - xlimits[0])*scan_vals[:,1].reshape(nscan)
    qscan   = np.array(qscan).reshape(nscan)

    if make_plots:
        fig, ax = plt.subplots()
        cmap = ax.pcolormesh(x, y, qscan, cmap='viridis', vmin=0.)#, vmax=25.)
        fig.colorbar(cmap)
        ax.set_xlabel('M_{mumu} [GeV]')
        ax.set_ylabel('width [GeV]')
        fig.savefig('figures/qscan_data_{0}.png'.format(channel))
        plt.close()

    ### Make some pseudo-data ###
    print 'Scanning ll ratio over {0} pseudo-datasets...'.format(nsims)
    bg_pdf  = lambda x: 0.5 + bg_params['a1']*x + 0.5*bg_params['a2']*(3*x**2 -1)
    sims = mc_generator(bg_pdf, n_total, nsims)

    if make_plots:
        fig1, axes1 = plt.subplots(3, 3)
        fig2, axes2 = plt.subplots(3, 3)
        fig3, axes3 = plt.subplots(3, 3)

    phiscan = []
    qmax_mc = []
    phi1 = []
    phi2 = []
    u1, u2 = 3., 6.
    for i, sim in enumerate(sims):
        if not i%10: 
            print 'Carrying out scan {0}...'.format(i+1)

        bg_result = minimize(regularization, 
                             (bg_params['a1'], bg_params['a2']),
                             method = 'SLSQP',
                             args   = (sim, bg_objective, 1., 1.),
                             bounds = bnds[-2:]
                             )
        nll_bg = bg_objective(bg_result.x, sim)

        qscan       = []
        params_best = []
        qmax_mc.append(0)
        for j, scan in enumerate(scan_vals):
            if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
                qscan.append(0.)
                continue

            bnds[1] = (scan[0], scan[0])
            bnds[2] = (scan[1], scan[1])
            result = minimize(regularization, 
                                   (1., scan[0], scan[1], bg_result.x[0], bg_result.x[1]),
                                   method = 'SLSQP',
                                   args   = (sim, bg_sig_objective),
                                   bounds = bnds
                                   )

            qtest = 2*(nll_bg - bg_sig_objective(result.x, sim))
            qscan.append(qtest)
            if qtest > qmax_mc[-1]: 
                params_best = result.x
                qmax_mc[-1] = qtest

        #if make_plots and i < 100:
        #    sim = scale_data(sim, invert=True)
        #    fit_plot(sim, combined_model, params_best, 
        #             legendre_polynomial, bg_result.x, 
        #             '{0}_{1}'.format(channel,i+1), path='figures/scan_fits')
        #print bg_result.x, params_best, qmax_mc[-1]

        ### Doing calculations
        qscan = np.array(qscan).reshape(nscan)
        phiscan.append([calculate_euler_characteristic((qscan > u) + 0.) for u in np.linspace(0., 20., 100.)])
        phi1.append(calculate_euler_characteristic((qscan > u1) + 0.))
        phi2.append(calculate_euler_characteristic((qscan > u2) + 0.))
        
        if make_plots and i < 9: 
            cmap = axes1[i/3][i%3].pcolormesh(x, y, qscan, cmap='viridis', vmin=0.)
            axes2[i/3][i%3].imshow((qscan > u1) + 0., cmap='Greys', interpolation='none', origin='lower')
            axes3[i/3][i%3].imshow((qscan > u2) + 0., cmap='Greys', interpolation='none', origin='lower')

    if make_plots:
        #fig1.subplots_adjust(right=0.8)
        #fig1.colorbar(cmap)
        fig1.savefig('figures/qscan_toys_{0}.png'.format(channel))
        fig2.savefig('figures/qscan_u1_{0}.png'.format(channel))
        fig3.savefig('figures/qscan_u2_{0}.png'.format(channel))
        plt.close()


    ### Calculate LEE correction ###
    exp_phi1, exp_phi2 = np.mean(phi1), np.mean(phi2)
    print 'E[phi_1] = {0}'.format(exp_phi1)
    print 'E[phi_2] = {0}'.format(exp_phi2)
    N1, N2, p_global = do_LEE_correction(np.sqrt(qmax), u1, u2, exp_phi1, exp_phi2)
    #N1, global_p = lee_1D(np.sqrt(qmax), u1, exp_phi1)

    validation_plot(phiscan, qmax_mc, N1, N2, channel)

    # Save scan data
    outfile = open('data/lee_scan_{0}_{1}.pkl'.format(channel, nsims), 'w')
    pickle.dump(qmax_mc, outfile)
    pickle.dump(phiscan, outfile)
    outfile.close()

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))

