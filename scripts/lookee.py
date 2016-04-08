#!/usr/bin/env python

from fitter import *
from toy_MC import *
from lee2d import *

from scipy.special import gamma
from scipy.misc import comb, factorial
import pickle

def rho_g(j, k, u):
    '''
    From therem 15.10.1 from Random Fields and Geometry (Adler)
    j: number of nuisance parameters (search dimensions)
    k: d.o.f. of chi2 random field
    u: threshold for excursions in the field
    '''
    coeff_num       = u**((k - j)/2.) * np.exp(-u/2.) 
    coeff_den       = (2.*np.pi)**(j/2.) * gamma(k/2.) * 2**((k-2.)/2.)
    indicate        = lambda m,l: (k >= j - m - 2.*l) + 0.
    sum_fraction    = lambda m,l: ((-1.)**(j-1.+m+l)*gamma(j-1)) / (gamma(m)*gamma(l)*2.**l)
    m_sum           = lambda l: np.sum([indicate(m,l)*comb(k-l, j-1.-m-2.*l)*sum_fraction(m,l) for m in np.arange(0, 1 + int(j-1.-2.*l))])   
    l_sum           = np.sum([m_sum(l) for l in np.arange(0., 1 + np.floor((j-1)/2))]) 

    return (coeff_num/coeff_den)*l_sum

def exp_phi_u(u, n_j, k=1):
    '''
    1 or 2 dimensional expressions for chi2 random field EC expectation
    
    Parameters
    ----------
    u: array of scan thresholds
    n_j: array of coefficients
    k: nDOF of chi2 field
    '''
    return chi2.sf(u,k) + np.sum([n*rho_g(j, k, u) for n,j in enumerate(n_j)])

def lee_objective(a, Y, dY, X):
    return (Y - exp_phi_u(X, a[1], a[2], j=1, k=a[0]))**2/dY 

def lee_nD(max_local_sig, u, phiscan, j=1, k=1):
    '''
    Carries GV style look elsewhere corrections with a twist.  Allows for an
    arbitrary number of search dimensions/nuisance parameters and allows the
    number of degrees of freedom of the chi2 random field to be a parameter of
    the model.  Cool shit.

    Parameters
    ----------
    max_local_sig: observed local significance (assumes sqrt(-2*nllr))
    u: array of scan thresholds
    phiscan: scan of EC for values in u
    j = numbers of search dimensions to calculate
    k = assumed numbers of degrees of freedom of chi2 field
    '''
    exp_phi = np.mean(phiscan, axis=0)
    var_phi = np.var(phiscan, axis=0)
    
    result = minimize(lee_objective, 
                      [k] + j*[1.],
                      method = 'SLSQP',
                      args   = (exp_phi, var_phi, u),
                      #bounds = bnds
                      )
    k = result.x[0]
    n = result.x[1:]
    p_global = exp_phi_u(max_local_sig**2, n, k)

    return n, p_global

def validation_plots(u, phi_scan, qmax, N1, N2, s, channel):
    '''Check that the GV tails look okay'''
    phi_scan    = np.array(phi_scan)
    exp_phi     = np.mean(phi_scan, axis=0)
    var_phi     = np.var(phi_scan, axis=0)
    qmax        = np.array(qmax)

    hval, hbins, _ = plt.hist(qmax, bins=21, cumulative=True)
    hval = hval.max() - hval
    herr = np.sqrt(hval)
    pval = hval/hval.max()
    perr = pval*(herr/hval)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(u, exp_phi, 'k-', linewidth=2.)
    ax.plot(u, exp_phi_u(u, N1, N2, s), 'r--', linewidth=2.)
    ax.plot(hbins[1:], pval, 'b-', linewidth=2.)
    ax.fill_between(hbins[1:], pval-perr, pval+perr, color='b', alpha=0.25, interpolate=True)

    ax.set_yscale('log')
    ax.set_ylim(1e-4, 10)
    ax.set_ylabel('P[max(q) > u]')
    ax.set_xlabel('u')
    fig.savefig('figures/GV_validate_{0}.png'.format(channel))
    plt.close()

def excursion_plot_1d(x, qscan, u1, suffix, path):
    fig, ax = plt.subplots()
    ax.set_xlabel('M_mumu [GeV]')
    ax.set_ylabel('q')
    ax.set_xlim([12., 70.])
    ax.plot(x, qscan, 'r-', linewidth=2.)
    ax.plot([12., 70.], [u1, u1], 'k-', linewidth=2.)

    fig.savefig('{0}/excursion_1D_{1}.pdf'.format(path, suffix))
    fig.savefig('{0}/excursion_1D_{1}.png'.format(path, suffix))
    plt.close()

if __name__ == '__main__':
    start = timer()

    ### Config 
    minalgo     = 'SLSQP'
    channel     = '1b1f'
    xlimits     = (12., 70.)
    nscan       = (50, 30)
    nsims       = 10
    ndim        = 1
    u1, u2      = 1., 2.
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

    scan_bnds = [(-0.9, 0.9), (0.04, 0.1)]
    if ndim == 1:
        nscan = (nscan[0], 1)
        scan_vals = np.array([(n1, params['width']) for n1 in np.linspace(scan_bnds[0][0], scan_bnds[0][1], nscan[0])]) 
        scan_div = ((scan_bnds[0][1] - scan_bnds[0][0])/nscan[0], 0.)
    elif ndim == 2:
        scan_vals = np.array([(n1, n2) 
                             for n1 in np.linspace(scan_bnds[0][0], scan_bnds[0][1], nscan[0]) 
                             for n2 in np.linspace(scan_bnds[1][0], scan_bnds[1][1], nscan[1])])

        scan_div = ((scan_bnds[0][1] - scan_bnds[0][0])/nscan[0], (scan_bnds[1][1] - scan_bnds[1][0])/nscan[1])

    ### Get data and scale
    #data, n_total = get_data('data/ntuple_{0}.csv'.format(channel), 'dimuon_mass', xlimits)
    data, n_total = get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)

    #######################
    ### Calculate LEE2D ###
    #######################

    ### scan over test data
    bnds    = [(0., 1.0), # A
               2*(params['mu'], ), 2*(params['width'], ), # mean, sigma
               (-1., 1.), (-1., 1.) # a1, a2 
               ]
    nll_bg = bg_objective(bg_params.values(), data)

    print 'Scanning ll ratio over data...'
    qscan = []
    qmax  = 2*(nll_bg - bg_sig_objective(params.values(), data))
    for i, scan in enumerate(scan_vals):
        # Remove edge effects
        if scan[0] - 3*scan[1] < -1 or scan[0] + 3*scan[1] > 1: 
            qscan.append(0.)
            continue

        bnds[1] = (scan[0], scan[0]+scan_div[0])
        bnds[2] = (scan[1], scan[1]+scan_div[1])
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

    paramscan   = []
    phiscan     = []
    qmax_mc     = []
    phi1        = []
    phi2        = []
    u_0         = np.linspace(0., 20., 1000.)
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

            bnds[1] = (scan[0], scan[0]+scan_div[0])
            bnds[2] = (scan[1], scan[1]+scan_div[1])
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

        if False: #make_plots and i < 9:
            sim = scale_data(sim, invert=True)
            fit_plot(sim, combined_model, params_best, 
                     legendre_polynomial, bg_result.x, 
                     '{0}_{1}'.format(channel,i+1), path='figures/scan_fits')
            if ndim == 1:
                excursion_plot_1d(np.linspace(xlimits[0], xlimits[1], nscan[0]), qscan, 1.,
                                 '{0}_{1}'.format(channel,i+1), path='figures/scan_fits')

        ### Doing calculations
        qscan = np.array(qscan).reshape(nscan)
        phiscan.append([calculate_euler_characteristic((qscan > u) + 0.) for u in u_0])
        phi1.append(calculate_euler_characteristic((qscan > u1) + 0.))
        phi2.append(calculate_euler_characteristic((qscan > u2) + 0.))
        
        if make_plots and i < 9 and ndim == 2: 
            cmap = axes1[i/3][i%3].pcolormesh(x, y, qscan, cmap='viridis', vmin=0., vmax=10.)
            axes2[i/3][i%3].imshow((qscan > u1) + 0., cmap='Greys', interpolation='none', origin='lower')
            axes3[i/3][i%3].imshow((qscan > u2) + 0., cmap='Greys', interpolation='none', origin='lower')

    phiscan = np.array(phiscan)
    if make_plots and ndim == 2:
        fig1.savefig('figures/qscan_toys_{0}.png'.format(channel))
        fig2.savefig('figures/qscan_u1_{0}.png'.format(channel))
        fig3.savefig('figures/qscan_u2_{0}.png'.format(channel))

    plt.close()


    ### Calculate LEE correction ###
    exp_phi1, exp_phi2 = np.mean(phi1), np.mean(phi2)
    var_phi1, var_phi2 = np.var(phi1), np.var(phi2)

    print 'u1 = {0}, u2 = {1}'.format(u1, u2)
    print 'E[phi_1] = {0} +/- {1}'.format(exp_phi1, np.sqrt(var_phi1))
    print 'E[phi_2] = {0} +/- {1}'.format(exp_phi2, np.sqrt(var_phi2))

    if ndim == 1:
        N1, p_global = lee_nD(np.sqrt(qmax), u_0, phiscan, j=1, k=1)
        print 'n1 = {0}'.format(N1)
        print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(norm.cdf(-np.sqrt(qmax)), np.sqrt(qmax))
        print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))

        validation_plots(u_0, phiscan, qmax_mc, N1, 0., 1., channel+'_1D')

        ### Scan over u ###
        exp_phi = phiscan.mean(axis=0) 
        var_phi = phiscan.var(axis=0) 
        lee_results = []
        for u, phi in zip(u_0, exp_phi):
            lee_results.append(lee1D(np.sqrt(qmax), u, phi, k=1))

        lee_results = np.array(lee_results).transpose()

    elif ndim == 2:
        N1, N2, p_global = do_LEE_correction(np.sqrt(qmax), u1, u2, exp_phi1, exp_phi2)
        validation_plots(u_0, phiscan, qmax_mc, N1, N2, channel+'_2D')

    # Save scan data
    outfile = open('data/lee_scan_{0}_{1}.pkl'.format(channel, nsims), 'w')
    pickle.dump(qmax_mc, outfile)
    pickle.dump(phiscan, outfile)
    outfile.close()

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))

