#!/usr/bin/env python

from lookee import *
from combination_fitter import *
from toy_MC import *
from lee2d import *

if __name__ == '__main__':
    start = timer()

    ### Get command line arguments
    if len(sys.argv) > 2:
        channel = str(sys.argv[1])
        nsims   = int(sys.argv[2])
        ndim    = int(sys.argv[3])
    else:
        channel = 'combined'
        nsims   = 100
        ndim    = 2

    ### Config 
    minalgo = 'SLSQP'
    xlimits = (12., 70.)
    nscan   = (50, 30)
    make_plots = True

    ### Parameters from fitting combined spectra
    params      = {'A1':0.89, 'A2':0.96, 'mu':-0.42, 'width':0.05, 'a1':0.31, 'a2':0.12, 'b1':0.21, 'b2':0.08} 
    bg_params   = {'a1':0.21, 'a2':0.02, 'b1':0.17, 'b2':0.06} 

    ### Define scan bounds and create subdivisions
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
    data_1b1f, n1_tot = get_data('data/events_pf_1b1f.csv', 'dimuon_mass', xlimits)
    data_1b1c, n2_tot = get_data('data/events_pf_1b1c.csv', 'dimuon_mass', xlimits)
    data = (data_1b1f, data_1b1c)

    #######################
    ### Calculate LEE2D ###
    #######################

    ### scan over test data
    bnds = [(0., 1.), (0., 1.), # A1, A2
            2*(params['mu'], ), 2*(params['width'], ), # mean, sigma
            (-1., 1.), (-1., 1.), # a1, a2
            (-1., 1.), (-1., 1.)  # b1, b2
            ]
    nll_bg = combination_bg_obj(bg_params.values(), data)

    print 'Scanning ll ratio over data...'
    qscan = []
    qdata = 0. #2*(nll_bg - combination_bg_sig_obj(params.values(), data))
    for i, scan in enumerate(scan_vals):
        # Remove edge effects
        if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
            qscan.append(0.)
            continue

        # Carry out combined fit minimizing the nll sum
        bnds[2] = (scan[0], scan[0]+scan_div[0])
        bnds[3] = (scan[1], scan[1]+scan_div[1])
        result = minimize(regularization, 
                               [1., 1., scan[0], scan[1], params['a1'], params['a2'], params['b1'], params['b2']],
                               method = 'SLSQP',
                               args   = ([data_1b1f, data_1b1c], combination_bg_sig_obj),
                               bounds = bnds
                               )

        qtest = 2*(nll_bg - combination_bg_sig_obj(result.x, data))
        qscan.append(qtest)
        if qtest > qdata: 
            qdata = qtest

    # convert scan value to physical values and prepare colormesh
    x       = scan_vals[:,0].reshape(nscan)
    y       = scan_vals[:,1].reshape(nscan)
    qscan   = np.array(qscan).reshape(nscan)

    if make_plots and ndim == 2:
        fig, ax = plt.subplots()
        cmap = ax.pcolormesh(x, y, qscan, cmap='viridis', vmin=0., vmax=25.)
        fig.colorbar(cmap)
        ax.set_xlabel('M_{mumu} [GeV]')
        ax.set_ylabel('width [GeV]')
        fig.savefig('figures/qscan_data_combination.png')
        plt.close()

    print 'q_max = {0}'.format(qdata)

    ### Make some pseudo-data ###
    print 'Scanning ll ratio over {0} pseudo-datasets...'.format(nsims)
    bg_pdf1 = lambda x: 0.5 + bg_params['a1']*x + 0.5*bg_params['a2']*(3*x**2 -1)
    sims1   = mc_generator(bg_pdf1, data_1b1f.size, nsims)

    bg_pdf2 = lambda x: 0.5 + bg_params['b1']*x + 0.5*bg_params['b2']*(3*x**2 -1)
    sims2   = mc_generator(bg_pdf2, data_1b1c.size, nsims)

    if make_plots and ndim == 2:
        fig1, axes1 = plt.subplots(3, 3)

    paramscan   = []
    phiscan     = []
    qmaxscan    = []
    u_0         = np.linspace(0.01, 25., 1250.)
    for i, sim in enumerate(zip(sims1, sims2)):
        if not i%10: 
            print 'Carrying out scan {0}...'.format(i+1)

        bg_result = minimize(regularization, 
                             (bg_params['a1'], bg_params['a2'], bg_params['b1'], bg_params['b2']),
                             method = 'SLSQP',
                             args   = (sim, combination_bg_obj),
                             bounds = bnds[-4:]
                             )
        nll_bg = combination_bg_obj(bg_result.x, sim)

        qscan       = []
        params_best = []
        qmax = 0
        for j, scan in enumerate(scan_vals):
            if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
                qscan.append(0.)
                continue

            bnds[2] = (scan[0], scan[0]+scan_div[0])
            bnds[3] = (scan[1], scan[1]+scan_div[1])
            result = minimize(regularization, 
                              (1., 1., scan[0], scan[1], params['a1'], params['a2'], params['b1'], params['b2']),
                              method = 'SLSQP',
                              args   = (sim, combination_bg_sig_obj),
                              bounds = bnds
                              )
            qtest = 2*(nll_bg - combination_bg_sig_obj(result.x, sim))
            qscan.append(qtest)
            if qtest > qmax: 
                params_best = result.x
                qmax = qtest
    
        qmaxscan.append(qmax)
        paramscan.append(params_best)

        if make_plots and i < 10:
            sim_1b1f = scale_data(sim[0], invert=True)
            sim_1b1c = scale_data(sim[1], invert=True)
            result_1b1f = params_best[np.array([0,2,3,4,5])]
            result_1b1c = params_best[np.array([1,2,3,6,7])]
            fit_plot(sim_1b1f, 
                     combined_model, result_1b1f, 
                     legendre_polynomial, bg_result.x[:2], 
                     '1b1f_combined_{0}'.format(i+1), path='figures/scan_fits')
            fit_plot(sim_1b1c,
                     combined_model, result_1b1c, 
                     legendre_polynomial, bg_result.x[2:], 
                     '1b1c_combined_{0}'.format(i+1), path='figures/scan_fits')

        ### Doing calculations
        qscan = np.array(qscan).reshape(nscan)
        phiscan.append([calculate_euler_characteristic((qscan > u) + 0.) for u in u_0])
        
        #if make_plots and i < 9: 
        #    cmap = axes1[i/3][i%3].pcolormesh(mass, width, qscan, cmap='viridis', vmin=0., vmax=10.)
        #    axes2[i/3][i%3].pcolormesh(mass, width, A_u1, cmap='Greys')
        #    axes3[i/3][i%3].pcolormesh(mass, width, A_u2, cmap='Greys')

    qmaxscan    = np.array(qmaxscan)
    phiscan     = np.array(phiscan)
    paramscan   = np.array(paramscan)

    if make_plots and ndim == 2:
        fig1.subplots_adjust(right=0.8)
        fig1.colorbar(cmap)
        axes1.set_xlabel('M_{mumu} [GeV]')
        axes1.set_ylabel('width [GeV]')
        fig.savefig('figures/qscan_data_combination.png')
        fig1.savefig('figures/qscan_toys_combination.png')


    ### Calculate LEE correction ###
    k1, nvals1, p_global = lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=1)
    k2, nvals2, p_global = lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=2)
    k, nvals, p_global   = lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim)
    validation_plots(u_0, phiscan, qmaxscan, 
                     [nvals1, nvals2, nvals], [k1, k2, k], 
                     '{0}_{1}D'.format('combination', ndim))

    print 'k = {0:.2f}'.format(k)
    for i,n in enumerate(nvals):
        print 'N{0} = {1:.2f}'.format(i, n)
    print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(norm.cdf(-np.sqrt(qmax)), np.sqrt(qmax))
    print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))

    # Save scan data
    outfile = open('data/lee_scan_combination_{0}.pkl'.format(nsims), 'w')
    pickle.dump(u_0, outfile)
    pickle.dump(qmaxscan, outfile)
    pickle.dump(phiscan, outfile)
    pickle.dump(paramscan, outfile)
    outfile.close()

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
