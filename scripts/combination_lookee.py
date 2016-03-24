#!/usr/bin/env python

from combination_fitter import *
from toy_MC import *
from lee2d import *

if __name__ == '__main__':
    start = timer()

    ### Config 
    xlimits = (12., 70.)
    nscan   = 100
    nsims   = 10
    scan_vals = np.array([(n1, n2) for n1 in np.linspace(-0.9, 0.9, nscan) for n2 in np.linspace(0.02, 0.15, nscan)])

    params      = {'A1':0.89, 'A2':0.96, 'mu':-0.42, 'width':0.05, 'a1':0.31, 'a2':0.12, 'b1':0.21, 'b2':0.08} 
    bg_params   = {'a1':0.21, 'a2':0.02, 'b1':0.17, 'b2':0.06} 

    ### Get data and scale
    ntuple_1b1f = pd.read_csv('data/ntuple_1b1f.csv')
    data_1b1f   = ntuple_1b1f['dimuon_mass'].values
    data_1b1f   = np.apply_along_axis(scale_data, 0, data_1b1f, xlow=xlimits[0], xhigh=xlimits[1])
    N1          = data_1b1f.size

    ntuple_1b1c = pd.read_csv('data/ntuple_1b1c.csv')
    data_1b1c   = ntuple_1b1c['dimuon_mass'].values
    data_1b1c   = np.apply_along_axis(scale_data, 0, data_1b1c, xlow=xlimits[0], xhigh=xlimits[1])
    N2          = data_1b1c.size

    data = [data_1b1f, data_1b1c]

    #######################
    ### Calculate LEE2D ###
    #######################

    ### scan over test data
    bnds = [(0., 1.05), (0., 1.05), # A1, A2
            2*(params['mu'], ), 2*(params['width'], ), # mean, sigma
            2*(params['a1'], ), 2*(params['a2'], ), # a1, a2
            2*(params['b1'], ), 2*(params['b2'], )] # b1, b2

    llbg = combination_bg_obj(bg_params.values(), data)

    print 'Scanning ll ratio over data...'
    qscan = []
    qmax  = 0.
    for i, scan in enumerate(scan_vals):
        # Remove edge effects
        if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
            qscan.append(0.)
            continue

        # Carry out combined fit minimizing the nll sum
        bnds[2] = (scan[0], scan[0])
        bnds[3] = (scan[1], scan[1])
        scan_result = minimize(regularization, 
                               [1., 1., scan[0], scan[1], params['a1'], params['a2'], params['b1'], params['b2']],
                               method = 'SLSQP',
                               args   = ([data_1b1f, data_1b1c], combination_bg_sig_obj),
                               bounds = bnds
                               )

        qtest = 2*(llbg - combination_bg_sig_obj(scan_result.x, data))
        qscan.append(qtest)
        if qtest > qmax: 
            qmax = qtest

    # convert scan value to physical values and prepare colormesh
    x       = scan_vals[:,0].reshape((nscan, nscan))
    y       = scan_vals[:,1].reshape((nscan, nscan))
    qscan   = np.array(qscan).reshape((nscan, nscan))

    fig, ax = plt.subplots()
    cmap = ax.pcolormesh(x, y, qscan, cmap='viridis', vmin=0., vmax=25.)
    fig.colorbar(cmap)
    ax.set_xlabel('M_{mumu} [GeV]')
    ax.set_ylabel('width [GeV]')
    fig.savefig('figures/qscan_data_combination.png')
    plt.close()

    print 'q_max = {0}'.format(qmax)

    ### Make some pseudo-data ###
    print 'Scanning ll ratio over {0} pseudo-datasets...'.format(nsims)
    bg_pdf1 = lambda x: 0.5 + bg_params['a1']*x + 0.5*bg_params['a2']*(3*x**2 -1)
    sims1   = mc_generator(bg_pdf1, data_1b1f.size, nsims)

    bg_pdf2 = lambda x: 0.5 + bg_params['b1']*x + 0.5*bg_params['b2']*(3*x**2 -1)
    sims2   = mc_generator(bg_pdf2, data_1b1c.size, nsims)

    fig1, axes1 = plt.subplots(3, 3)
    fig2, axes2 = plt.subplots(3, 3)
    fig3, axes3 = plt.subplots(3, 3)

    phi1 = []
    phi2 = []
    u1, u2 = 1., 2.
    for i, sim in enumerate(zip(sims1, sims2)):
        llbg = combination_bg_obj(bg_params.values(), sim)
        qscan = []
        for j, scan in enumerate(scan_vals):
            if scan[0] - 2*scan[1] < -1 or scan[0] + 2*scan[1] > 1: 
                qscan.append(0.)
                continue
            bnds[2] = (scan[0], scan[0])
            bnds[3] = (scan[1], scan[1])
            scan_result = minimize(regularization, 
                                   (1., 1., scan[0], scan[1], params['a1'], params['a2'], params['b1'], params['b2']),
                                   method = 'SLSQP',
                                   #jac    = True,
                                   args   = (sim, combination_bg_sig_obj),
                                   bounds = bnds
                                   )
            qtest = 2*(llbg - combination_bg_sig_obj(scan_result.x, sim))
            qscan.append(qtest)

        ### Doing calculations
        qscan_re = np.array(qscan).reshape(nscan, nscan)
        A_u1 = (qscan_re > u1) + 0.
        A_u2 = (qscan_re > u2) + 0.
        phi1.append(calculate_euler_characteristic(A_u1))
        phi2.append(calculate_euler_characteristic(A_u2))
        
        if i < 9: 
            cmap = axes1[i/3][i%3].pcolormesh(mass, width, qscan_re, cmap='viridis', vmin=0., vmax=10.)
            axes2[i/3][i%3].pcolormesh(mass, width, A_u1, cmap='Greys')
            axes3[i/3][i%3].pcolormesh(mass, width, A_u2, cmap='Greys')

    fig1.subplots_adjust(right=0.8)
    fig1.colorbar(cmap)
    axes1.set_xlabel('M_{mumu} [GeV]')
    axes1.set_ylabel('width [GeV]')
    fig.savefig('figures/qscan_data_combination.png')
    fig1.savefig('figures/qscan_toys_combination.png')

    axes2.set_xlabel('M_{mumu} [GeV]')
    axes2.set_ylabel('width [GeV]')
    fig2.savefig('figures/qscan_u1_combination.png')

    axes3.set_xlabel('M_{mumu} [GeV]')
    axes3.set_ylabel('width [GeV]')
    fig3.savefig('figures/qscan_u2_combination.png')
    plt.close()

    ### Calculate LEE correction ###
    exp_phi1, exp_phi2 = np.mean(phi1), np.mean(phi2)
    print 'E[phi_1] = {0}'.format(exp_phi1)
    print 'E[phi_2] = {0}'.format(exp_phi2)
    do_LEE_correction(np.abs(norm.ppf(1 - chi2.cdf(qmax, 1))), u1, u2, exp_phi1, exp_phi2)

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
