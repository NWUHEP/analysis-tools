#!/usr/bin/env python

import sys
import pickle

#from fitter import *
import future_fitter as ff
import toy_MC as mc

from scipy.optimize import fsolve
from scipy.stats import chi2, norm
from scipy.ndimage.morphology import *
from scipy.ndimage import *
from scipy.special import gamma
from scipy.misc import comb, factorial

def calculate_euler_characteristic(a):
   '''Calculate the Euler characteristic for level set a'''
   face_filter=np.zeros((2,2))+1
   right_edge_filter = np.array([[1,1]])
   bottom_edge_filter = right_edge_filter.T
   
   n_faces = np.sum(convolve(a,face_filter,mode='constant')>3)
   n_edges = np.sum(convolve(a,right_edge_filter,mode='constant')>1)
   n_edges += np.sum(convolve(a,bottom_edge_filter,mode='constant')>1)
   n_vertices = np.sum(a>0)
   
   EC = n_vertices-n_edges+n_faces
   #print '%d-%d+%d=%d' %(n_vertices,n_edges,n_faces,EulerCharacteristic) 
   
   return EC

def rho_g(u, j=1, k=1):
    '''
    From theorem 15.10.1 from Random Fields and Geometry (Adler and Taylor)

    Parameters
    ----------
    j: number of nuisance parameters (search dimensions)
    k: d.o.f. of chi2 random field
    u: threshold for excursions in the field
    '''

    coeff_num       = u**((k - j)/2.) * np.exp(-u/2.) 
    coeff_den       = (2.*np.pi)**(j/2.) * gamma(k/2.) * 2**((k-2.)/2.)
    indicate        = lambda m,l: float(k >= j - m - 2.*l)
    sum_fraction    = lambda m,l: ((-1.)**(j - 1. + m + l) * factorial(j - 1)) / (factorial(m)*factorial(l)*2.**l)
    m_terms         = lambda l: np.array([indicate(m,l) * comb(k-l, j-1.-m-2.*l) * sum_fraction(m,l) * u**(m+l) 
                                        for m in np.arange(0, 1 + int(j-1.-2.*l))])
    m_sum           = lambda l: np.sum(m_terms(l), axis=0)
    l_sum           = np.sum([m_sum(l) for l in np.arange(0., 1 + np.floor((j-1)/2))], axis=0) 

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
    return chi2.sf(u,k) + np.sum([n*rho_g(u, j+1, k) for j,n in enumerate(n_j)], axis=0)

def lee_objective(a, Y, dY, X):
    '''
    Defines the objective function for regressing the <EC> of our chi2 field.
    The minimization should be done on the quadratic cost weighted by the
    inverse of the variance on the measurement.  There is an additional term
    which will enforce preference for the fit result being greater than the
    data point.  The reasoning is that we would like to have an upper bound on
    our tails (that is, we are being conservative here).

    Parameters
    ----------
    a: list of parameters
    Y: target data
    dY: variance on the data
    X: independent variable values corresponding to values of Y
    '''

    ephi    = exp_phi_u(X, a[1:], k = a[0])
    qcost   = np.sum((Y - ephi)**2/dY)
    ubound  = np.sum(ephi < Y)/Y.size 
    L1_reg  = np.sum(np.abs(a)) 
    L2_reg  = np.sum(a**2)

    return qcost #+ 0.5*ubound

def lee_nD(max_local_sig, u, phiscan, j=1, k=None):
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
    k = assumed numbers of degrees of freedom of chi2 field. If not specified
        it will be a floating parmeter in the LEE estimation (recommended)
    '''
    exp_phi = phiscan.mean(axis=0)
    var_phi = phiscan.var(axis=0)

    ### Remove points where exp_phi > 0 ###
    phimask = (exp_phi > 0.)
    exp_phi = exp_phi[phimask]
    var_phi = var_phi[phimask]
    u       = u[phimask]

    ### if variance on phi is 0, use the poisson error on dY ###
    var_phi[var_phi==0] = 1./np.sqrt(phiscan.shape[0])
    
    if not k:
        print 'd.o.f. not specified => fit the EC with scan free parameters N_j and k...'
        bnds   = [(1, None)] + j*[(0., None)]
        p_init = [1.] + j*[1.,]
        result = minimize(lee_objective,
                          p_init,
                          method = 'Nelder-Mead',
                          args   = (exp_phi, var_phi, u),
                          #bounds = bnds
                          )
        k = result.x[0]
        n = result.x[1:]
    else:
        print 'd.o.f. specified => fit the EC scan with free parameters N_j and k={0}...'.format(k)
        mask  = np.arange(1, 1 + j)*100
        xvals = u[mask]
        ephis = exp_phi[mask]
        eq    = lambda n: [ephi - exp_phi_u(x, n, k=k) for ephi,x in zip(ephis, xvals)]
        n     = fsolve(eq, j*(1,))

    p_global = exp_phi_u(max_local_sig**2, n, k)

    return k, n, p_global

def validation_plots(u, phiscan, qmax, Nvals, kvals, channel):
    '''Check that the GV tails look okay'''

    ### Get the mean and variance from the phi scan ###
    phiscan = np.array(phiscan)
    exp_phi = np.mean(phiscan, axis=0)
    var_phi = np.var(phiscan, axis=0)
    qmax    = np.array(qmax)


    ### Construct the survival function spectrum from maximum q of each scan ###
    hval, hbins, _ = plt.hist(qmax, bins=30, range=(0.,30.), cumulative=True)
    hval = hval.max() - hval
    herr = np.sqrt(hval)
    pval = hval/hval.max()
    perr = pval*(herr/hval)
    pval = np.concatenate(([1], pval))
    perr = np.concatenate(([0], perr))
    plt.close()

    ### Remove points where values are 0 ###
    pmask = pval > 0.
    emask = exp_phi > 0.

    ### Make the plots ###
    fig, ax = plt.subplots()
    ax.plot(hbins[pmask], pval[pmask], 'm-', linewidth=2.)
    ax.plot(u[emask], exp_phi[emask], 'k-', linewidth=2.)
    ax.fill_between(hbins, pval-perr, pval+perr, color='m', alpha=0.25, interpolate=True)
    for N ,k in zip(Nvals, kvals):
        ax.plot(u, exp_phi_u(u, N, k), '--', linewidth=2.)


    ### Stylize ###
    ax.legend([r'$1 -  \mathrm{CDF}(q(\theta))$', r'$\overline{\phi}_{\mathrm{sim.}}$'] 
            + [r'$\overline{{\phi}}_{{ \mathrm{{th.}} }}; k={0}$'.format(k) if type(k) == int 
                else r'$\overline{{\phi}}_{{ \mathrm{{th.}} }}; k={0:.2f}$'.format(k) for k in kvals])
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 10)
    ax.set_ylabel(r'$\mathbb{\mathrm{P}}[q_{\mathrm{max}} > u]$')
    ax.set_xlabel(r'$u$')
    fig.savefig('figures/GV_validate_{0}.png'.format(channel))
    fig.savefig('figures/GV_validate_{0}.pdf'.format(channel))
    plt.close()

def excursion_plot_1d(x, qscan, u1, suffix, path):
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$M_{\mu\mu}$ [GeV]')
    ax.set_ylabel('q')
    ax.set_xlim([12., 70.])
    ax.plot(x, qscan, 'r-', linewidth=2.)
    ax.plot([12., 70.], [u1, u1], 'k-', linewidth=2.)

    fig.savefig('{0}/excursion_1D_{1}.pdf'.format(path, suffix))
    fig.savefig('{0}/excursion_1D_{1}.png'.format(path, suffix))
    plt.close()

if __name__ == '__main__':
    start = timer()

    ### Get command line arguments
    if len(sys.argv) > 2:
        channel = str(sys.argv[1])
        nsims   = int(sys.argv[2])
        ndim    = int(sys.argv[3])
    else:
        channel = '1b1f'
        nsims   = 100
        ndim    = 1

    ### Config 
    minalgo    = 'SLSQP'
    xlimits    = (12., 70.)
    nscan      = (50, 30)
    make_plots = True

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
    data, n_total = get_data('data/events_pf_{0}.csv'.format(channel), 'dimuon_mass', xlimits)

    #######################
    ### Calculate LEE2D ###
    #######################

    ### scan over test data
    bnds    = [(0., 1.0), # A
               2*(params['mu'], ), 2*(params['width'], ), # mean, sigma
               (None, None), (None, None) # a1, a2 
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

    if make_plots and ndim == 2:
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
    sims = mc.mc_generator(bg_pdf, n_total, nsims)

    if make_plots and ndim == 2:
        fig1, axes1 = plt.subplots(3, 3)

    paramscan   = []
    phiscan     = []
    qmaxscan    = []
    u_0         = np.linspace(0.01, 25., 1250.)
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
        qmaxscan.append(0)
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
            if qtest > qmaxscan[-1]: 
                params_best = result.x
                qmaxscan[-1] = qtest

        if make_plots and i < 9:
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
        
        if make_plots and i < 9 and ndim == 2: 
            cmap = axes1[i/3][i%3].pcolormesh(x, y, qscan, cmap='viridis', vmin=0., vmax=10.)

    qmaxscan    = np.array(qmaxscan)
    phiscan     = np.array(phiscan)
    paramscan   = np.array(paramscan)

    if make_plots and ndim == 2:
        fig1.savefig('figures/qscan_toys_{0}.png'.format(channel))
        plt.close()

    ### Calculate LEE correction ###
    k1, nvals1, p_global = lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=1)
    k2, nvals2, p_global = lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim, k=2)
    k, nvals, p_global   = lee_nD(np.sqrt(qmax), u_0, phiscan, j=ndim)
    #validation_plots(u_0, phiscan, qmaxscan, [nvals], [k], '{0}_{1}D'.format(channel, ndim))
    validation_plots(u_0, phiscan, qmaxscan, [nvals1, nvals2, nvals], [k1, k2, k], '{0}_{1}D'.format(channel, ndim))

    print 'k = {0:.2f}'.format(k)
    for i,n in enumerate(nvals):
        print 'N{0} = {1:.2f}'.format(i, n)
    print 'local p_value = {0:.7f},  local significance = {1:.2f}'.format(norm.cdf(-np.sqrt(qmax)), np.sqrt(qmax))
    print 'global p_value = {0:.7f}, global significance = {1:.2f}'.format(p_global, -norm.ppf(p_global))

    # Save scan data
    outfile = open('data/lee_scan_{0}_{1}.pkl'.format(channel, nsims), 'w')
    pickle.dump(u_0, outfile)
    pickle.dump(qmaxscan, outfile)
    pickle.dump(phiscan, outfile)
    pickle.dump(paramscan, outfile)
    outfile.close()

    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))

