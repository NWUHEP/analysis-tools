#!/usr/bin/env python

from fitter import *
# global options
np.set_printoptions(precision=3.)

### Fitting tools
def combination_bg_obj(params, X):
    '''
    Expects the list of parameters to be ordered as follow:
    a1, a2, b1, b2
    '''
    pdf1  = legendre_polynomial(X[0], (params[0], params[1]))
    pdf2  = legendre_polynomial(X[1], (params[2], params[3]))
    ll      = -np.sum(np.log(pdf1)) - np.sum(np.log(pdf2))
    return ll

def combination_sig_bg_obj(params, X):
    '''
    Expects the list of parameters to be ordered as follow:
    A_1, A_2, mu, sigma, a1, a2, b1, b2
    '''
    pdf_bg1     = legendre_polynomial(X[0], (params[4], params[5]))
    pdf_sig1    = gaussian(X[0], (params[2], params[3]))
    pdf1        = params[0]*pdf_bg1 + (1-params[0])*pdf_sig1

    pdf_bg2     = legendre_polynomial(X[1], (params[6], params[7]))
    pdf_sig2    = gaussian(X[1], (params[2], params[3]))
    pdf2        = params[1]*pdf_bg2 + (1-params[1])*pdf_sig2

    ll          = -np.sum(np.log(pdf1)) - np.sum(np.log(pdf2))
    return ll

### Plotting scripts ###
def fit_plot(pdf, data, params, suffix):
    N       = data.size
    nbins   = 29.
    binning = 2.
    x = np.linspace(-1, 1, num=10000)
    y = (N*binning/nbins)*pdf(x, params) 
    x = scale_data(x, invert=True)

    h = plt.hist(data, bins=nbins, range=[12., 70.], normed=False, histtype='step')
    bincenters  = (h[1][1:] + h[1][:-1])/2.
    binerrs     = np.sqrt(h[0]) 

    plt.clf()
    plt.errorbar(bincenters, h[0], yerr=binerrs, fmt='o')
    plt.plot(x, y, linewidth=2.)
    if suffix == '1b1f':
        plt.title('mumu + 1 b jet + 1 forward jet')
    elif suffix == '1b1c':
        plt.title('mumu + 1 b jet + 1 central jet + MET < 40 + deltaPhi(mumu,bj)')
    plt.xlabel('M_mumu [GeV]')
    plt.ylabel('entries / 2 GeV')
    plt.xlim([12., 70.])
    plt.ylim([0., np.max(y)*1.8])
    plt.savefig('figures/dimuon_mass_fit_{0}.pdf'.format(suffix))
    plt.close()

    #plt.rc('text', usetex=True)
    #fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    #ax1 = axes[0]
    #ax1.errorbar(bincenters, h[0], yerr=binerrs, fmt='o')
    #ax1.plot(x, y)
    #ax1.set_xlabel('M_{\mu\mu} [GeV]')
    #ax1.set_ylabel('entries/2 GeV')
    #fig.show()

if __name__ == '__main__':
    # Start the timer
    start = timer()

    # get data and convert variables to be on the range [-1, 1]
    print 'Getting data and scaling to lie in range [-1, 1].'
    ntuple_1b1f = pd.read_csv('data/ntuple_1b1f.csv')
    data_1b1f   = ntuple_1b1f['dimuon_mass'].values
    data_1b1f   = np.apply_along_axis(scale_data, 0, data_1b1f, xlow=12, xhigh=70)
    N_1b1f      = data_1b1f.size

    ntuple_1b1c = pd.read_csv('data/ntuple_1b1c.csv')
    data_1b1c   = ntuple_1b1c['dimuon_mass'].values
    data_1b1c   = np.apply_along_axis(scale_data, 0, data_1b1c, xlow=12, xhigh=70)
    N_1b1c      = data_1b1c.size

    # fit background only model
    print 'Performing background only fit with second order Legendre polynomial normalized to unity.'
    bnds = [(0., 2.), (0., 0.5), # a1, a2
            (0., 2.), (0., 0.5)] # b1, b2
    bg_result = minimize(regularization, 
                         [0.5, 0.05, 0.5, 0.05], 
                         method = 'SLSQP', 
                         bounds = bnds,
                         args   = ([data_1b1f, data_1b1c], combination_bg_obj)
                         )
    bg_sigma, bg_corr = get_corr(combination_bg_obj, bg_result.x, [data_1b1f, data_1b1c])   

    print '\n'
    print 'RESULTS'
    print '-------'
    print 'a1       = {0:.3f} +/- {1:.3f}'.format(bg_result.x[0], bg_sigma[0])
    print 'a2       = {0:.3f} +/- {1:.3f}'.format(bg_result.x[1], bg_sigma[1])
    print 'b1       = {0:.3f} +/- {1:.3f}'.format(bg_result.x[2], bg_sigma[2])
    print 'b2       = {0:.3f} +/- {1:.3f}'.format(bg_result.x[3], bg_sigma[3])
    print'\n'
    #print 'correlation matrix:'
    #print bg_corr
    #print'\n'

    # fit signal+background model
    print 'Performing background plus signal fit with second order Legendre polynomial normalized to unity plus a Gaussian kernel.'
    bnds = [(0., 1.05), (0., 1.05),# A1, A2
            (-0.8, -0.2), (0., 0.4), # mean, sigma
            (0., 2.), (0., 0.5), # a1, a2
            (0., 2.), (0., 0.5)] # b1, b2
    result = minimize(regularization, 
                      [1., 1., -0.3, 0.1, bg_result.x[0], bg_result.x[1], bg_result.x[2], bg_result.x[3]], 
                      method = 'SLSQP',
                      #jac    = True,
                      args   = (data_scaled, bg_sig_objective, 1., 1.),
                      bounds = bnds
                      )
    comb_sigma, comb_corr = get_corr(combination_bg_sig_obj, result.x, [data_1b1f, data_1b1c])   
    qtest = np.sqrt(2*np.abs(combination_bg_sig_obj(result.x, [data_1b1f, data_1b1c]) - bg_objective(bg_result.x, [data_1b1f, data_1b1c])))

    # Convert back to measured mass values
    pct_sigma   = np.abs(comb_sigma/result.x)
    mu          = scale_data(result.x[2], invert=True) 
    sig_mu      = mu*pct_sigma[1]
    width       = result.x[3]*(70. - 12.)/2. 
    sig_width   = width*pct_sigma[2]

    print '\n'
    print 'RESULTS'
    print '-------'
    print 'A1       = {0:.3f} +/- {1:.3f}'.format(result.x[0], comb_sigma[0])
    print 'A2       = {0:.3f} +/- {1:.3f}'.format(result.x[1], comb_sigma[1])
    print 'mu       = {0:.3f} +/- {1:.3f}'.format(mu, sig_mu)
    print 'width    = {0:.3f} +/- {1:.3f}'.format(width, sig_width)
    print 'a1       = {0:.3f} +/- {1:.3f}'.format(result.x[4], comb_sigma[4])
    print 'a2       = {0:.3f} +/- {1:.3f}'.format(result.x[5], comb_sigma[5])
    print 'b1       = {0:.3f} +/- {1:.3f}'.format(result.x[6], comb_sigma[6])
    print 'b2       = {0:.3f} +/- {1:.3f}'.format(result.x[7], comb_sigma[7])
    print'\n'
    print 'Correlation matrix:'
    print comb_corr
    print'\n'

    #=======================#
    ### Caluculate yields ###
    #=======================#
    # integrate over background only function in the range (mu - 2*sigma, mu +
    # 2*sigma) to determine background yields.  Signal yields come from
    # N*(1-A).
    f_bg    = lambda x: legendre_polynomial(x, (result.x[3], result.x[4]))
    xlim    = (result.x[1] - 2*result.x[2], result.x[1] + 2*result.x[2])
    N_b     = result.x[0]*N*integrate.quad(f_bg, xlim[0], xlim[1])[0]
    sig_b   = N_b/np.sqrt(N*result.x[0])
    N_s     = N*(1 - result.x[0]) 
    sig_s   = N*comb_sigma[0]

    print 'N_b = {0:.2f} +\- {1:.2f}'.format(N_b, sig_b)
    print 'N_s = {0:.2f} +\- {1:.2f}'.format(N_s, sig_s)
    print 'q = {0:.3f}'.format(qtest)

    ### Simple p-value ###
    print ''
    print 'Calculating local p-value and significance...'
    toys    = rng.normal(N_b, sig_b, int(1e8))
    pvars   = rng.poisson(toys)
    pval    = pvars[pvars > N_b + N_s].size/1e8
    print 'local p-value = {0}'.format(pval)
    print 'local significance = {0:.2f}'.format(np.abs(norm.ppf(pval)))

    ### Make plots ###
    fit_plot(combined_model, data, result.x, channel)

    '''
    print ''
    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
