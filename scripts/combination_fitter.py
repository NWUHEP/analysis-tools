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

def combination_bg_sig_obj(params, X):
    '''
    Expects the list of parameters to be ordered as follow:
    A_1, A_2, mu, sigma, a1, a2, b1, b2
    '''
    pdf_bg1     = legendre_polynomial(X[0], params[4:6])
    pdf_sig1    = gaussian(X[0], params[2:4])
    pdf1        = params[0]*pdf_bg1 + (1 - params[0])*pdf_sig1

    pdf_bg2     = legendre_polynomial(X[1], params[6:8])
    pdf_sig2    = gaussian(X[1], params[2:4])
    pdf2        = params[1]*pdf_bg2 + (1 - params[1])*pdf_sig2

    ll          = -np.sum(np.log(pdf1)) - np.sum(np.log(pdf2))
    return ll

if __name__ == '__main__':
    # Start the timer
    start = timer()

    # get data and convert variables to be on the range [-1, 1]
    print 'Getting data and scaling to lie in range [-1, 1].'
    xlimits = (12., 70.)
    data_1b1f, N1 = get_data('data/events_pf_1b1f.csv', 'dimuon_mass', xlimits)
    data_1b1c, N2 = get_data('data/events_pf_1b1c.csv', 'dimuon_mass', xlimits)
    data = (data_1b1f, data_1b1c)

    # fit background only model
    print 'Performing background only fit with second order Legendre polynomial normalized to unity.'
    bnds = [(0., 2.), (0., 0.5), # a1, a2
            (0., 2.), (0., 0.5)] # b1, b2
    bg_result = minimize(regularization, 
                         [0.5, 0.05, 0.5, 0.05], 
                         method = 'SLSQP', 
                         bounds = bnds,
                         args   = (data, combination_bg_obj)
                         )
    bg_sigma, bg_corr = get_corr(combination_bg_obj, bg_result.x, data)   

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
    bnds = [(0., 1.05), (0., 1.05),     # A1, A2
            (-0.8, -0.2), (0.05, 0.5),    # mean , sigma 
            (-1., 1.), (-1., 1.),        # a1, a2
            (-1., 1.), (-1., 1.)]        # b1, b2
    inits = [1., 1., -0.4, 0.1, 
            bg_result.x[0], bg_result.x[1], 
            bg_result.x[2], bg_result.x[3]] 
    result = minimize(regularization, inits,
                      method = 'SLSQP',
                      args   = (data, combination_bg_sig_obj),
                      bounds = bnds
                      )
    comb_sigma, comb_corr = get_corr(combination_bg_sig_obj, result.x, data)   
    qtest = np.sqrt(2*np.abs(combination_bg_sig_obj(result.x, data) - combination_bg_obj(bg_result.x, data)))

    # Convert back to physical mass values
    pct_sigma   = np.abs(comb_sigma/result.x)
    mu          = scale_data(result.x[2], invert=True) 
    sig_mu      = mu*pct_sigma[2]
    width       = result.x[3]*(70. - 12.)/2. 
    sig_width   = width*pct_sigma[3]

    print '\n'
    print 'RESULTS'
    print '-------'
    print 'A1       = {0:.3f} +/- {1:.3f}'.format(1 - result.x[0], comb_sigma[0])
    print 'A2       = {0:.3f} +/- {1:.3f}'.format(1 - result.x[1], comb_sigma[1])
    print 'mu1      = {0:.3f} +/- {1:.3f}'.format(mu, sig_mu)
    print 'width1   = {0:.3f} +/- {1:.3f}'.format(width, sig_width)
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
    N       = N1 + N2
    A1      = result.x[0]
    A2      = result.x[1]
    A       = (A1*N1 + A2*N2)/N
    f_bg1   = lambda x: legendre_polynomial(x, (result.x[4], result.x[5])) 
    f_bg2   = lambda x: legendre_polynomial(x, (result.x[6], result.x[7]))
    xlim    = (result.x[2] - 2*result.x[3], result.x[2] + 2*result.x[3])
    N_b1    = A1*N1*integrate.quad(f_bg1, xlim[0], xlim[1])[0]
    N_b2    = A2*N2*integrate.quad(f_bg2, xlim[0], xlim[1])[0]
    sig_b   = (N_b1 + N_b2)/np.sqrt(N*A)
    N_s     = N*(1 - A) 
    sig_s   = np.sqrt(N_s) #N*comb_sigma[0]

    print 'N_b = {0:.2f} +\- {1:.2f}'.format(N_b1+N_b2, sig_b)
    print 'N_s = {0:.2f} +\- {1:.2f}'.format(N_s, sig_s)

    ### Simple p-value ###
    calc_local_pvalue(N_b1+N_b2, N_s, sig_b, 1e8)
    print 'sqrt(q) = {0:.3f}'.format(qtest)

    ### Make plots ###
    result_1b1f = result.x[np.array([0,2,3,4,5])]
    result_1b1c = result.x[np.array([1,2,3,6,7])]
    fit_plot(scale_data(data_1b1f, invert=True), combined_model, result_1b1f, legendre_polynomial, bg_result.x[:2], '1b1f_combined')
    fit_plot(scale_data(data_1b1c, invert=True), combined_model, result_1b1c, legendre_polynomial, bg_result.x[2:], '1b1c_combined')

    print ''
    print 'Runtime = {0:.2f} ms'.format(1e3*(timer() - start))
