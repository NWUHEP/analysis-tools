from fitter import *

import emcee

### For monte carlo studies ###
def mc_generator(pdf, samp_per_toy=100, ntoys=1, domain=(-1.,1.)):
    '''Rejection sampling with broadcasting gives approximately the requested number of toys'''

    # Generate random numbers and map into domain
    rnums = rng.rand(2, 2*ntoys*samp_per_toy) 
    x = rnums[0]
    x = (domain[1] - domain[0])*x + domain[0]

    # Carry out rejection sampling
    keep = pdf(x) > rnums[1]
    x    = x[keep]
    
    # Remove excess events and shape to samp_per_toy
    x = x[:-(x.size%samp_per_toy)]
    x = x.reshape(x.size/samp_per_toy, samp_per_toy)

    # if ntoys is not produced try again
    #if x.shape[0] < ntoys:
    #    x = np.concatenate((x, mc_generator(pdf, samp_per_toy, (ntoys-x.shape[0]), domain)))

    return x

def lnprob(x, pdf, bounds):
    if np.any(x < bounds[0]) or np.any(x > bounds[1]):
        return -np.inf
    else:
        return np.log(pdf(x))

def emcee_generator_1D(pdf, samples_per_toy=100, ntoys=10, bounds=(-1, 1)):
    '''
    Wrapper for emcee the MCMC hammer

    Parameters
    ==========
    pdf: distribution to be samples
    samples_per_toy: number of draws to be assigned to each pseudo-experiment
    ntoys: number of toy models to produce
    bounds: (xmin, xmax) for values of X
    '''
    ndim = 1
    nwalkers = 100
    #lnprob = lambda x: np.log(pdf(x))*np.any(x > bounds[0])*np.any(x < bounds[1])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[pdf, bounds])

    p0 = [np.random.rand(1) for i in xrange(nwalkers)]
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 1000, rstate0=state)

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 250-dimensional
    # vector.
    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
    
    # Estimate the integrated autocorrelation time for the time series in each
    # parameter.
    print("Autocorrelation time:", sampler.get_autocorr_time())
    
    # Finally, you can plot the projected histograms of the samples using
    # matplotlib as follows (as long as you have it installed).

    plt.hist(sampler.flatchain[:,0], 100)
    plt.show()

    return sampler.flatchain[:, 0]

def local_sig(p, ntoys, samp_per_toy):
    bg_pdf  = lambda x: 0.5 + p['a1']*x + 0.5*p['a2']*(3*x**2 -1)
    sim = mc_generator(bg_pdf, samp_per_toy, ntoys)
    results = []
    nlls = []
    bnds = [(0., 1.04), # A
            2*(p['mu'], ), 2*(p['width'], ), # mean, sigma
            2*(p['a1'], ), 2*(p['a2'], )] # a1, a2
    for toy in sim:
        res = minimize(regularization, 
                       [1., p['mu'], p['width'], p['a1'], p['a2']], 
                       method = 'SLSQP',
                       #jac    = True,
                       args   = (toy, bg_sig_objective, 0., 0.),
                       bounds = bnds)
        if res.success:
            results.append(res.x) 
            nlls.append(bg_sig_objective(res.x, toy))

    #return sim, np.array(results), nlls
    return  np.array(results)

if __name__ == '__main__':
    start = timer()

    ### Load fit ###

    ### TOY MC ###
    #proc = Process(name='test_process', target=local_sig, args=(bg_result.x, result.x, 10000))
    params = {'A':0.88, 'mu':-0.42, 'width':0.054, 'a1':0.32, 'a2':0.133}

    output = []
    for _ in range(10):
        mc_results = local_sig(params, ntoys=100000, samp_per_toy=166)
        output.append(mc_results[:,0])

    #plt.yscale('log')
    plt.hist(1-mc_results[:,0], bins=50, range=[-0.2, 0.2], histtype='stepfilled')
    plt.show()

    ### Save to file ###
    print 'Saving result of toy MC to file data/toy_output.pkl'
    toy_file = open('data/toy_ouput.pkl', 'wb')
    #pickle.dump(mc_toys, toy_file)
    pickle.dump(mc_results, toy_file)
    toy_file.close()

    print timer() - start
