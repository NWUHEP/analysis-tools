from fitter import *

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
