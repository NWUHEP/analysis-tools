#!/home/naodell/opt/anaconda3/bin/python

import pickle
import os
from functools import partial
from collections import namedtuple
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize, basinhopping
from tqdm import tqdm

import scripts.plot_tools as pt
import scripts.fit_helpers as fh
from nllfit.nllfitter import ScanParameters


ScanData = namedtuple('ScanData', ['param_name', 'scan_points', 'results', 'costs'])

if __name__ == '__main__':

    # input arguments
    parser = argparse.ArgumentParser(description='Run likelihood scans for all parameters.')
    parser.add_argument('input',
                        help = 'specify input directory',
                        type = str
                        )
    parser.add_argument('-p', '--parameter-file',
                        help = 'Change from default parameter file.',
                        default = 'data/model_parameters_default.csv',
                        type = str
                        )
    args = parser.parse_args()
    ##########################

    processes  = ['ttbar', 't', 'ww', 'wjets', 'zjets_alt', 'diboson', 'fakes']
    selections = [
                  'ee', 'mumu',
                  'emu',
                  'mutau', 'etau',
                  'mu4j', 'e4j'
                 ]
    plot_labels = fh.fancy_labels
    
    # initialize fit data and generate asimov dataset
    if os.path.isdir(args.input):
        fit_data = fh.FitData(args.input, selections, processes, process_cut=0.05)
    else:
        infile = open(args.input, 'rb')
        fit_data = pickle.load(infile)
        infile.close()

    parameters  = fit_data._parameters.copy()
    params_pre  = parameters['val_init'].values.copy()
    asimov_data = {cat:fit_data.mixture_model(params_pre, cat) for cat in fit_data._model_data.keys()}

    # minimizer options
    min_options = dict(#eps=1e-9, 
                       #xtol=1e-3, 
                       #ftol=1e-9, 
                       #stepmx=0.1, 
                       #maxCGit=50, 
                       #accuracy=1e-10,
                       gtol=1e-4,
                       disp=None
                      )

    # configure the objective
    mask = fit_data._pmask
    fobj = partial(fit_data.objective,
                   data = asimov_data,
                   do_bb_lite = True,
                   lu_test = None
                  )

    fobj_jac = partial(fit_data.objective_jacobian,
                       data = asimov_data,
                       do_bb_lite = True,
                       lu_test = None 
                      )

    # prepare scan data
    scan_dict = dict()
    for ix, (pname, pdata) in tqdm(enumerate(parameters.iterrows()), total=parameters.shape[0]):

        if pdata.active == 0:# or ix < 7:
            continue
                            
        mask[ix] = False
        scan_vals = np.linspace(pdata.val_fit - 3*pdata.err_fit, pdata.val_fit + 3*pdata.err_fit, 7)
        #scan_vals = np.linspace(pdata.val_init - 3*pdata.err_init, pdata.val_init + 3*pdata.err_init, 7)

        # carry out scan and save results
        results   = []
        cost      = []
        sv_accept = []
        for sv in tqdm(scan_vals, leave=False):

            # set scan value and carry out minimization
            fit_data._pval_fit[ix] = sv
            #pinit = fit_data._pval_fit[mask]
            pinit = params_pre[mask].copy()
            result = minimize(fobj, pinit,
                              jac     = fobj_jac,
                              method  = 'BFGS',
                              options = min_options,
                             )

            
            sv_accept.append(sv)
            results.append(result.x)
            cost.append(result.fun)
            
            #tqdm.write(f'{pname} = {sv:.3f}: {fobj(pinit):.2f}, {result.fun:.2f}')
            #if result.success or result.status == 1:
            #    sv_accept.append(sv)
            #    results.append(result.x)
            #    cost.append(result.fun)
            #    
            #    # unpack cost cache
            #    #new_cache = []
            #    #for cat, cache in fit_data._cache.items():
            #    #    new_cache.extend(cache['cost'])
            #    #cost_cache.append(new_cache)
            #    
            #else:
            #    print(result)
            #    print(sv)
                
        mask[ix] = True
        fit_data._pval_fit[ix] = params_pre[ix]

        # process scan data
        results   = np.array(results)
        cost      = np.array(cost)
        sv_accept = np.array(sv_accept)

        # save results to file
        scan_dict[pname] = ScanData(pname, sv_accept, results, cost)

        # fit data to a second order polynomial
        #mask       = cost >= 0
        nll_coeff  = np.polyfit(sv_accept, cost, deg=2)#, w=np.exp(-0.5*(params_init[ix] - scan_points)**2/sigma_fisher**2))
        nll_poly   = np.poly1d(nll_coeff)
        d_nll_poly = np.polyder(nll_poly, m=2)
        sigma_post = 1./np.sqrt(d_nll_poly(0))

        if params_pre[ix] != 0:
            err = sigma_post*100/params_pre[ix]
        else:
            err = sigma_post*100
        tqdm.write(f'{pname} error: {err:.2f}') 

        # output plot
        fig, ax = plt.subplots(1, 1, figsize=(8,8), facecolor='white')

        x = np.linspace(scan_vals[0], scan_vals[-1], 1000)
        ax.plot(sv_accept, cost, 'ko', label='nll scan')
        ax.plot(x, nll_poly(x), 'r--', label='quadratic fit')
        ax.plot(x, (x - params_pre[ix])**2/(2*fit_data._perr_init[ix]**2), 'C0--', alpha=0.5, label='log prior')

        ax.grid()
        ax.legend()
        ax.set_xlabel(parameters.loc[pname].label)
        ax.set_ylabel('NLL')
        ax.text(sv_accept[1], 0.95*np.max(cost), r'$\hat{\theta} = $' + f'{sv_accept[np.argmin(cost)]:.3f}')
        ax.text(sv_accept[1], 0.89*np.max(cost), r'$\sigma_{\theta} = $' + f'{err:.2f}')
        ax.set_ylim(0, None)

        plt.savefig(f'plots/nll_scans/{pname}.png')
        fig.clear()
        plt.close()

    outfile = open('local_data/nll_scan_data.pkl', 'wb')
    pickle.dump(scan_dict, outfile)


