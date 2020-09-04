#!/home/naodell/opt/anaconda3/bin/python

import pickle
import os
from functools import partial
from collections import namedtuple
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    parser.add_argument('-p', '--prescan',
                        help = 'Uses results from previously completed n.p. scan.  Will only produce plots.',
                        default = '',
                        type = str
                        )

    args = parser.parse_args()
    ##########################

    processes  = ['ttbar', 't', 'ww', 'wjets', 'zjets_alt', 'diboson', 'fakes']
    selections = [
                  'ee', 'mumu',
                  'emu',
                  'mutau', 'etau',
                  #'mujet', 'ejet'
                 ]
    plot_labels = fh.fancy_labels
    pt.set_default_style()
    timestamp = pt.get_current_time()
    pt.make_directory(f'local_data/nll_scans/{timestamp}')
    pt.make_directory(f'plots/nll_scans/{timestamp}')
    
    # initialize fit data 
    if os.path.isdir(args.input):
        fit_data = fh.FitData(args.input, selections, processes, process_cut=0.05)
    else:
        infile = open(args.input, 'rb')
        fit_data = pickle.load(infile)
        fit_data._initialize_fit_tensor(0.05)  # won't need this at some point
        infile.close()

        if args.prescan != '':
            scan_file = open(args.prescan, 'rb')
            scan_dict = pickle.load(scan_file)
            scan_file.close()

    parameters  = fit_data._parameters.copy()
    params_pre  = parameters['val_init'].values.copy()

    # minimizer options
    min_options = dict(#eps=1e-9, 
                       #xtol=1e-3, 
                       #ftol=1e-9, 
                       #stepmx=0.1, 
                       #maxCGit=50, 
                       #accuracy=1e-10,
                       gtol=1e-3,
                       disp=None
                      )

    # configure the objective
    asimov_data = {cat:fit_data.mixture_model(parameters.val_init.values, cat) for cat in fit_data._model_data.keys()}
    mask = fit_data._pmask
    sample = None
    fobj = partial(fit_data.objective,
                   data       = sample,
                   do_bb_lite = True,
                   lu_test    = None
                  )

    fobj_jac = partial(fit_data.objective_jacobian,
                       data       = sample,
                       do_bb_lite = True,
                       lu_test    = None
                      )

    # prepare scan data
    if args.prescan == '':
        scan_dict = dict()

    n_masked = 0
    for ix, (pname, pdata) in tqdm(enumerate(parameters.iterrows()), total=parameters.shape[0]):

        if pdata.active == 0 or ('jes' not in pname and 'escale' not in pname):
            if pdata.active == 0:
                n_masked += 1
            continue
                            
        if args.prescan is '':

            mask[ix] = False

            # carry out finer binned scan near the MLE value
            scan_vals_central = np.linspace(pdata.val_fit - pdata.err_fit, pdata.val_fit + pdata.err_fit, 5)
            scan_vals_down    = np.linspace(pdata.val_fit - 4*pdata.err_fit, pdata.val_fit - 2*pdata.err_fit, 3)
            scan_vals_up      = np.linspace(pdata.val_fit + 2*pdata.err_fit, pdata.val_fit + 4*pdata.err_fit, 3)
            scan_vals         = np.concatenate([scan_vals_down, scan_vals_central, scan_vals_up])

            # carry out scan and save results
            results   = []
            cost      = []
            sv_accept = []
            for sv in tqdm(scan_vals, 
                           desc = 'scanning profiled nuisance parameters',
                           leave=False
                           ):

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
                
                tqdm.write(f'{pname} = {sv:.3f}: {fobj(pinit):.2f}, {result.fun:.2f}')
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

            # subtract off minimum of cost
            cost -= cost.min()

            # save results to file
            scan_dict[pname] = ScanData(pname, sv_accept, results, cost)

        else:
            # unpack scan dict
            cost      = scan_dict[pname].costs
            sv_accept = scan_dict[pname].scan_points
            results   = scan_dict[pname].results

        cost_nll, cost_np, cost_constraint, cost_bb = [], [], [], []
        for sv, r in zip(sv_accept, results):

            params       = parameters['val_init'].values.copy()
            mask[ix]     = False
            params[ix]   = sv
            params[mask] = r
            mask[ix]     = True

            cost_full = fobj(params[mask])
            bb_penalty = 0
            for k, v in fit_data._bb_penalty.items():
                bb_penalty += v.sum()

            cost_bb.append(bb_penalty)
            cost_nll.append(cost_full - fit_data._np_cost.sum() - bb_penalty)
            cost_np.append(fit_data._np_cost[ix-n_masked])
            cost_constraint.append(fit_data._np_cost.sum() - cost_np[-1])

            tqdm.write(f'{cost_nll[-1]}, {cost_constraint[-1]}, {cost_np[-1]}, {cost_bb[-1]}')

        cost_nll, cost_np = np.array(cost_nll), np.array(cost_np)
        cost_constraint, cost_bb = np.array(cost_constraint), np.array(cost_bb) 
        cost_nll        -= cost_nll.min()
        cost_constraint -= cost_constraint.min()
        cost_bb         -= cost_bb.min()

        # fit profile scan data to a second order polynomial
        cmask      = cost < 20
        x_fit      = np.linspace(sv_accept.min(), sv_accept.max(), 1000)
        nll_coeff  = np.polyfit(sv_accept[cmask], cost[cmask], deg=3)
        nll_poly   = np.poly1d(nll_coeff)
        d_nll_poly = np.polyder(nll_poly, m=2)
        sigma_post = 1./np.sqrt(d_nll_poly(0))

        # get non-profiled values of the cost
        params = parameters['val_fit'].copy()
        cost_noprofile = []
        for sv in x_fit:
            params[pname] = sv
            cost_obj = fobj(params[fit_data._pmask])
            cost_noprofile.append(cost_obj)

        cost_noprofile = np.array(cost_noprofile)
        cost_noprofile -= cost_noprofile.min()

        # fit profile scan data to a second order polynomial
        cmask           = cost_noprofile < 20
        nll_coeff      = np.polyfit(x_fit[cmask], cost_noprofile[cmask], deg=3)
        nll_poly_nopro = np.poly1d(nll_coeff)
        d_nll_poly     = np.polyder(nll_poly_nopro, m=2)
        sigma_nopro    = 1./np.sqrt(d_nll_poly(0))

        # output plot
        fig, ax = plt.subplots(1, 1, figsize=(8,8), facecolor='white')

        ax.plot(sv_accept, cost, 'ko', label='scan points')
        ax.plot(sv_accept, cost_nll, 'C3o', label=r'$NLL(\theta)$')
        ax.plot(sv_accept, cost_bb, 'C2o', label='MC stat.')
        ax.plot(sv_accept, cost_np, 'C0o', label=r'$\pi(\theta_{i})$')
        ax.plot(sv_accept, cost_constraint, 'C4o', label=r'$\sum\pi(\theta_{j\neq i})$')

        ax.plot(x_fit, nll_poly(x_fit), 'k--', label='polynomial fit')
        #ax.plot(x_fit, cost_noprofile, 'C4', label='nll (no profiling)')

        x = np.linspace(pdata.val_init - 5*pdata.err_init, pdata.val_init + 5*pdata.err_init, 1000)
        ax.plot(x, (x - pdata.val_init)**2/(2*pdata.err_init**2), 'C0:', alpha=1., label='prefit')
        ax.plot(x, (x - pdata.val_fit)**2/(2*pdata.err_fit**2), 'C1:', alpha=1., label='postfit')

        ax.grid()
        ax.legend(loc='upper right')
        ax.set_title(parameters.loc[pname].label)
        ax.set_ylabel('NLL - min(NLL)')
        ax.set_xlabel(r'$\theta$')
        param_string = '\n'.join((
            r'$\theta_{prefit} = $' + f'{pdata.val_init:.3f}' + r'$\pm$' + f'{pdata.err_init:.3f}',
            r'${\theta}_{postfit} = $'  + f'{pdata.val_fit:.3f}' + r'$\pm$' + f'{pdata.err_fit:.3f}',
            r'$\sigma_{{scan}} = {0:.3f}$'.format(sigma_post),
            r'$\sigma_{{no profile}} = {0:.3f}$'.format(sigma_nopro)
            ))
        ax.text(0.05, 0.75, param_string, transform=ax.transAxes, fontsize=20,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
        ax.set_ylim(0, 10)

        if pname in ['beta_e', 'beta_mu', 'beta_tau', 'beta_h']:
            ax.set_xlim(0.9*sv_accept[0], 1.1*sv_accept[-1])

        plt.savefig(f'plots/nll_scans/{timestamp}/{pname}.png')
        fig.clear()
        plt.close()

    outfile = open('local_data/nll_scans/timestamp.pkl', 'wb')
    pickle.dump(scan_dict, outfile)

