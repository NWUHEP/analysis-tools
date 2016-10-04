from __future__ import division

import sys
from timeit import default_timer as timer
from itertools import product

import numpy as np
from numpy.polynomial.legendre import legval
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf
from scipy.stats import chi2, norm, multivariate_normal
from lmfit import Parameter, Parameters

from nllfitter import Model, NLLFitter
import nllfitter.fit_tools as ft
import nllfitter.plot_tools as pt

# global options
np.set_printoptions(precision=3.)

def bg_pdf(x, a):
    '''
    Legendre polynomial background pdf
    '''
    fx = legval(x[0], a[:3])
    fx *= legval(x[1], a[3:])
    return fx

def sig_pdf(x, a, normalize=False):
    '''
    2D Legendre polynomial plus a bivariate Gaussian.
    '''
    bg  = bg_pdf(x, a[6:])
    mvn = multivariate_normal([a[1], a[3]], [[a[2]**2, 0.], [0., a[4]**2]])
    sig = mvn.pdf(zip(x[0], x[1])) 
    fx = (1 - a[0])*bg + a[0]*sig
    return fx

if __name__ == '__main__':

    ### Start the timer
    start = timer()

    ### Configuration
    pt.set_new_tdr()
    ntuple_dir  = 'data/flatuples/mumu_2012'
    lumi        = 19.8e3
    selection   = ('mumu', 'combined')
    period      = 2012
    model       = 'Gaussian'
    output_path = 'plots/fits/{0}_{1}'.format('_'.join(selection), period)

    datasets = ['muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D'] 
    features = ['dilepton_mass', 'dilepton_b_mass', 'dilepton_pt_over_m']
    cuts     = 'lepton1_q != lepton2_q and n_bjets == 1' #and dilepton_pt_over_m > 2'
    
    if selection[1] == '1b1f':
        cuts += 'and (n_fwdjets > 0 and n_jets == 0)'
    elif selection[1] == '1b1c':
        cuts += 'and (n_fwdjets == 0 and n_jets == 1) \
                 and four_body_delta_phi > 2.5 and met_mag < 40'
    elif selection[1] == 'combined':
        cuts += 'and ((n_fwdjets > 0 and n_jets == 0) or \
                (n_fwdjets == 0 and n_jets == 1 and four_body_delta_phi > 2.5 and met_mag < 40))'

    ### Get dataframes with features for each of the datasets ###
    data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                  dataset_names = datasets,
                                  selection     = selection[0],
                                  scale         = lumi,
                                  cuts          = cuts
                                 )
    df_data = data_manager.get_dataframe('data')
    data = df_data[features]
    data = data.values.transpose()
    data_scaled = [ft.scale_data(data[0], xmin=12, xmax=70), 
                   ft.scale_data(data[1], xmin=50, xmax=350)]

    '''
    ### work on this ###
    df_data['test'] = ['1b1f' if x == 1 else '1b1c' for x in df_data.n_fwdjets]
    df_data['test'] = np.where(((df_data['dilepton_mass'] > 24) & (df_data['dilepton_mass'] < 32)), df_data['test']+'_sr', 'sideband')
    g = sns.pairplot(df_data, 
                 vars      = ['dilepton_mass', 'dilepton_b_mass', 'dilepton_pt'],
                 hue       = 'test',
                 palette   = 'husl',
                 kind      = 'scatter',
                 diag_kind = 'hist',
                 markers   = ['o', 's', 'D'],
                 plot_kws  = dict(s=50, linewidth=0.5),
                 diag_kws  = dict(bins=30, histtype='stepfilled', stacked=True, alpha=0.5, linewidth=1),
                 size=3, aspect=2,
                )
    g.savefig('plots/pairplot.png')
    plt.close()
    '''

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(
                       ('a0', 0.5, False, 0.45, 0.55, None),
                       ('a1', 0., True, None, None, None),
                       ('a2', 0., True, None, None, None),
                       ('b0', 0.5, False, 0.45, 0.55, None),
                       ('b1', 0., True, None, None, None),
                       ('b2', 0., True, None, None, None),
                       ('b3', 0., True, None, None, None),
                       ('b4', 0., True, None, None, None),
                       ('b5', 0., True, None, None, None),
                       ('b6', 0., True, None, None, None),
                       #('b7', 0., True, None, None, None),
                       #('b8', 0., True, None, None, None),
                       #('b9', 0., True, None, None, None),
                       #('b10', 0., True, None, None, None),
                      )

    bg_model  = Model(bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model, min_algo='SLSQP', lmult=(20, 0))
    bg_result = bg_fitter.fit(data_scaled, calculate_corr=False)

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    if model == 'Gaussian':
        sig_params.add_many(
                            ('A'       , 0.01  , True , 0.0   , 1.  , None) ,
                            ('mu1'     , -0.4 , True , -0.8  , -0.1 , None) ,
                            ('sigma1'  , 0.02  , True , 0.015 , 0.2 , None) ,
                            ('mu2'    , -0.3 , True , -0.8  , -0.1  , None) ,
                            ('sigma2' , 0.08  , True , 0.001 , 0.1 , None),
                            ('sigma12' , 0.0  , True , -0.2 , 0.2, None),
                           )
        #for n,p in bg_params.iteritems():
        #    p.vary = False
        sig_params += bg_params.copy()
        sig_model  = Model(sig_pdf, sig_params)

   # elif model == 'Voigt':
   #     sig_params.add_many(
   #                         ('A'     , 0.01   , True , 0.0   , 1.    , None),
   #                         ('mu'    , -0.43  , True , -0.8  , 0.8   , None),
   #                         ('gamma' , 0.033  , True , 0.01  , 0.1   , None),
   #                        )
   #     sig_params += bg_params.copy()
   #     sig_model  = Model(ft.sig_pdf_alt, sig_params)

    sig_fitter = NLLFitter(sig_model, lmult=(0,0))
    sig_result = sig_fitter.fit(data_scaled, calculate_corr=False)

    #x0 = np.linspace(-1, 1, 10000)
    #h, b, p = plt.hist(data_scaled[1], bins=45, range=(-1, 1), histtype='step',  normed=True)
    #plt.plot(x0, legval(x0, bg_result.x[3:]))
    #plt.plot(x0, sig_pdf([x0, x0], sig_result.x[]))
    #plt.xlim((-1, 1))
    #plt.ylim((0, 1.3*np.max(h)))

    x = np.linspace(-1, 1, 1000)
    xx = list(product(*[x, x]))
    x1, x2 = zip(*xx)
    fx = bg_pdf([x1, x2], bg_result.x)
    fx = fx.reshape(1000, 1000).transpose()
    z1, z2 = np.linspace(12, 70, 1000), np.linspace(50, 350, 1000)
    plt.pcolormesh(z1, z2, fx, 
                   alpha = 0.75,
                   cmap  = 'viridis',
                   vmin  = 0.,
                   rasterized=True
                  )
    plt.scatter(data[0], data[1], 
                s=30*(1+data[2]/3), 
                cmap = 'viridis',
                #c=50*(1+data[2]/3), 
                c='w',
                alpha=0.75
               )

    fx = sig_pdf([x1, x2], sig_result.x)
    fx = fx.reshape(1000, 1000).transpose()
    plt.contour(z1, z2, fx, 
                levels     = np.linspace(0, 6., 40),
                alpha      = 0.7,
                colors     = 'k',
                #cmap       = 'hot',
                linewidths = 2.5,
               )

    plt.xlabel(r'$\sf m_{\mu\mu}$ [GeV]')
    plt.ylabel(r'$\sf m_{\mu\mu b}$ [GeV]')
    plt.xlim(12, 70)
    plt.ylim(50, 350)

    plt.savefig('plots/fits/test.png')
    plt.savefig('plots/fits/test.pdf')
    plt.close()


    ### Plots!!! ###
    #print 'Making plot of fit results...'
    #ft.fit_plot(data, xlimits, sig_model, bg_model, 
    #            '{0}_{1}'.format(channel, model), path='plots/fits/{0}'.format(period))

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q_max = 2*(bg_model.calc_nll(data_scaled) - sig_model.calc_nll(data_scaled))
    p_value = 0.5*chi2.sf(q_max, 1)
    z_score = -norm.ppf(p_value)
    print 'q       = {0:.3f}'.format(q_max)
    print 'p_local = {0:.3e}'.format(p_value)
    print 'z_local = {0}'.format(z_score)

    '''
    ### Calculate the number of events in around the peak
    f_bg    = lambda x: ft.bg_pdf(x, (sig_result.x[3], sig_result.x[4]))
    xlim    = (sig_result.x[1] - 2*sig_result.x[2], sig_result.x[1] + 2*sig_result.x[2])
    N_b     = (1 - sig_result.x[0])*n_total*integrate.quad(f_bg, xlim[0], xlim[1])[0]
    sig_b   = N_b/np.sqrt(n_total*sig_result.x[0])
    N_s     = n_total*sig_result.x[0]
    sig_s   = np.sqrt(N_s)
	#sig_s   = n_total*comb_sigma[0]

    print 'N_b = {0:.2f} +\- {1:.2f}'.format(N_b, sig_b)
    print 'N_s = {0:.2f} +\- {1:.2f}'.format(N_s, sig_s)
    print ''
    '''

    print ''
    print 'runtime: {0:.2f} ms'.format(1e3*(timer() - start))
