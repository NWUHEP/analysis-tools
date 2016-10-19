from __future__ import division

import sys
from timeit import default_timer as timer
from itertools import product

import numpy as np
from numpy.polynomial.legendre import legval
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from scipy.special import erf
from scipy.stats import chi2, norm, gamma, multivariate_normal
from scipy import integrate
from lmfit import Parameter, Parameters

from nllfitter import Model, NLLFitter
import nllfitter.fit_tools as ft
import nllfitter.plot_tools as pt

# global options
np.set_printoptions(precision=3.)

def bg_pdf(x, a):
    '''
    Legendre polynomial times a gamma distribution background pdf
    '''
    z   = ft.scale_data(x[0], xmin=12, xmax=70)
    fx  = legval(z, [0.5, a[0], a[1]])*2/(70 - 12)
    fx *= gamma.pdf(x[1], a=a[2], loc=a[3], scale=a[4])
    return fx

def sig_pdf(x, a, normalize=True):
    '''
    2D Legendre polynomial plus a bivariate Gaussian.
    '''
    bg  = bg_pdf(x, a[5:])

    #mvn = multivariate_normal([a[1], a[3]], [[a[2]**2, 0.], [0., a[4]**2]])
    #sig = mvn.pdf(zip(x[0], x[1])) 

    sig_fx = lambda z: ft.voigt(z, [a[1], a[2], 0.45])
    sig_fy = lambda z: norm.pdf(z, a[3], a[4])
    #sig_y = ft.voigt(x[1], [a[3], a[4], 4.]) 
    sig = sig_fx(x[0])*sig_fy(x[1])

    if normalize:
        # this would be needed if the two signal kernels are somehow
        # correlated.  It is extremely slow so to use it in a fit would
        # probably require a more efficient implementation
        #integrand = lambda z2, z1: sig_fx(z1)*sig_fy(z2)
        #sig_norm = integrate.dblquad(integrand, 12, 70, lambda lb:50, lambda ub:350)[0]

        sig_norm = 1
        sig_norm *= integrate.quad(sig_fx, 12, 70)[0]
        sig_norm *= integrate.quad(sig_fy, 50, 350)[0]
    else:
        sig_norm = 1.

    fx = (1 - a[0])*bg + a[0]*sig/sig_norm
    return fx

if __name__ == '__main__':

    ### Start the timer
    start = timer()

    ### Configuration
    pt.set_new_tdr()
    ntuple_dir  = 'data/flatuples/mumu_2012'
    selection   = ('mumu', 'combined')
    period      = 2012
    model       = 'Gaussian'
    output_path = 'plots/fits/{0}_{1}'.format(selection[0], period)
    ext         = 'png'

    datasets    = ['muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D']
    features    = ['dilepton_mass', 'dilepton_b_mass', 'dilepton_pt_over_m']
    
    ### Define the selction criteria
    cuts        = '(\
                    lepton1_pt > 25 and abs(lepton1_eta) < 2.1\
                    and lepton2_pt > 25 and abs(lepton2_eta) < 2.1\
                    and lepton1_q != lepton2_q\
                    and n_bjets == 1\
                    and 12 < dilepton_mass < 70\
                    and 50 < dilepton_b_mass < 350\
                   )'

    if selection[1] == '1b1f':
        cuts += ' and n_fwdjets > 0 and n_jets == 0'
    elif selection[1] == '1b1c':
        cuts += ' and n_fwdjets == 0 and n_jets == 1 \
                  and four_body_delta_phi > 2.5 and met_mag < 40'
    elif selection[1] == 'combined':
        cuts += ' and ((n_fwdjets > 0 and n_jets == 0) or \
                  (n_fwdjets == 0 and n_jets == 1 and four_body_delta_phi > 2.5 and met_mag < 40))'
    pt.make_directory(output_path, clear=False)
    ### Get dataframes with features for each of the datasets ###
    data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                  dataset_names = datasets,
                                  period        = 2012,
                                  selection     = selection[0],
                                  cuts          = cuts
                                 )
    df_data = data_manager.get_dataframe('data')
    data = df_data[features]
    data = data.values.transpose()

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(
                       ('a1'    , 0.  , True , None , None , None),
                       ('a2'    , 0.  , True , None , None , None),
                       ('a'     , 2.  , True , 1.   , None , None),
                       ('loc'   , 50. , False , 20   , 60  , None),
                       ('scale' , 30. , True , 0    , None , None),
                      )

    bg_model  = Model(bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model, min_algo='SLSQP')
    bg_result = bg_fitter.fit(data, calculate_corr=True)

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    sig_params.add_many(
                        ('A'     , 0.01 , True , 0.0  , 1.   , None),
                        ('mu1'   , 30.  , True , 20.  , 40.  , None),
                        ('sigma' , 1.   , True , 0.45 , 2.5  , None),
                        ('mu2'   , 150. , True , 120. , 180. , None),
                        ('gamma' , 5.   , True , 2.   , 20.  , None),
                       )
    #for n,p in bg_params.iteritems():
    #    p.vary = False
    sig_params += bg_params.copy()
    sig_model  = Model(sig_pdf, sig_params)
    sig_fitter = NLLFitter(sig_model)
    sig_result = sig_fitter.fit(data, calculate_corr=True)

    #h, b, p = plt.hist(data[0], bins=29, range=(12, 70), histtype='step',  normed=True)
    #x1 = np.linspace(12, 70, 10000)
    #y1 = bg_pdf([x1,x2], bg_result.x)
    #y2 = sig_pdf([x], sig_result.x)
    #plt.plot(x1, y1, 'r-')
    #plt.plot(x2, y1, 'b--')
    #plt.xlim((12, 70))
    #plt.ylim((0, 1.3*np.max(h)))

    ### Plots!!! ###
    z1, z2 = np.linspace(12, 70, 1000), np.linspace(50, 350, 1000)
    x  = np.array(list(product(*[z1, z2]))).transpose()
    fx = bg_pdf(x, bg_result.x)
    fx = fx.reshape(1000, 1000).transpose()
    plt.pcolormesh(z1, z2, fx, 
                   alpha = 0.75,
                   cmap  = 'viridis',
                   vmin  = 0.,
                   rasterized=True
                  )
    #cbar = plt.colorbar()
    #cbar.set_label(r'probability')

    plt.scatter(data[0], data[1], 
                s=30*(1+data[2]/3), 
                cmap = 'viridis',
                #c=50*(1+data[2]/3), 
                c='k',
                alpha=0.7
               )

    fx = sig_pdf(x, sig_result.x)
    fx = fx.reshape(1000, 1000).transpose()
    plt.contour(z1, z2, fx, 
                levels     = np.linspace(0, 0.0006, 15),
                alpha      = 0.7,
                colors     = 'w',
                #cmap       = 'hot',
                linewidths = 3.,
               )
    plt.xlabel(r'$\sf m_{\mu\mu}$ [GeV]')
    plt.ylabel(r'$\sf m_{\mu\mu b}$ [GeV]')
    plt.xlim(12, 70)
    plt.ylim(50, 350)

    plt.savefig('{0}/mumub_mumu_{1}.{2}'.format(output_path, selection[1], ext))
    plt.close()

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(a1, a2, fx.transpose(), rstride=4, cstride=4, alpha=0.2, cmap='coolwarm')
    #ax.contour(a1, a2, fx.transpose(), zdir='z', offset=0, cmap='coolwarm')
    #ax.contour(a1, a2, fx.transpose(), zdir='y', offset=350, cmap='coolwarm')
    #ax.contour(a1, a2, fx.transpose(), zdir='x', offset=30, cmap='coolwarm')
    #plt.show()

    ### Calculate the likelihood ration between the background and signal model
    ### given the data and optimized parameters
    q_max = 2*(bg_model.calc_nll(data) - sig_model.calc_nll(data))
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
