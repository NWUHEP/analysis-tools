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
    bg_fx = lambda z: legval(ft.scale_data(z, xmin=12, xmax=70), [0.5, a[0], a[1]])*2/(70 - 12)
    bg_fy = lambda z: gamma.pdf(z, a=a[2], loc=a[3], scale=a[4])
    bg = bg_fx(x[0])*bg_fy(x[1])
    return bg

def sig_pdf(x, a, normalize=True):
    '''
    2D Legendre polynomial plus a bivariate Gaussian.
    '''
    bg  = bg_pdf(x, a[5:])

    sig_fx = lambda z: ft.voigt(z, [a[1], a[2], 0.45])
    sig_fy = lambda z: norm.pdf(z, a[3], a[4])
    sig = sig_fx(x[0])*sig_fy(x[1])

    if normalize:
        sig_norm = 1
        sig_norm *= integrate.quad(sig_fx, 12, 70)[0]
        #sig_norm *= integrate.quad(sig_fy, 50, 500)[0]
    else:
        sig_norm = 1.

    return (1 - a[0])*bg + a[0]*sig/sig_norm

def fit_plot_profile(data, x, y_bg1, y_bg2, y_sig, 
                     bins, xlim, suffix, 
                     path='plots'
                    ):

    binning = (xlim[1] - xlim[0])/bins

    # Get histogram of data points
    hist, bins = np.histogram(data, bins=bins, range=xlim)
    bins    = (bins[1:] + bins[:-1])/2.
    binerrs = np.sqrt(hist) 
    hist, bins, binerrs = hist[hist>0], bins[hist>0], binerrs[hist>0]
    
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(x , y_sig , 'b-'  , linewidth=2.5)
    ax.plot(x , y_bg1 , 'b--' , linewidth=2.5)
    ax.plot(x , y_bg2 , 'r-.' , linewidth=2.5)
    ax.errorbar(bins, hist, 
                yerr       = binerrs,
                fmt        = 'ko',
                capsize    = 0,
                elinewidth = 2,
                markersize = 9
               )
    ax.legend(['BG+Sig.', 'BG', 'BG only', 'Data']) 

    if suffix == 'mumu':
        ax.set_xlabel(r'$\sf m_{\mu\mu}$ [GeV]')
    elif suffix == 'mumub':
        ax.set_xlabel(r'$\sf m_{\mu\mu b}$ [GeV]')

    ax.set_ylim([0., 1.6*np.max(hist)])
    ax.set_ylabel('Entries / {0} GeV'.format(int(binning)))
    ax.set_xlim(xlim)
    ax.grid()

    ### Add lumi text ###
    ax.text(0.06, 0.9, r'$\bf CMS$', fontsize=30, transform=ax.transAxes)
    ax.text(0.17, 0.9, r'$\it Preliminary $', fontsize=20, transform=ax.transAxes)
    ax.text(0.68, 1.01, r'$\sf{19.7\,fb^{-1}}\,(\sqrt{\it{s}}=8\,\sf{TeV})$', fontsize=20, transform=ax.transAxes)


    fig.savefig(path)
    plt.close()

if __name__ == '__main__':

    ### Start the timer
    start = timer()

    ### Configuration
    pt.set_new_tdr()
    use_official = False
    use_data     = False
    ntuple_dir  = 'data/flatuples/mumu_2012'
    selection   = ('mumu', 'combined')
    period      = 2012
    model       = 'Gaussian'
    output_path = 'plots/fits/{0}_{1}'.format(selection[0], period)
    ext         = 'pdf'

    features    = ['dilepton_mass', 'dilepton_b_mass']#, 'dilepton_pt_over_m']
    
    ### Define the selction criteria
    cuts        = '(\
                    lepton1_pt > 25 and abs(lepton1_eta) < 2.1\
                    and lepton2_pt > 25 and abs(lepton2_eta) < 2.1\
                    and lepton1_q != lepton2_q\
                    and n_bjets == 1\
                    and 12 < dilepton_mass < 70\
                   )'

    if selection[1] == '1b1f':
        cuts += ' and n_fwdjets > 0 and n_jets == 0'
    elif selection[1] == '1b1c':
        cuts += ' and n_fwdjets == 0 and n_jets == 1 \
                  and four_body_delta_phi > 2.5 and met_mag < 40'
    elif selection[1] == 'combined':
        cuts += ' and ((n_fwdjets > 0 and n_jets == 0) \
                  or (n_fwdjets == 0 and n_jets == 1 \
                  and four_body_delta_phi > 2.5 and met_mag < 40))'
    pt.make_directory(output_path, clear=False)
    ### Get dataframes with features for each of the datasets ###
    if use_official:
        if selection[1] == 'combined':
            df_1b1f = pd.read_csv('data/fit/events_1b1f_olga.txt')
            df_1b1c = pd.read_csv('data/fit/events_1b1c_olga.txt')
            df_data = df_1b1f.append(df_1b1c)
        else:
            df_data = pd.read_csv('data/fit/events_{0}_olga.txt'.format(selection[1]))
        data = df_data[features]
        data = data.values.transpose()
    elif use_data:
        datasets    = ['muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D']
        data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                      dataset_names = datasets,
                                      period        = 2012,
                                      selection     = selection[0],
                                      cuts          = cuts
                                     )
        df_data = data_manager.get_dataframe('data')
        data = df_data[features]
        data = data.values.transpose()
    else:
        datasets     = ['ttbar_lep', 'ttbar_lep', 'zjets_m-50', 'zjets_m-10to50', 'bprime_xb']
        data_manager = pt.DataManager(input_dir     = ntuple_dir,
                                      dataset_names = datasets,
                                      period        = 2012,
                                      selection     = selection[0],
                                      cuts          = cuts
                                     )
        df_ttbar  = data_manager.get_dataframe('ttbar')[features][:720]
        df_bprime = data_manager.get_dataframe('bprime_xb')[features][:40]
        df_bprime['dilepton_mass'] = ft.generator(lambda x: ft.voigt(x, [29, 1.9, 0.45]), (24,33), 200)[:40]

        data = df_ttbar.append(df_bprime).values
        data = data.transpose()

    ### Define bg model and carry out fit ###
    bg_params = Parameters()
    bg_params.add_many(
                       ('a1'    , 0.  , True , None , None , None),
                       ('a2'    , 0.  , True , None , None , None),
                       ('k'     , 2.  , True , 1.   , None , None),
                       ('x0'    , 50. , False , 20   , 60  , None),
                       ('theta' , 30. , True , 0    , None , None),
                      )

    bg_model  = Model(bg_pdf, bg_params)
    bg_fitter = NLLFitter(bg_model, min_algo='SLSQP')
    bg_result = bg_fitter.fit(data, calculate_corr=True)

    ### Define bg+sig model and carry out fit ###
    sig_params = Parameters()
    sig_params.add_many(
                        ('A'     , 0.01 , True , 0.0  , 1.   , None),
                        ('mu1'   , 30.  , True , 20.  , 40.  , None),
                        ('gamma' , 1.   , True , 0.45 , 3.   , None),
                        ('mu2'   , 150. , True , 120. , 180. , None),
                        ('sigma' , 5.   , True , 1.   , 25.  , None),
                       )
    #for n,p in bg_params.iteritems():
    #    p.vary = False
    sig_params += bg_params.copy()
    sig_model  = Model(sig_pdf, sig_params)
    sig_fitter = NLLFitter(sig_model)
    sig_result = sig_fitter.fit(data, calculate_corr=True)

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

    ### Makes an overlay of the data, bg, and sig+bg models in 2D ###
    x1, x2 = np.linspace(12, 70, 1000), np.linspace(50, 350, 1000)
    x = np.meshgrid(x1, x2)
    fx = bg_pdf(x, bg_result.x)
    plt.pcolormesh(x1, x2, fx, 
                   alpha = 0.75,
                   cmap  = 'viridis',
                   vmin  = 0.,
                   rasterized=True
                  )
    #cbar = plt.colorbar()
    #cbar.set_label(r'probability')
    data_probabilities = sig_pdf(data, sig_result.x),
    data_scale = data_probabilities/np.max(data_probabilities)
    plt.scatter(data[0], data[1], 
                cmap  = 'plasma',
                #c     = data_scale,
                s     = 60, #200*data_scale,
                c    = 'k',
                alpha = 0.7
               )

    fx = sig_pdf(x, sig_result.x)
    plt.contour(x1, x2, fx, 
                levels     = np.linspace(0, 0.0006, 15),
                alpha      = 0.6,
                colors     = 'w',
                #cmap       = 'hot',
                linewidths = 3.,
               )
    plt.xlabel(r'$\sf m_{\mu\mu}$ [GeV]')
    plt.ylabel(r'$\sf m_{\mu\mu b}$ [GeV]')
    plt.xlim(12, 70)
    plt.ylim(50, 350)

    plt.savefig('{0}/mumub_mumu_{1}_2D.{2}'.format(output_path, selection[1], ext))
    plt.close()

    ### 3D surface plot of bg+signal fit ###
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x[0], x[1], fx, rstride=20, cstride=20, alpha=0.9, cmap='Blues')
    ax.contour(x[0], x[1], fx, zdir='z', offset=0, cmap='viridis')
    #ax.contour(x[0], x[1], fx, zdir='y', offset=159, cmap='coolwarm')
    #ax.contour(x[0], x[1], fx, zdir='x', offset=70, cmap='coolwarm')
    ax.set_xlabel(r'$\sf M_{\mu\mu}$')
    ax.set_ylabel(r'$\sf M_{\mu\mu b}$')

    plt.savefig('{0}/mumub_mumu_{1}_3D.{2}'.format(output_path, selection[1], ext))
    plt.close()

    ### Conditional distributions and projections ###
    mu1            = sig_params['mu1'].value
    mu2            = sig_params['mu2'].value
    gam            = sig_params['gamma'].value
    sig            = sig_params['sigma'].value
    mask_mumu_sig  = (data[0] < mu1-gam) | (data[0] > mu1+gam)
    mask_mumub_sig = (data[1] < mu2-2*sig) | (data[1] > mu2+2*sig)

    ### mumub signal in mumu
    f_sig = lambda xint, z: sig_pdf([z, xint], sig_result.x)
    f_sbg = lambda xint, z: bg_pdf([z, xint], sig_result.x[5:])
    f_bg  = lambda xint, z: bg_pdf([z, xint], bg_result.x)
    x     = np.linspace(12, 70, 116)

    y_sig1 = np.array([integrate.quad(f_sig, 0, mu2-2*sig, args = (xx))[0] for xx in x])
    y_sbg1 = np.array([integrate.quad(f_sbg, 0, mu2-2*sig, args = (xx))[0] for xx in x])
    y_bg1  = np.array([integrate.quad(f_bg, 0, mu2-2*sig, args = (xx))[0] for xx in x])

    y_sig2 = np.array([integrate.quad(f_sig, mu2-2*sig, mu2+2*sig, args = (xx))[0] for xx in x])
    y_sbg2 = np.array([integrate.quad(f_sbg, mu2-2*sig, mu2+2*sig, args = (xx))[0] for xx in x])
    y_bg2  = np.array([integrate.quad(f_bg, mu2-2*sig, mu2+2*sig, args = (xx))[0] for xx in x])

    y_sig3 = np.array([integrate.quad(f_sig, mu2+2*sig, 1000, args = (xx))[0] for xx in x])
    y_sbg3 = np.array([integrate.quad(f_sbg, mu2+2*sig, 1000, args = (xx))[0] for xx in x])
    y_bg3  = np.array([integrate.quad(f_bg, mu2+2*sig, 1000, args = (xx))[0] for xx in x])
    
    data_masked = data[0][mask_mumub_sig==False]
    scale = 2*data_masked.size/(0.5*y_bg2.sum())
    y_bg = scale*y_bg2

    scale = 2*data_masked.size/(0.5*y_sig2.sum())
    y_sbg = scale*(1 - sig_result.x[0])*y_sbg2
    y_sig = scale*y_sig2

    fit_plot_profile(data_masked, x, y_sbg, y_bg, y_sig, 
                     bins=29, 
                     xlim=(12, 70), 
                     suffix='mumu',
                     path=output_path+'/mumu_fit_profile_signal.pdf'
                    )

    data_masked = data[0][mask_mumub_sig]
    scale = 2*data_masked.size/(0.5*np.sum(y_bg1 + y_bg3))
    y_bg = scale*(y_bg1 + y_bg3)

    scale = 2*data_masked.size/(0.5*np.sum(y_sig1 + y_sig3))
    y_sbg = scale*(1 - sig_result.x[0])*(y_sbg1 + y_sbg3)
    y_sig = scale*(y_sig1+y_sig3)

    fit_plot_profile(data_masked, x, y_sbg, y_bg, y_sig, 
                     bins=29, 
                     xlim=(12, 70), 
                     suffix='mumu',
                     path=output_path+'/mumu_fit_profile_sideband.pdf'
                    )

    ### mumu signal in mumub
    f_sig = lambda xint, z: sig_pdf([xint, z], sig_result.x)
    f_sbg = lambda xint, z: bg_pdf([xint, z], sig_result.x[5:])
    f_bg  = lambda xint, z: bg_pdf([xint, z], bg_result.x)
    x     = np.linspace(0, 400, 400)

    y_sig1 = np.array([integrate.quad(f_sig, 12, mu1-gam, args = (xx))[0] for xx in x])
    y_sbg1 = np.array([integrate.quad(f_sbg, 12, mu1-gam, args = (xx))[0] for xx in x])
    y_bg1  = np.array([integrate.quad(f_bg, 12, mu1-gam, args = (xx))[0] for xx in x])

    y_sig2 = np.array([integrate.quad(f_sig, mu1-gam, mu1+gam, args = (xx))[0] for xx in x])
    y_sbg2 = np.array([integrate.quad(f_sbg, mu1-gam, mu1+gam, args = (xx))[0] for xx in x])
    y_bg2  = np.array([integrate.quad(f_bg, mu1-gam, mu1+gam, args = (xx))[0] for xx in x])

    y_sig3 = np.array([integrate.quad(f_sig, mu1+gam, 70, args = (xx))[0] for xx in x])
    y_sbg3 = np.array([integrate.quad(f_sbg, mu1+gam, 70, args = (xx))[0] for xx in x])
    y_bg3  = np.array([integrate.quad(f_bg, mu1+gam, 70, args = (xx))[0] for xx in x])
    
    data_masked = data[1][mask_mumu_sig==False]
    scale = 10*data_masked.size/y_bg2.sum()
    y_bg = scale*y_bg2

    scale = 10*data_masked.size/y_sig2.sum()
    y_sbg = scale*(1 - sig_result.x[0])*y_sbg2
    y_sig = scale*y_sig2

    fit_plot_profile(data_masked, x, y_sbg, y_bg, y_sig, 
                     bins=40, 
                     xlim=(0, 400), 
                     suffix='mumub',
                     path=output_path+'/mumub_fit_profile_signal.pdf'
                    )

    data_masked = data[1][mask_mumu_sig]
    scale = 10*data_masked.size/np.sum(y_bg1 + y_bg3)
    y_bg = scale*(y_bg1 + y_bg3)

    scale = 10*data_masked.size/np.sum(y_sig1 + y_sig3)
    y_sbg = scale*(1 - sig_result.x[0])*(y_sbg1 + y_sbg3)
    y_sig = scale*(y_sig1+y_sig3)

    fit_plot_profile(data_masked, x, y_sbg, y_bg, y_sig, 
                     bins=40, 
                     xlim=(0, 400), 
                     suffix='mumub',
                     path=output_path+'/mumub_fit_profile_sideband.pdf'
                    )
    print ''
    print 'runtime: {0:.2f} s'.format((timer() - start))
