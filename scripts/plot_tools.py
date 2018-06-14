'''
    Tools for creating HEP style plots with bacon pickles :)
'''

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import beta

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from tqdm import tqdm
tqdm.monitor_interval = 0

dataset_dict = dict(
                    muon     = ['muon_2016B', 'muon_2016C', 'muon_2016D', 
                                'muon_2016E', 'muon_2016F', 'muon_2016G', 'muon_2016H'],
                    electron = ['electron_2016B', 'electron_2016C', 'electron_2016D', 
                                'electron_2016E', 'electron_2016F', 'electron_2016G', 
                                'electron_2016H'
                                ],
                    ttbar    = ['ttbar_inclusive'], #'ttbar_lep', 'ttbar_semilep',
                    t        = ['t_tw', 'tbar_tw'], #'t_t', 'tbar_t',
                    wjets    = ['w1jets', 'w2jets', 'w3jets', 'w4jets'],
                    zjets    = ['zjets_m-50',  'zjets_m-10to50', 
                                'z1jets_m-50', 'z1jets_m-10to50', 
                                'z2jets_m-50', 'z2jets_m-10to50', 
                                'z3jets_m-50', 'z3jets_m-10to50', 
                                'z4jets_m-50', 'z4jets_m-10to50'
                                ],
                    qcd      = ['qcd_ht100to200', 'qcd_ht200to300', 'qcd_ht300to500', 
                                'qcd_ht500to1000', 'qcd_ht1000to1500', 'qcd_ht1500to2000', 
                                'qcd_ht2000'
                                ],
                    diboson  = ['ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q'], #'zz_4l',
                    fakes    = ['muon_2016B_fakes', 'muon_2016C_fakes', 'muon_2016D_fakes', 
                                'muon_2016E_fakes', 'muon_2016F_fakes', 'muon_2016G_fakes', 
                                'muon_2016H_fakes'
                                ],
                    )

cuts = dict(
            ee    = 'lepton1_q != lepton2_q and lepton1_pt > 30 and lepton2_pt > 10 \
                     and dilepton1_mass > 12 and (dilepton1_mass > 101 or dilepton1_mass < 81) \
                     and n_bjets >= 1',
            mumu  = 'lepton1_q != lepton2_q and lepton1_pt > 25 and lepton2_pt > 10 \
                     and dilepton1_mass > 12  and (dilepton1_mass > 101 or dilepton1_mass < 81) \
                     and n_bjets >= 1',
            emu   = 'lepton1_q != lepton2_q and lepton1_pt > 10 and lepton2_pt > 10 \
                     and dilepton1_mass > 12',
            etau  = 'lepton1_q != lepton2_q and lepton1_pt > 30 and lepton2_pt > 20 \
                     and dilepton1_mass > 12 \
                     and n_bjets >= 1',
            mutau = 'lepton1_q != lepton2_q and lepton1_pt > 25 and lepton2_pt > 20 \
                     and dilepton1_mass > 12 \
                     and n_bjets >= 1',
            e4j   = 'lepton1_pt > 30 and n_jets >= 4 and n_bjets >= 1',
            mu4j  = 'lepton1_pt > 25 and n_jets >= 4 and n_bjets >= 1',
            )

def make_directory(file_path, clear=True):
    if not os.path.exists(file_path):
        os.system('mkdir -p '+file_path)

    if clear and len(os.listdir(file_path)) != 0:
        os.system('rm -r '+file_path+'/*')

def calculate_efficiency(num, den, bins, alpha=0.317):
    '''
    Calculates efficiencies given the provided binning and estimates
    uncertainties using the Clopper-Pearson interval construction. 
    
    Parameters:
    ===========
    num: array for numerator (subset of denominator)
    den: array for denominator
    bins: bin edges for histogram
    alpha: confidence interval will correspond to 1 - alpha
    '''
    n, _ = np.histogram(num, bins=bins)
    d, b = np.histogram(den, bins=bins)
    
    x = (b[1:] + b[:-1])/2.
    x_err = (b[1:] - b[:-1])/2.
    eff = n.astype(float)/d
    eff_err = [np.abs(eff - beta.ppf(alpha/2, n, d - n + 1)), 
               np.abs(eff - beta.ppf(1 - alpha/2, n + 1, d - n))]
    
    return x, eff, x_err, eff_err

def hist_to_errorbar(data, nbins, xlim, normed=False):
    '''
    Wrapper for converting a histogram to data for drawing markers with errorbars.

    Parameters:
    ===========
    data: data to be histogrammed
    nbins: number of bins
    xlim: tuple specifying minimum and maximum values of x axis
    '''
    y, bins = np.histogram(data, bins=int(nbins), range=xlim)
    x = (bins[1:] + bins[:-1])/2.
    yerr = np.sqrt(y)

    return x, y, yerr


def ratio_errors(num, sig_num, den, sig_den):
    '''
    Error of ratio assuming numerator and denominator are uncorrelated.

    Parameters:
    ===========
    num : numerator
    num : error on the numerator
    den : denominator
    den : error on the denominator
    '''
    ratio = num/den
    error = ratio*np.sqrt(sig_num**2/num**2 + sig_den**2/den**2)
    return error 


def poisson_errors(bin_content, suppress_zero=False):
    '''
    Returns a high and low 1-sigma error bar for an input bin value, as defined
    in: https://www-cdf.fnal.gov/physics/statistics/notes/pois_eb.txt.

    If bin_content > 9, returns the sqrt(bin_content)
    '''
    error_dict = {
        0: (0.000000, 1.000000),
        1: (0.381966, 2.618034),
        2: (1.000000, 4.000000),
        3: (1.697224, 5.302776),
        4: (2.438447, 6.561553),
        5: (3.208712, 7.791288),
        6: (4.000000, 9.000000),
        7: (4.807418, 10.192582),
        8: (5.627719, 11.372281),
        9: (6.458619, 12.541381)}

    if suppress_zero and bin_content == 0:
        return (0, 0)
    elif bin_content in error_dict:
        return error_dict[bin_content]
    else:
        return (np.sqrt(bin_content), np.sqrt(bin_content))


def get_data_and_weights(dataframes, feature, labels, condition):
    data    = []
    weights = []
    for label in labels:
        if label not in dataframes.keys():
            continue

        if condition == 'None':
            df = dataframes[label]
        else:
            df = dataframes[label].query(condition)
        data.append(df[feature].values)
        weights.append(df['weight'].values)

    return data, weights

def set_new_tdr():
    plt.style.use('default')
    plt.rcParams['font.size']         = 18
    #plt.rcParams['font.family']       = 'serif'
    #plt.rcParams['font.serif']        = 'Ubuntu'
    #plt.rcParams['font.monospace']    = 'Ubuntu Mono'
    #plt.rcParams['mathtext.fontset']  = 'custom'
    #plt.rcParams['mathtext.sf']       = 'Ubuntu'

    plt.rcParams['axes.labelsize']    = 20
    plt.rcParams['xtick.labelsize']   = 18
    plt.rcParams['ytick.labelsize']   = 18
    plt.rcParams['figure.titlesize']  = 20
    plt.rcParams['figure.figsize']    = (10, 10)
    plt.rcParams['legend.fontsize']   = 20
    plt.rcParams['legend.numpoints']  = 1

def add_lumi_text(ax, lumi, period):
    #ax.text(0.04, 0.9, r'$\bf CMS$', fontsize=30, transform=ax.transAxes)
    #ax.text(0.20, 0.9, r'$\it Preliminary $', fontsize=20, transform=ax.transAxes)

    if period == 2012:
        ax.text(0.60, 1.01, 
                r'$\sf{19.7\,fb^{-1}}\,(\sqrt{\it{s}}=8\,\sf{TeV})$', 
                fontsize=20, 
                transform=ax.transAxes
                )
    elif period == 2016:
        ax.text(0.60, 1.01, 
                r'$\sf{{ {0:.1f}\,fb^{{-1}}}}\,(\sqrt{{\it{{s}}}}=13\,\sf{{TeV}})$'.format(lumi/1000.), 
                fontsize=20, 
                transform=ax.transAxes
                )

class DataManager():
    def __init__(self, input_dir, dataset_names, selection,
                 period   = 2016,
                 scale    = 1,
                 cuts     = '',
                 combine  = True,
                 features = None
                 ):
        self._input_dir     = input_dir
        self._dataset_names = dataset_names
        self._selection     = selection
        self._period        = period
        self._scale         = scale
        self._cuts          = cuts
        self._combine       = combine
        self._features      = features
        self._load_luts()
        self._load_dataframes()

    def _load_luts(self):
        '''
        Retrieve look-up tables for datasets and variables
        '''
        self._event_counts = pd.read_csv('{0}/event_counts.csv'.format(self._input_dir, self._selection))
        self._lut_datasets = pd.read_excel('data/plotting_lut.xlsx',
                                           sheet_name='datasets_{0}'.format(self._period),
                                           index_col='dataset_name'
                                          ).dropna(how='all')
        lut_features_default = pd.read_excel('data/plotting_lut.xlsx',
                                             sheet_name='variables',
                                             index_col='variable_name'
                                            ).dropna(how='all')
        lut_features_select = pd.read_excel('data/plotting_lut.xlsx',
                                            sheet_name='variables_{0}'.format(self._selection),
                                            index_col='variable_name'
                                           ).dropna(how='all')
        self._lut_features = pd.concat([lut_features_default, lut_features_select])

    def _load_dataframes(self):
        '''
        Get dataframes from input directory.  This method is only for execution
        while initializing the class instance.
        '''
        dataframes = {}
        for dataset in tqdm(self._dataset_names,
                            desc       = 'Loading dataframes',
                            unit_scale = True,
                            ncols      = 75,
                            total      = len(self._dataset_names)
                            ):

            fname = f'{self._input_dir}/ntuple_{dataset}.pkl'
            if not os.path.isfile(fname):
                continue

            df = pd.read_pickle(fname)
            if df.size == 0:
                continue

            ### apply selection cuts ###
            if self._cuts != '':
                df = df.query(self._cuts).copy()

            ### only keep certain features ###
            if self._features is not None:
                df = df[self._features + ['weight']]

            init_count        = self._event_counts[dataset][0]
            lut_entry         = self._lut_datasets.loc[dataset]
            label             = lut_entry.label
            df.loc[:,'label'] = df.shape[0]*[label, ]

            ### update weights with lumi scale factors ###
            if label == 'data':
                df.loc[:, 'weight'] = 1.
            elif label == 'fakes':
                df.loc[:, 'weight'] *= lut_entry.cross_section
            else:
                scale = self._scale
                scale *= lut_entry.cross_section
                scale *= lut_entry.branching_fraction

                if dataset.split('_')[0] == 'zjets_alt':
                    scale *= df.gen_weight
                    neg_count = self._event_counts[dataset][9]
                    scale /= init_count - 2*neg_count
                else:
                    scale /= init_count

                df.loc[:, 'weight'] *= scale

            ### combined datasets if required ###
            if self._combine:
                if label not in dataframes.keys():
                    dataframes[label] = df
                else:
                    dataframes[label] = dataframes[label].append(df)
            else:
                dataframes[dataset] = df
    

        # hack to remove overlapping data; remove when this is fixed upstream :(
        if 'data' in dataframes.keys():
            df = dataframes['data']
            dataframes['data'] = df.drop_duplicates(subset=['run_number', 'event_number'])

        self._dataframes = dataframes

    def get_dataframe(self, dataset_name, condition=''):
        df = self._dataframes[dataset_name]
        if condition != '':
            return df.query(condition)
        else:
            return df

    def get_dataframes(self, dataset_names, concat=False, condition=''):
        dataframes = {}
        for dataset in dataset_names:
            df = self._dataframes[dataset]
            if condition == '':
                dataframes[dataset] = df
            else:
                dataframes[dataset] = df.query(condition)

        if concat:
            df = pd.concat(list(dataframes.values()))
            return df
        else:
            return dataframes

    def get_dataset_names(self):
        return self._dataset_names

    def get_bounds_dict(self):
        df = self._lut_features[['xmin', 'xmax']]
        bdict = df.T.to_dict('list')
        return bdict

    def print_yields(self, dataset_names,
                     exclude    = [],
                     conditions = [''],
                     mc_scale   = True,
                     do_string  = False,
                     fmt        = 'markdown'
                     ):
        '''
        Prints sum of the weights for the provided datasets

        Parameters
        ==========
        dataset_names : list of datasets to print
        exclude       : list of datasets to exclude from sum background calculation
        conditions    : list of conditions to apply
        mc_scale      : scale MC according to weights and scale
        do_string     : format of output cells: if True then string else float
        fmt           : formatting of the table (default:markdown)
        '''

        # print header
        table = OrderedDict()
        dataset_names = [dn for dn in dataset_names if dn in self._dataframes.keys()]
        dataframes = self.get_dataframes(dataset_names)
        for i, condition in enumerate(conditions):
            table[f'condition_{i+1}'] = []
            if not do_string:
                table[f'error_{i+1}'] = []

            bg_total = [0., 0.]
            for dataset in dataset_names:
                df = dataframes[dataset]
                if condition != '' and condition != 'preselection':
                    df = df.query(condition)

                if mc_scale:
                    n     = df.weight.sum()
                    n_err = np.sqrt(np.sum(df.weight**2))
                else:
                    n     = df.shape[0]
                    n_err = np.sqrt(n)

                # calculate sum of bg events
                if dataset not in exclude and dataset != 'data':
                    bg_total[0] += n
                    bg_total[1] += n_err**2

                if do_string:
                    if dataset == 'data':
                        table[f'condition_{i+1}'].append('${0}$'.format(int(n)))
                    else:
                        table[f'condition_{i+1}'].append('${0:.1f} \pm {1:.1f}$'.format(n, n_err))
                else:
                    table[f'condition_{i+1}'].append(n)
                    table[f'error_{i+1}'].append(n_err)

                dataframes[dataset] = df  # update dataframes so cuts are applied sequentially

            if do_string:
                table[f'condition_{i+1}'].append('${0:.1f} \pm {1:.1f}$'.format(bg_total[0], np.sqrt(bg_total[1])))
            else:
                table[f'condition_{i+1}'].append(bg_total[0])
                table[f'error_{i+1}'].append(np.sqrt(bg_total[1]))

        if do_string:
            labels = [self._lut_datasets.loc[d].text for d in dataset_names]
        else:
            labels = dataset_names
        table = pd.DataFrame(table, index=labels+['background'])
        return table


class PlotManager():
    def __init__(self, data_manager, features, stack_labels, 
                 overlay_labels = [],
                 top_overlay    = False,
                 output_path    = 'plots',
                 file_ext       = 'png'
                 ):
        self._dm             = data_manager
        self._features       = features
        self._stack_labels   = [l for l in stack_labels if l in data_manager._dataframes.keys()]
        self._overlay_labels = overlay_labels
        self._top_overlay    = top_overlay
        self._output_path    = output_path
        self._file_ext       = file_ext
        self._initialize_colors()

    def _initialize_colors(self):
        lut = self._dm._lut_datasets
        self._stack_colors   = [lut.loc[l].color for l in self._stack_labels]
        self._overlay_colors = [lut.loc[l].color for l in self._overlay_labels]

    def make_overlays(self, features,
                      plot_data     = True,
                      normed        = False,
                      do_cms_text   = True,
                      overlay_style = 'line',
                      do_ratio      = False
                      ):
        dm = self._dm
        make_directory(self._output_path)

        ### alias dataframes and datasets lut###
        dataframes   = dm._dataframes
        lut_datasets = dm._lut_datasets

        ### initialize legend text ###
        legend_text = []
        legend_text.extend([lut_datasets.loc[label].text for label in self._stack_labels[::-1]])
        #legend_text.extend([lut_datasets.loc[label].text for label in self._overlay_labels[::-1]])

        if len(self._stack_labels) > 0:
            legend_text.append('BG error')
        if plot_data:
            legend_text.append('Data')

        for feature in tqdm(features, 
                            desc='plotting...', 
                            unit_scale=True, 
                            ncols=75, 
                            total=len(features)
                            ):
            if feature not in self._features:
                print('{0} not in features.')
                continue

            ### Get style data for the feature ###
            lut_entry = dm._lut_features.loc[feature]

            ### initialize figure ###
            if do_ratio:
                fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]})
                fig.subplots_adjust(hspace=0)
                ax = axes[0]
            else:
                fig, ax = plt.subplots(1, 1)
            #legend_handles = []

            ### Get stack data and apply mask if necessary ###
            y_min, y_max = 1e9, 0.
            if len(self._stack_labels) > 0:
                stack_data, stack_weights = get_data_and_weights(dataframes, feature, self._stack_labels, lut_entry.condition)
                stack, bins, p = ax.hist(stack_data, 
                                         bins      = int(lut_entry.n_bins),
                                         range     = (lut_entry.xmin, lut_entry.xmax),
                                         color     = self._stack_colors,
                                         alpha     = 1.,
                                         linewidth = 0.5,
                                         stacked   = True,
                                         histtype  = 'stepfilled',
                                         weights   = stack_weights
                                        )

                ### Need to histogram the stack with the square of the weights to get the errors ### 
                stack_noscale = np.histogram(np.concatenate(stack_data),
                                             bins    = int(lut_entry.n_bins),
                                             range   = (lut_entry.xmin, lut_entry.xmax),
                                             weights = np.concatenate(stack_weights)**2
                                            )[0] 
                stack_sum = stack[-1] if len(stack_data) > 1 else stack
                stack_x   = (bins[1:] + bins[:-1])/2.
                stack_err = np.sqrt(stack_noscale)

                if do_ratio:
                    denominator = (stack_x, stack_sum, stack_err)

                no_blanks = stack_sum > 0
                stack_sum, stack_x, stack_err = stack_sum[no_blanks], stack_x[no_blanks], stack_err[no_blanks]
                eb = ax.errorbar(stack_x, stack_sum, yerr=stack_err, 
                                 fmt        = 'none',
                                 ecolor     = 'k',
                                 capsize    = 0,
                                 elinewidth = 10,
                                 alpha      = 0.15
                                )
                if stack_sum.min() < y_min and stack_sum.min() > 0.:
                    y_min = stack_sum.min() 
                if stack_sum.max() > y_max:
                    y_max = stack_sum.max() 
                #legend_handles.append(eb[0])


            ### Get overlay data and apply mask if necessary ###
            if len(self._overlay_labels) > 0:
                overlay_data, overlay_weights = get_data_and_weights(dataframes, feature, self._overlay_labels, lut_entry.condition)
                if overlay_style == 'line':
                    hists, bins, p = ax.hist(overlay_data,
                                             bins      = lut_entry.n_bins,
                                             range     = (lut_entry.xmin, lut_entry.xmax),
                                             color     = self._overlay_colors,
                                             alpha     = 1.,
                                             histtype  = 'step',
                                             linewidth = 2.,
                                             #linestyle = '--',
                                             normed    = normed,
                                             bottom    = 0 if y_max == 0 or not self._top_overlay else stack[-1],
                                             weights   = overlay_weights
                                            )

                    hists = np.array(hists).flatten()
                    if hists.min() < y_min and hists.min() > 0.:
                        y_min = hists.min()
                    if hists.max() > y_max:
                        y_max = hists.max()
                    #legend_handles.append(p)
                elif overlay_style == 'errorbar':
                    x, y, yerr = hist_to_errorbar(overlay_data,
                                                  nbins = lut_entry.n_bins,
                                                  xlim  = (lut_entry.xmin, lut_entry.xmax)
                                                 )
                    if do_ratio:
                        numerator = (x, y, yerr)

                    x, y, yerr = x[y>0], y[y>0], yerr[y > 0]
                    eb = ax.errorbar(x, y, yerr=yerr,
                                  fmt        = 'bo',
                                  capsize    = 0,
                                  elinewidth = 2
                                 )

            ### If there's data to overlay: apply feature condition and get
            ### datapoints plus errors
            if plot_data:
                data, _ = get_data_and_weights(dataframes, feature, ['data'], lut_entry.condition)
                x, y, yerr = hist_to_errorbar(data, 
                                              nbins = lut_entry.n_bins,
                                              xlim  = (lut_entry.xmin, lut_entry.xmax)
                                             )
                if do_ratio:
                    numerator = (x, y, yerr)

                x, y, yerr = x[y>0], y[y>0], yerr[y>0]
                eb = ax.errorbar(x, y, yerr=yerr, 
                              fmt        = 'ko',
                              capsize    = 0,
                              elinewidth = 2
                             )
                if y.size > 0:
                    if y.min() < y_min and y.min() > 0.:
                        y_min = y.min()
                    if y.max() > y_max:
                        y_max = y.max() 
                #legend_handles.append(eb[0])

            ### make the legend ###
            ax.legend(legend_text, loc=1)

            ### labels and x limits ###
            if do_ratio:
                axes[1].set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))
                axes[1].set_ylabel(r'Data/MC')
                axes[1].set_ylim((0.5, 1.49))
                axes[1].grid()

                ### calculate ratios 
                mask = (numerator[1] > 0) & (denominator[1] > 0)
                 
                ratio = numerator[1][mask]/denominator[1][mask]
                error = ratio*np.sqrt(numerator[2][mask]**2/numerator[1][mask]**2 + denominator[2][mask]**2/denominator[1][mask]**2)
                axes[1].errorbar(numerator[0][mask], ratio, yerr=error,
                                 fmt = 'ko',
                                 capsize = 0,
                                 elinewidth = 2
                                )
                axes[1].plot([lut_entry.xmin, lut_entry.xmax], [1., 1.], 'r--')
            else:
                ax.set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))

            ax.set_ylabel(r'$\sf {0}$'.format(lut_entry.y_label))
            ax.set_xlim((lut_entry.xmin, lut_entry.xmax))
            ax.grid()

            ### Add lumi text ###
            if do_cms_text:
                add_lumi_text(ax, dm._scale, dm._period)

            ### Make output directory if it does not exist ###
            make_directory('{0}/linear/{1}'.format(self._output_path, lut_entry.category), False)
            make_directory('{0}/log/{1}'.format(self._output_path, lut_entry.category), False)

            ### Save output plot ###
            ### linear scale ###
            ax.set_ylim((0., 1.6*y_max))
            fig.savefig('{0}/linear/{1}/{2}.{3}'.format(self._output_path, 
                                                        lut_entry.category, 
                                                        feature, 
                                                        self._file_ext
                                                       ))

            ### log scale ###
            ax.set_yscale('log')
            ax.set_ylim(0.05, 15.*y_max)
            fig.savefig('{0}/log/{1}/{2}.{3}'.format(self._output_path,
                                                     lut_entry.category,
                                                     feature,
                                                     self._file_ext
                                                     ))

            fig.clear()
            plt.close()

    def make_sideband_overlays(self, label, cuts, features,
                               do_cms_text = True,
                               do_stacked  = False
                               ):

        ### alias dataframes and datasets lut###
        df_pre = self._dm.get_dataframe(label)
        df_sr  = df_pre.query(cuts[0])
        df_sb  = df_pre.query(cuts[1])
        for feature in tqdm(features, 
                            desc       = 'Plotting',
                            unit_scale = True,
                            ncols      = 75,
                            total      = len(features
                           )):
            if feature not in self._features:
                print('{0} not in features.')
                continue

            fig, ax = plt.subplots(1, 1)
            lut_entry = self._dm._lut_features.loc[feature]
            x_sr = df_sr[feature].values
            x_sb = df_sb[feature].values
            hist, bins, _ = ax.hist([x_sr, x_sb],
                                    bins      = lut_entry.n_bins,
                                    range     = (lut_entry.xmin, lut_entry.xmax),
                                    color     = ['k', 'r'],
                                    alpha     = 0.9,
                                    histtype  = 'step',
                                    linewidth = 2.,
                                    normed    = True,
                                    stacked   = do_stacked
                                   )

            ### make the legend ###
            #legend_text = cuts # Need to do something with this
            legend_text = [r'$\sf M_{\mu\mu}\,\notin\,[24,33]$', r'$\sf M_{\mu\mu}\,\in\,[24, 33]$']
            ax.legend(legend_text)

            ### labels and x limits ###
            ax.set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))
            ax.set_ylabel(r'$\sf {0}$'.format(lut_entry.y_label))
            ax.set_xlim((lut_entry.xmin, lut_entry.xmax))
            ax.grid()

            ### Add lumi text ###
            #if do_cms_text:
            #    add_lumi_text(ax, dm._scale, dm._period)

            ### Make output directory if it does not exist ###
            make_directory('{0}/linear/{1}'.format(self._output_path, lut_entry.category), False)
            make_directory('{0}/log/{1}'.format(self._output_path, lut_entry.category), False)

            ### Save output plot ###
            ### linear scale ###
            y_max = np.max(hist)
            ax.set_ylim((0., 1.8*y_max))
            fig.savefig('{0}/linear/{1}/{2}.{3}'.format(self._output_path, 
                                                        lut_entry.category, 
                                                        feature, 
                                                        self._file_ext
                                                       ))

            ### log scale ###
            ax.set_yscale('log')
            ax.set_ylim((0.1*np.min(hist), 15.*y_max))
            fig.savefig('{0}/log/{1}/{2}.{3}'.format(self._output_path, 
                                                     lut_entry.category, 
                                                     feature, 
                                                     self._file_ext
                                                    ))
            fig.clear()
            plt.close()


    def make_conditional_overlays(self, labels, features, conditions, 
                                  bg_labels   = None, 
                                  legend      = None,
                                  do_data     = False,
                                  do_cms_text = True,
                                  do_stacked  = False
                                 ):

        ### get the samples to be split conditionally ###
        df_combined = self._dm.get_dataframes(labels, concat=True)
        df_split    = [df_combined.query(c) for c in conditions]

        ### get the background data ###
        if bg_labels is not None:
            df_split.append(self._dm.get_dataframes(bg_labels, concat=True))
            df_split = df_split[::-1]

        if do_data:
            df_data = self._dm.get_dataframe('data')

        for feature in tqdm(features, 
                            desc       = 'Plotting',
                            unit_scale = True,
                            ncols      = 75,
                            total      = len(features
                           )):
            if feature not in self._features:
                print('{0} not in features.')
                continue

            fig, ax   = plt.subplots(1, 1)
            lut_entry = self._dm._lut_features.loc[feature]
            data      = [df[feature].values for df in df_split]
            weights   = [df['weight'].values for df in df_split]
            hist, bins, _ = ax.hist(data,
                                    bins      = lut_entry.n_bins,
                                    range     = (lut_entry.xmin, lut_entry.xmax),
                                    #color     = ['k', 'r'],
                                    alpha     = 0.9,
                                    histtype  = 'stepfilled' if do_stacked else 'step',
                                    linewidth = 2.,
                                    weights   = weights,
                                    normed    = not do_stacked,
                                    stacked   = do_stacked
                                   )

            if do_data:
                x, y, yerr = hist_to_errorbar(df_data[feature], 
                                              nbins = lut_entry.n_bins,
                                              xlim  = (lut_entry.xmin, lut_entry.xmax)
                                             )
                x, y, yerr = x[y>0], y[y>0], yerr[y>0]
                eb = ax.errorbar(x, y, yerr=yerr, 
                              fmt        = 'ko',
                              capsize    = 0,
                              elinewidth = 2
                             )


            ### make the legend ###
            #legend_text = cuts # Need to do something with this
            if legend == None:
                legend_text = conditions
            else:
                legend_text = legend
            ax.legend(legend_text[::-1] + ['data'])

            ### labels and x limits ###
            ax.set_xlabel(r'$\sf {0}$'.format(lut_entry.x_label))
            ax.set_ylabel(r'$\sf {0}$'.format(lut_entry.y_label))
            ax.set_xlim((lut_entry.xmin, lut_entry.xmax))
            ax.grid()

            ### Add lumi text ###
            #add_lumi_text(ax)

            ### Make output directory if it does not exist ###
            make_directory('{0}/linear/{1}'.format(self._output_path, lut_entry.category), False)
            make_directory('{0}/log/{1}'.format(self._output_path, lut_entry.category), False)

            plt.tight_layout()
            ### Save output plot ###
            ### linear scale ###
            y_max = np.max(hist)
            ax.set_ylim((0., 1.3*y_max))
            fig.savefig('{0}/linear/{1}/{2}.{3}'.format(self._output_path, 
                                                        lut_entry.category, 
                                                        feature, 
                                                        self._file_ext
                                                       ))

            ### log scale ###
            ax.set_yscale('log')
            ax.set_ylim((0.1*np.min(hist), 10.*y_max))
            fig.savefig('{0}/log/{1}/{2}.{3}'.format(self._output_path, 
                                                     lut_entry.category, 
                                                     feature, 
                                                     self._file_ext
                                                    ))
            fig.clear()
            plt.close()