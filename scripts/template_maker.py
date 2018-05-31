import numpy as np
import pandas as pd
from skhep.modeling import bayesian_blocks

import scripts.plot_tools as pt
import scripts.systematic_tools as st

if __name__ == '__main__':

    do_bb_binning = True
    output_path   = f'data/templates/bjet_binned_test'
    datasets = [
                'muon_2016B', 'muon_2016C', 'muon_2016D',
                'muon_2016E', 'muon_2016F', 'muon_2016G', 'muon_2016H',
                'electron_2016B', 'electron_2016C', 'electron_2016D',
                'electron_2016E', 'electron_2016F', 'electron_2016G', 'electron_2016H',

                'ttbar_inclusive', 
                't_tw', 'tbar_tw',
                'w1jets', 'w2jets', 'w3jets', 'w4jets',
                'zjets_m-50', 'zjets_m-10to50',
                'z1jets_m-50', 'z1jets_m-10to50',
                'z2jets_m-50', 'z2jets_m-10to50',
                'z3jets_m-50', 'z3jets_m-10to50',
                'z4jets_m-50', 'z4jets_m-10to50',
                ]


    selections = ['ee', 'mumu', 'emu', 'etau', 'mutau', 'e4j', 'mu4j']
    for selection in selections:
        pt.make_directory(f'{output_path}/{selection}')
        ntuple_dir = f'data/flatuples/single_lepton_test/{selection}_2016'
        features   = ['lepton1_pt']
        cuts       = 'lepton1_pt > 25 and abs(lepton1_eta) < 2.4'

        # category specific parameters
        labels = ['zjets', 'wjets']
        if selection == 'mumu':
            features += ['lepton2_pt']
            cuts  += ' and lepton2_pt > 10 \
                      and lepton1_q != lepton2_q \
                      and dilepton1_mass > 12 \
                      and (dilepton1_mass < 80 or dilepton1_mass > 100)'
        if selection == 'ee':
            features += ['lepton2_pt']
            cuts  += ' and lepton1_pt > 30 \
                      and lepton1_q != lepton2_q \
                      and dilepton1_mass > 12 \
                      and (dilepton1_mass < 80 or dilepton1_mass > 100)'
        elif selection == 'emu':
            features += ['trailing_lepton_pt', 'dilepton1_pt_asym']
            cuts  += ' and lepton1_q != lepton2_q \
                      and dilepton1_mass > 12'
        elif selection == 'etau':
            features += ['lepton2_pt', 'dilepton1_pt_asym']
            cuts  += ' and lepton1_pt > 30 \
                      and lepton2_pt > 20 and abs(lepton2_eta) < 2.3 \
                      and lepton1_q != lepton2_q \
                      and dilepton1_mass > 12'
        elif selection == 'mutau':
            features += ['lepton2_pt', 'dilepton1_pt_asym']
            cuts  += ' and lepton1_pt > 30 \
                      and lepton2_pt > 20 and abs(lepton2_eta) < 2.3 \
                      and lepton1_q != lepton2_q \
                      and dilepton1_mass > 12'
        elif selection == 'mu4j':
            labels += ['fakes']
            datasets.append('fakes')

        #print(f'Running over selection {selection}...')
        for i, bcut in enumerate(['n_bjets == 0', 'n_bjets == 1', 'n_bjets >= 2']):
            dm = pt.DataManager(input_dir     = ntuple_dir,
                                dataset_names = datasets,
                                selection     = selection,
                                scale         = 35.9e3,
                                cuts          = cuts + f' and {bcut}',
                                features      = features + ['n_pu', 'n_bjets', 'gen_cat', 'run_number', 'event_number']
                                )

            # this could be added as a method to the data_manager so I don't have to
            # make copies of (possibly very large) dataframes

            # prepare dataframes with signal templates
            # sigal samples are split according the decay of the W bosons
            decay_map     = pd.read_csv('data/decay_map.csv').set_index('id')
            mc_conditions = {decay_map.loc[i, 'decay']: f'gen_cat == {i}' for i in range(1, 22)}
            df_top        = dm.get_dataframes(['ttbar', 't'], concat=True)
            df_model      = {n: df_top.query(c) for n, c in mc_conditions.items()}
            for l in labels:
                df_model[l] = dm.get_dataframe(l)

            # get the data
            if df_top.shape[0] == 0: continue

            # bin the datasets to derive templates
            for feature in features:
                hist_lut  = dm._lut_features.loc[feature]

                ### calculate binning
                x = df_top[feature].values
                if do_bb_binning:
                    print('Calculating Bayesian block binning...')
                    binning = bayesian_blocks(x[:30000], p0=0.001)

                    if feature == 'dilepton1_pt_asym':
                        dxmin, dxmax = 0.01, 1
                    else:
                        dxmin, dxmax = 1, 1e9

                    dx      = np.abs(binning[1:] - binning[:-1])
                    mask    = np.logical_and(dx > dxmin, dx < dxmax)
                    mask    = np.concatenate(([True], mask))

                    if feature != 'dilepton1_pt_asym':
                        binning = np.around(binning[mask])

                    print('Done!')
                    #print(f'bins for {feature}: {binning}')
                    bin_range = None
                else:
                    print('Using user-defined binning...')
                    binning = hist_lut.n_bins
                    bin_range = (hist_lut.xmin, hist_lut.xmax)

                # bin the data
                x = dm.get_dataframe('data')[feature]
                h, b = np.histogram(x,
                                    bins  = binning,
                                    range = bin_range,
                                    )
                overflow = x[x > b[-1]].size
                h = np.append(h, overflow)

                ### get signal and background templates
                template_vals = dict(bins=b, data=h)
                template_vars = dict(bins=b, data=h)
                for label, df in df_model.items():
                    x = df[feature].values

                    # get the binned data
                    w = df.weight
                    h, _ = np.histogram(x,
                                        bins    = binning,
                                        range   = bin_range,
                                        weights = w
                                        )
                    overflow = np.sum(w[x > b[-1]])
                    h = np.append(h, overflow)

                    hvar, _ = np.histogram(x,
                                           bins    = binning,
                                           range   = bin_range,
                                           weights = w**2
                                           )
                    overflow = np.sum(w[x > b[-1]]**2)
                    hvar = np.append(hvar, overflow)

                    template_vals[label] = h
                    template_vars[label] = hvar

                df_vals = pd.DataFrame(template_vals)
                df_vals = df_vals.set_index('bins')
                df_vals.to_csv(f'{output_path}/{selection}/{feature}_bin-{i}_val.csv')

                df_vars = pd.DataFrame(template_vars)
                df_vars = df_vars.set_index('bins')
                df_vars.to_csv(f'{output_path}/{selection}/{feature}_bin-{i}_var.csv')

                ### produce morphing templates for shape systematics

                # pileup
                df_sys                = pd.DataFrame(dict(bins=binning[:-1]))
                pu_up, pu_down        = st.pileup_morph(df_top, feature, binning)
                df_sys['pileup_up']   = pu_up
                df_sys['pileup_down'] = pu_down

                df_sys.to_csv(f'{output_path}/{selection}/{feature}_bin-{i}_sys.csv', index=False)

