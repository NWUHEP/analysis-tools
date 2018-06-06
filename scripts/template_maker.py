import numpy as np
import pandas as pd
from skhep.modeling import bayesian_blocks

import scripts.plot_tools as pt
import scripts.systematic_tools as st

features = dict()
features['mumu']  = ['lepton1_pt', 'lepton2_pt']
features['ee']    = ['lepton1_pt', 'lepton2_pt']
features['emu']   = ['lepton1_pt', 'trailing_lepton_pt', 'dilepton1_pt_asym']
features['mutau'] = ['lepton1_pt', 'lepton2_pt', 'dilepton1_pt_asym']
features['etau']  = ['lepton1_pt', 'lepton2_pt', 'dilepton1_pt_asym']
features['mu4j']  = ['lepton1_pt']
features['e4j']   = ['lepton1_pt']

if __name__ == '__main__':

    do_bb_binning = True
    output_path   = f'data/templates/bjet_binned_test'
    data_labels  = ['muon', 'electron']
    model_labels = ['wjets', 'zjets', 't', 'ttbar']
    datasets = [d for l in data_labels + model_labels for d in pt.dataset_dict[l]]

    selections = ['ee', 'mumu', 'emu', 'etau', 'mutau', 'e4j', 'mu4j']
    for selection in selections:
        pt.make_directory(f'{output_path}/{selection}')
        ntuple_dir = f'data/flatuples/single_lepton_test/{selection}_2016'

        # category specific parameters
        labels = ['zjets', 'wjets']
        if selection == 'mu4j' or selection == 'mutau':
            dataset_names = datasets + pt.dataset_dict['fakes']
            labels += ['fakes']

        dm = pt.DataManager(input_dir     = ntuple_dir,
                            dataset_names = datasets,
                            selection     = selection,
                            scale         = 35.9e3,
                            cuts          = pt.cuts[selection],
                            features      = features[selection] + ['n_pu', 'n_bjets', 'gen_cat', 'run_number', 'event_number']
                                )
        #print(f'Running over selection {selection}...')
        for i, bcut in enumerate(['n_bjets == 0', 'n_bjets == 1', 'n_bjets >= 2']):
            if selection is not 'emu' and bcut == 'n_bjets == 0': 
                continue


            # this could be added as a method to the data_manager so I don't have to
            # make copies of (possibly very large) dataframes

            # prepare dataframes with signal templates
            # sigal samples are split according the decay of the W bosons
            decay_map     = pd.read_csv('data/decay_map.csv').set_index('id')
            mc_conditions = {decay_map.loc[i, 'decay']: f'gen_cat == {i}' for i in range(1, 22)}
            df_top        = dm.get_dataframes(['ttbar', 't'], concat=True).query(bcut)
            df_model      = {n: df_top.query(c) for n, c in mc_conditions.items()}
            for l in labels:
                df_model[l] = dm.get_dataframe(l).query(bcut)

            # get the data
            if df_top.shape[0] == 0: continue

            # bin the datasets to derive templates
            for feature in features[selection]:

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
                    hist_lut  = dm._lut_features.loc[feature]
                    binning = hist_lut.n_bins
                    bin_range = (hist_lut.xmin, hist_lut.xmax)

                # bin the data
                x = dm.get_dataframe('data').query(bcut)[feature]
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

