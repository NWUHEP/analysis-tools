import numpy as np
import pandas as pd
from skhep.modeling import bayesian_blocks

import scripts.plot_tools as pt
import scripts.systematic_tools as st

features = dict()
features['mumu']  = ['lepton2_pt']
features['ee']    = ['lepton2_pt']
features['emu']   = ['trailing_lepton_pt']
features['mutau'] = ['lepton2_pt']#, 'dilepton1_pt_asym']
features['etau']  = ['lepton2_pt']#, 'dilepton1_pt_asym']
features['mu4j']  = ['lepton1_pt']
features['e4j']   = ['lepton1_pt']

if __name__ == '__main__':

    do_bb_binning = True
    output_path   = f'data/templates/bjet_binned_test'
    data_labels  = ['muon', 'electron']
    model_labels = ['wjets', 'zjets', 't', 'ttbar']
    datasets = [d for l in data_labels + model_labels for d in pt.dataset_dict[l]]

    #datasets = [
    #            'ttbar_inclusive_isrup', 'ttbar_inclusive_isrdown',
    #            'ttbar_inclusive_fsrup', 'ttbar_inclusive_fsrdown',
    #            'ttbar_inclusive_hdampup', 'ttbar_inclusive_hdampdown',
    #            'ttbar_inclusive_tuneup', 'ttbar_inclusive_tunedown',
    #           ]

    selections = ['ee', 'mumu', 'emu', 'etau', 'mutau', 'e4j', 'mu4j']
    for selection in selections:
        pt.make_directory(f'{output_path}/{selection}')
        ntuple_dir = f'data/flatuples/single_lepton/{selection}_2016'

        # category specific parameters
        labels = ['zjets', 'wjets']
        if selection == 'mu4j' or selection == 'mutau':
            dataset_names = datasets + pt.dataset_dict['fakes']
            labels += ['fakes']
        else:
            dataset_names = datasets

        dm = pt.DataManager(input_dir     = ntuple_dir,
                            dataset_names = dataset_names,
                            selection     = selection,
                            scale         = 35.9e3,
                            cuts          = pt.cuts[selection],
                            )

        dm_syst = pt.DataManager(input_dir     = ntuple_dir,
                                 dataset_names = dataset_names,
                                 selection     = selection,
                                 scale         = 35.9e3,
                                 cuts          = pt.cuts[selection],
                                 )
        #print(f'Running over selection {selection}...')
        for i, bcut in enumerate(['n_bjets == 0', 'n_bjets == 1', 'n_bjets >= 2']):
            if selection in ['mu4j', 'e4j'] and bcut == 'n_bjets == 0': 
                continue

            if selection in ['e4j', 'mu4j']:
                jet_cut = 'n_jets >= 4'
            else:
                jet_cut = 'n_jets >= 2'

            # prepare dataframes with signal templates
            # sigal samples are split according the decay of the W bosons
            decay_map     = pd.read_csv('data/decay_map.csv').set_index('id')
            mc_conditions = {decay_map.loc[i, 'decay']: f'gen_cat == {i}' for i in range(1, 22)}
            df_top        = dm.get_dataframes(['ttbar', 't'], concat=True).query(jet_cut + ' and ' + bcut)
            df_model      = {n: df_top.query(c) for n, c in mc_conditions.items()}
            for l in labels:
                df_model[l] = dm.get_dataframe(l).query(jet_cut + ' and ' + bcut)

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
                    binning   = hist_lut.n_bins
                    bin_range = (hist_lut.xmin, hist_lut.xmax)

                # bin the data
                x = dm.get_dataframe('data').query(jet_cut + ' and ' + bcut)[feature]
                h, b = np.histogram(x,
                                    bins  = binning,
                                    range = bin_range,
                                    )
                #overflow = x[x > b[-1]].size
                #h = np.append(h, overflow)

                ### get signal and background templates
                template_vals = dict(bins=b[:-1], data=h)
                template_vars = dict(bins=b[:-1], data=h)
                for label, df in df_model.items():
                    x = df[feature].values

                    # get the binned data
                    w = df.weight
                    h, _ = np.histogram(x,
                                        bins    = binning,
                                        range   = bin_range,
                                        weights = w
                                        )
                    #overflow = np.sum(w[x > b[-1]])
                    #h = np.append(h, overflow)

                    hvar, _ = np.histogram(x,
                                           bins    = binning,
                                           range   = bin_range,
                                           weights = w**2
                                           )
                    #overflow = np.sum(w[x > b[-1]]**2)
                    #hvar = np.append(hvar, overflow)

                    template_vals[label] = h
                    template_vars[label] = hvar

                df_vals = pd.DataFrame(template_vals)
                df_vals = df_vals.set_index('bins')
                df_vals.to_csv(f'{output_path}/{selection}/{feature}_bin-{i}_val.csv')

                df_vars = pd.DataFrame(template_vars)
                df_vars = df_vars.set_index('bins')
                df_vars.to_csv(f'{output_path}/{selection}/{feature}_bin-{i}_var.csv')

                ### produce morphing templates for shape systematics
                df_sys = pd.DataFrame(dict(bins=binning[:-1]))

                ### jet systematics ###
                df_top_no_jetcut = dm.get_dataframes(['ttbar', 't'], concat=True) # we need to get the dataframe without the jet cuts

                # jes
                df_sys['jes_up'], df_sys['jes_down'] = st.jet_scale(df_top_no_jetcut, feature, binning, 'jes', jet_cut + ' and ' + bcut)

                # jer
                df_sys['jer_up'], df_sys['jer_down'] = st.jet_scale(df_top_no_jetcut, feature, binning, 'jer', jet_cut + ' and ' + bcut)

                # b tag eff
                df_sys['btag_up'], df_sys['btag_down'] = st.jet_scale(df_top_no_jetcut, feature, binning, 'btag', jet_cut + ' and ' + bcut)

                # mistag eff
                df_sys['mistag_up'], df_sys['mistag_down'] = st.jet_scale(df_top_no_jetcut, feature, binning, 'mistag', jet_cut + ' and ' + bcut)

                # pileup
                df_sys['pileup_up'], df_sys['pileup_down'] = st.pileup_morph(df_top, feature, binning)

                ### lepton energy scale ###
                if selection in ['mumu', 'emu', 'mu4j']:
                    scale = 0.01
                    df_sys['mu_es_up'], df_sys['mu_es_down'] = st.les_morph(df_top, feature, binning, scale)

                if selection in ['ee', 'emu', 'e4j']:
                    scale = 0.01
                    df_sys['el_es_up'], df_sys['el_es_down'] = st.les_morph(df_top, feature, binning, scale)

                if selection in ['etau', 'mutau']:
                    scale = 0.01
                    df_sys['tau_es_up'], df_sys['tau_es_down'] = st.les_morph(df_top, feature, binning, scale)

                # theory systematics

                # write the systematics file
                df_sys.to_csv(f'{output_path}/{selection}/{feature}_bin-{i}_syst.csv', index=False)
