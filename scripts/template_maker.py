import pickle

import numpy as np
import pandas as pd
from skhep.modeling import bayesian_blocks

import scripts.plot_tools as pt
import scripts.fit_helpers as fh
import scripts.systematic_tools as st


if __name__ == '__main__':

    do_bb_binning = True
    output_path   = f'data/templates/bjet_binned'
    data_labels  = ['muon', 'electron']
    model_labels = ['wjets', 'zjets', 't', 'ttbar', 'diboson']
    datasets = [d for l in data_labels + model_labels for d in pt.dataset_dict[l]]

    datasets_ttbar_syst = [
                           'ttbar_inclusive_isrup', 'ttbar_inclusive_isrdown',
                           'ttbar_inclusive_fsrup', 'ttbar_inclusive_fsrdown',
                           'ttbar_inclusive_hdampup', 'ttbar_inclusive_hdampdown',
                           'ttbar_inclusive_tuneup', 'ttbar_inclusive_tunedown',
                          ]

    # sigal samples are split according the decay of the W bosons
    decay_map     = pd.read_csv('data/decay_map.csv').set_index('id')
    mc_conditions = {decay_map.loc[i, 'decay']: f'gen_cat == {i}' for i in range(1, 22)}

    selections = ['ee']#, 'mumu', 'emu', 'etau', 'mutau', 'e4j', 'mu4j']
    pt.make_directory(f'{output_path}')
    for selection in selections:
        print(f'Running over selection {selection}...')
        feature    = fh.features[selection]
        ntuple_dir = f'data/flatuples/single_lepton/{selection}_2016'
        outfile    = open(f'{output_path}/{selection}_templates.pkl', 'wb')

        # category specific parameters
        labels = ['ttbar', 't', 'wjets', 'zjets', 'diboson']
        if selection == 'mu4j':
            dataset_names = datasets + pt.dataset_dict['fakes']
            labels += ['fakes']
        elif selection == 'etau' or selection == 'mutau':
            dataset_names = datasets + pt.dataset_dict['fakes_ss']
            labels += ['fakes_ss']
        else:
            dataset_names = datasets

        dm = pt.DataManager(input_dir     = ntuple_dir,
                            dataset_names = dataset_names,
                            selection     = selection,
                            scale         = 35.9e3,
                            cuts          = pt.cuts[selection],
                            )

        #dm_syst = pt.DataManager(input_dir     = f'data/flatuples/single_lepton_ttbar_syst/{selection}_2016',
        #                         dataset_names = datasets_ttbar_syst,
        #                         selection     = selection,
        #                         scale         = 35.9e3,
        #                         cuts          = pt.cuts[selection]
        #                         )


        data = dict()
        for i, bcut in enumerate(['n_bjets == 0', 'n_bjets == 1', 'n_bjets >= 2']):
            if selection not in ['mutau', 'etau'] and bcut == 'n_bjets == 0': 
                continue

            if selection in ['e4j', 'mu4j']:
                jet_cut = 'n_jets >= 4'
            else:
                jet_cut = 'n_jets >= 2'
            full_cut = f'{jet_cut} and {bcut}'

            ### calculate binning based on ttbar sample
            df_ttbar = dm.get_dataframe('ttbar', full_cut)
            if df_ttbar.shape[0] == 0: continue

            x = df_ttbar[feature].values 
            if do_bb_binning:
                print('Calculating Bayesian block binning...')
                binning = bayesian_blocks(x[:30000], p0=0.00001)

                if feature == 'dilepton1_pt_asym':
                    dxmin, dxmax = 0.01, 1
                else:
                    dxmin, dxmax = 2, 1e9

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

            # generate the data template
            df_data = dm.get_dataframe('data', full_cut)
            h, b = np.histogram(df_data[feature],
                                bins  = binning,
                                range = bin_range,
                                )
            #overflow = x[x > b[-1]].size
            #h = np.append(h, overflow)

            ### get signal and background templates
            templates = dict(data = dict(val = h, var = h))
            for label in labels:
                if label in ['ttbar', 't', 'wjets']: 

                    # divide ttbar and tW samples into 21 decay modes and
                    # w+jets sample into 6 decay modes
                    mode_dict = dict()
                    for n, c in mc_conditions.items():
                        idecay = int(c.split()[-1])
                        if label == 'wjets' and (idecay < 16): 
                            continue

                        x = dm.get_dataframe(label, f'{full_cut} and {c}')[feature].values
                        w = dm.get_dataframe(label, f'{full_cut} and {c}')['weight'].values
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

                        if label == 'wjets':
                            n = n.rsplit('_', 1)[0]

                        # save templates, statistical errors, and systematic variations
                        df_temp = pd.DataFrame(dict(bins=binning[:-1], val=h, var=hvar))
                    
                        ### produce morphing templates for shape systematics
                        df = dm.get_dataframe(label)
                        syst_gen = st.SystematicTemplateGenerator(selection, binning, h, full_cut, i)
                        syst_gen.jet_shape_systematics(df_temp, h, df, binning, selection, full_cut)

                        #df = df.query(full_cut)
                        #st.reco_shape_systematics(df_temp, h, df, binning, selection)
                        #st.theory_shape_systematics(df_temp, h, df, binning, selection)
                        mode_dict[n] = df_temp

                    templates[label] = mode_dict

                else: # 1 template per background
                    x = dm.get_dataframe(label, full_cut)[feature].values
                    w = dm.get_dataframe(label, full_cut)['weight'].values

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

                    if label == 'fakes_ss':
                        label = 'fakes'
                    templates[label] = dict(val = h, var = hvar)

            data[i] = dict(bins = binning, templates = templates, systematics = df_sys)

        # write the templates and morphing templates to file
        pickle.dump(data, outfile)
        outfile.close()
