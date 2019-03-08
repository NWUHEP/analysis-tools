import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from skhep.modeling import bayesian_blocks

import scripts.plot_tools as pt
import scripts.fit_helpers as fh
import scripts.systematic_tools as st

if __name__ == '__main__':

    # input arguments
    parser = argparse.ArgumentParser(description='Produce data/MC overlays')
    parser.add_argument('input',
                        help = 'specify input directory',
                        type = str
                        )
    parser.add_argument('output',
                        help = 'specify output directory',
                        type = str
                        )
    parser.add_argument('-b', '--bayesian-block',
                        help    = 'use Bayesian Block binning',
                        type    = bool,
                        default = True
                        )
    args = parser.parse_args()
    ##########################

    datasets_ttbar_syst = [
                           'ttbar_inclusive_isrup', 'ttbar_inclusive_isrdown',
                           'ttbar_inclusive_fsrup', 'ttbar_inclusive_fsrdown',
                           'ttbar_inclusive_hdampup', 'ttbar_inclusive_hdampdown',
                           'ttbar_inclusive_tuneup', 'ttbar_inclusive_tunedown',
                          ]

    # sigal samples are split according the decay of the W bosons
    decay_map     = pd.read_csv('data/decay_map.csv').set_index('id')
    mc_conditions = {decay_map.loc[i, 'decay']: f'gen_cat == {i}' for i in range(1, 22)}

    selections = ['ee', 'mumu', 'emu', 'etau', 'mutau', 'e4j', 'mu4j']
    pt.make_directory(f'{args.output}')
    for selection in selections:
        print(f'Running over category {selection}...')
        feature    = fh.features[selection]
        ntuple_dir = f'{args.input}/{selection}_2016'
        outfile    = open(f'{args.output}/{selection}_templates.pkl', 'wb')

        if selection in ['ee', 'etau', 'e4j']:
            data_labels = ['electron']
        elif selection in ['mumu', 'mutau', 'mu4j']:
            data_labels = ['muon']
        elif selection == 'emu':
            data_labels = ['electron', 'muon']
        labels = pt.selection_dataset_dict[selection]
        datasets = [d for l in data_labels + labels for d in pt.dataset_dict[l]]

        dm = pt.DataManager(input_dir     = ntuple_dir,
                            dataset_names = datasets,
                            selection     = selection,
                            scale         = 35.9e3,
                            cuts          = pt.cuts[selection],
                            )

        #dm_syst = pt.DataManager(input_dir     = f'local_data/flatuples/ttbar_systematics/{selection}_2016',
        #                         dataset_names = datasets_ttbar_syst,
        #                         selection     = selection,
        #                         scale         = 35.9e3,
        #                         cuts          = pt.cuts[selection]
        #                         )

        data = dict()
        selection_categories = [c for c, citems in pt.categories.items() if selection in citems.selections]
        for category in tqdm(selection_categories,
                             desc       = 'binning jet categories...',
                             unit_scale = True,
                             ncols      = 75,
                            ):
            cat_items = pt.categories[category]

            ### calculate binning based on data sample
            df_data = dm.get_dataframe('data', cat_items.cut)
            if df_data.shape[0] == 0: continue

            x = df_data[feature].values 
            if args.bayesian_block:
                #print('Calculating Bayesian block binning...')
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

                #print('Done!')
                bin_range = None
            else:
                #print('Using user-defined binning...')
                hist_lut  = dm._lut_features.loc[feature]
                binning   = hist_lut.n_bins
                bin_range = (hist_lut.xmin, hist_lut.xmax)

            # generate the data template
            df_data = dm.get_dataframe('data', cat_items.cut)
            h, b = np.histogram(df_data[feature],
                                bins  = binning,
                                range = bin_range,
                                )

            ### get signal and background templates
            templates = dict(data = dict(val = h, var = h))
            for label in labels:
                if label in ['ttbar', 't', 'wjets', 'ww']: 
                    # divide ttbar and tW samples into 21 decay modes and
                    # w+jets sample into 6 decay modes

                    mode_dict = dict()
                    for n, c in mc_conditions.items():
                        idecay = int(c.split()[-1])
                        if label == 'wjets' and (idecay < 16): 
                            continue

                        df = dm.get_dataframe(label, f'{cat_items.cut} and {c}')
                        x = df[feature].values
                        w = df['weight'].values
                        h, _ = np.histogram(x,
                                            bins    = binning,
                                            range   = bin_range,
                                            weights = w
                                            )

                        hvar, _ = np.histogram(x,
                                               bins    = binning,
                                               range   = bin_range,
                                               weights = w**2
                                               )

                        if label == 'wjets':
                            n = n.rsplit('_', 1)[0]

                        # save templates, statistical errors, and systematic variations
                        df_temp = pd.DataFrame(dict(bins=binning[:-1], val=h, var=hvar))
                        df_temp = df_temp.set_index('bins')
                         
                        ### produce morphing templates for shape systematics
                        if np.any(h != 0):
                            if np.sqrt(np.sum(hvar))/np.sum(h) < 0.05: 

                                df = dm.get_dataframe(label, c)
                                syst_gen = st.SystematicTemplateGenerator(selection, f'{label}_{n}', 
                                                                          feature, binning, 
                                                                          h, cat_items.cut, category)
                                # don't apply jet cut for jet syst.
                                syst_gen.jes_systematics(df)
                                syst_gen.btag_systematics(df)

                                # apply jet cut for all other systematics
                                df = df.query(cat_items.cut)
                                syst_gen.misc_systematics(df)
                                if selection in ['mumu', 'emu', 'mutau', 'mu4j']:
                                    syst_gen.muon_systematics(df)

                                if selection in ['ee', 'emu', 'etau', 'e4j']:
                                    syst_gen.electron_systematics(df)

                                # tau misid 
                                if selection in ['etau', 'mutau']: 
                                    if label == 'wjets' or idecay in [16, 17, 18, 19, 20, 21]:
                                        syst_gen.tau_misid_systematics(df)
                                    elif idecay in [7, 8, 9, 12, 15]:
                                        syst_gen.tau_systematics(df)

                                # theory systematics
                                if label == 'ttbar':
                                    syst_gen.top_pt_systematics(df)
                                    syst_gen.theory_systematics(df, label, cat_items.njets, f'{cat_items.cut} and {c}')

                                df_temp = pd.concat([df_temp, syst_gen.get_syst_dataframe()], axis=1)

                        mode_dict[n] = df_temp

                    templates[label] = mode_dict

                else: # 1 template per background
                    if label not in dm._dataframes.keys():
                        continue

                    x = dm.get_dataframe(label, cat_items.cut)[feature].values
                    w = dm.get_dataframe(label, cat_items.cut)['weight'].values

                    h, _ = np.histogram(x,
                                        bins    = binning,
                                        range   = bin_range,
                                        weights = w
                                        )

                    hvar, _ = np.histogram(x,
                                           bins    = binning,
                                           range   = bin_range,
                                           weights = w**2
                                           )

                    if label == 'fakes_ss':
                        label = 'fakes'

                    # save templates, statistical errors, and systematic variations
                    df_temp = pd.DataFrame(dict(bins=binning[:-1], val=h, var=hvar))
                    df_temp = df_temp.set_index('bins')

                    ### produce morphing templates for shape systematics
                    total_var = np.sqrt(np.sum(hvar))/np.sum(h)
                    if label == 'zjets_alt' and np.abs(total_var) < 0.05: # only consider systematics if sigma_N/N < 5%

                        df = dm.get_dataframe(label)
                        syst_gen = st.SystematicTemplateGenerator(selection, f'{label}', 
                                                                  feature, binning, 
                                                                  h_nominal = h, 
                                                                  cut = f'{cat_items.cut}', 
                                                                  cut_name = category
                                                                  )
                        # don't apply jet cut for jet syst.
                        syst_gen.jes_systematics(df)
                        syst_gen.btag_systematics(df)

                        # apply jet cut for all other systematics
                        df = df.query(cat_items.cut)
                        syst_gen.misc_systematics(df)
                        if selection in ['mumu', 'emu', 'mutau', 'mu4j']:
                            syst_gen.muon_systematics(df)

                        if selection in ['ee', 'emu', 'etau', 'e4j']:
                            syst_gen.electron_systematics(df)

                        # tau misid 
                        if selection in ['etau', 'mutau']: 
                            if label == 'wjets' or idecay in [16, 17, 18, 19, 20, 21]:
                                syst_gen.tau_misid_systematics(df)
                            elif idecay in [7, 8, 9, 12, 15]:
                                syst_gen.tau_systematics(df)

                        # theory systematics
                        syst_gen.theory_systematics(df, label, cat_items.njets, f'{cat_items.cut} and {c}')

                        df_temp = pd.concat([df_temp, syst_gen.get_syst_dataframe()], axis=1)

                    templates[label] = df_temp

            data[category] = dict(bins = binning, templates = templates)

        # write the templates and morphing templates to file
        pickle.dump(data, outfile)
        outfile.close()
