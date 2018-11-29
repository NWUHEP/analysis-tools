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
    #selections = ['etau', 'mutau']
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

        dm_syst = pt.DataManager(input_dir     = f'local_data/flatuples/ttbar_systematics/{selection}_2016',
                                 dataset_names = datasets_ttbar_syst,
                                 selection     = selection,
                                 scale         = 35.9e3,
                                 cuts          = pt.cuts[selection]
                                 )

        # the theory systematics should not modify the overall ttbar
        # cross-section.  To enforce this, a scaling factor is derived for each
        # systematic type so that their integral is the same as the nominal
        # ttbar sample (technically, this should be done over all final state
        # categories... maybe later).

        #corr_theory = calculate_theory_rescales(dm, dm_syst)
        k_theory = dict()
        df_ttbar = dm.get_dataframe('ttbar')
        w_nominal = df_ttbar.weight
        n_nominal = w_nominal.sum()

        # mur
        k_theory['mur_up']       = np.sum(w_nominal*df_ttbar.qcd_weight_up_nom)/n_nominal
        k_theory['mur_down']     = np.sum(w_nominal*df_ttbar.qcd_weight_down_nom)/n_nominal

        # muf
        k_theory['muf_up']       = np.sum(w_nominal*df_ttbar.qcd_weight_nom_up)/n_nominal
        k_theory['muf_down']     = np.sum(w_nominal*df_ttbar.qcd_weight_nom_down)/n_nominal

        # muf
        k_theory['mur_muf_up']   = np.sum(w_nominal*df_ttbar.qcd_weight_up_up)/n_nominal
        k_theory['mur_muf_down'] = np.sum(w_nominal*df_ttbar.qcd_weight_down_down)/n_nominal

        # pdf
        k_theory['pdf_up']       = np.sum(w_nominal*(1 + np.sqrt(df_ttbar.pdf_var)/np.sqrt(100)))/n_nominal
        k_theory['pdf_down']     = np.sum(w_nominal*(1 - np.sqrt(df_ttbar.pdf_var)/np.sqrt(100)))/n_nominal

        #for ds in datasets_ttbar_syst:
        #    ds = ds.replace('_inclusive', '')
        #    df_syst  = dm_syst.get_dataframe(ds)
        #    l = ds.split('_')[-1]
        #    k_theory[l] = df_syst.weight.sum()/n_nominal

        data = dict()
        for i, (category, cat_items) in enumerate(tqdm(pt.categories.items(),
                                                       desc       = 'binning jet categories...',
                                                       unit_scale = True,
                                                       ncols      = 75,
                                                       )):
            if selection not in cat_items.selections:
                continue

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

                        x = dm.get_dataframe(label, f'{cat_items.cut} and {c}')[feature].values
                        w = dm.get_dataframe(label, f'{cat_items.cut} and {c}')['weight'].values
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
                        df_temp = df_temp.set_index('bins')
                         
                        ### produce morphing templates for shape systematics
                        if np.any(h != 0):
                            if np.sqrt(np.sum(hvar))/np.sum(h) < 0.1: # only consider systematics if sigma_N/N < 10%

                                df = dm.get_dataframe(label, c)
                                syst_gen = st.SystematicTemplateGenerator(selection, f'{label}_{n}', 
                                                                          feature, binning, 
                                                                          h, cat_items.cut, category)
                                syst_gen.jet_shape_systematics(df) # don't apply jet cut for jet syst.

                                df = df.query(cat_items.cut)
                                syst_gen.reco_shape_systematics(df)
                                syst_gen.electron_reco_systematics(df)
                                if label == 'ttbar':
                                    syst_gen.theory_shape_systematics(df, f'{cat_items.cut} and {c}', k_theory)

                                df_temp = pd.concat([df_temp, syst_gen.get_syst_dataframe()], axis=1)

                        mode_dict[n] = df_temp

                    templates[label] = mode_dict

                else: # 1 template per background
                    if label not in dm._dataframes.keys():
                        continue

                    x = dm.get_dataframe(label, cat_items[0])[feature].values
                    w = dm.get_dataframe(label, cat_items[0])['weight'].values

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
                        #print(selection, category, h)
                        label = 'fakes'

                    # save templates, statistical errors, and systematic variations
                    df_temp = pd.DataFrame(dict(bins=binning[:-1], val=h, var=hvar))
                    df_temp = df_temp.set_index('bins')

                    ### produce morphing templates for shape systematics
                    if label == 'zjets_alt' and np.any(h != 0):
                        if np.sqrt(np.sum(hvar))/np.sum(h) < 0.1: # only consider systematics if sigma_N/N < 10%

                            df = dm.get_dataframe(label)
                            syst_gen = st.SystematicTemplateGenerator(selection, f'{label}', 
                                                                      feature, binning, 
                                                                      h, cat_items.cut, category)
                            syst_gen.jet_shape_systematics(df) # don't apply jet cut for jet syst.

                            df = df.query(cat_items.cut)
                            syst_gen.reco_shape_systematics(df)
                            syst_gen.electron_reco_systematics(df)
                            df_temp = pd.concat([df_temp, syst_gen.get_syst_dataframe()], axis=1)

                    templates[label] = df_temp

            data[category] = dict(bins = binning, templates = templates)

        # write the templates and morphing templates to file
        pickle.dump(data, outfile)
        outfile.close()
