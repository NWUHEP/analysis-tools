import pickle

import numpy as np
import pandas as pd
from skhep.modeling import bayesian_blocks

import scripts.plot_tools as pt
import scripts.fit_helpers as fh
import scripts.systematic_tools as st


if __name__ == '__main__':

    do_bb_binning = True
    output_path   = f'local_data/templates/bjet_binned'
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

    selections = ['ee', 'mumu', 'emu', 'etau', 'mutau', 'e4j', 'mu4j']
    pt.make_directory(f'{output_path}')
    for selection in selections:
        print(f'Running over category {selection}...')
        feature    = fh.features[selection]
        ntuple_dir = f'local_data/flatuples/single_lepton/{selection}_2016'
        outfile    = open(f'{output_path}/{selection}_templates.pkl', 'wb')

        # category specific parameters
        labels = ['ttbar', 't', 'wjets', 'zjets', 'diboson']
        if selection == 'mu4j' or selection == 'e4j':
            dataset_names = datasets + pt.dataset_dict['fakes']
            labels += ['fakes']
        elif selection == 'etau' or selection == 'mutau':
            dataset_names = datasets + pt.dataset_dict['fakes_ss']
            labels += ['fakes_ss']
        else:
            dataset_names = datasets

        #labels =['ttbar']

        dm = pt.DataManager(input_dir     = ntuple_dir,
                            dataset_names = dataset_names,
                            selection     = selection,
                            scale         = 35.9e3,
                            cuts          = pt.cuts[selection],
                            )

        dm_syst = pt.DataManager(input_dir     = f'local_data/flatuples/single_lepton_ttbar_syst/{selection}_2016',
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

        for ds in datasets_ttbar_syst:
            ds = ds.replace('_inclusive', '')
            df_syst  = dm_syst.get_dataframe(ds)
            l = ds.split('_')[-1]
            k_theory[l] = df_syst.weight.sum()/n_nominal

        data = dict()
        for i, bcut in enumerate(['n_bjets == 0', 'n_bjets == 1', 'n_bjets >= 2']):
            if selection in ['mu4j', 'e4j'] and bcut == 'n_bjets == 0': 
                continue

            if selection in ['e4j', 'mu4j']:
                jet_cut = 'n_jets >= 4'
            else:
                jet_cut = 'n_jets >= 2'
            full_cut = f'{jet_cut} and {bcut}'
            print(f'Applying selection "{full_cut}".')


            ### calculate binning based on data sample
            df_data = dm.get_dataframe('data', full_cut)
            if df_data.shape[0] == 0: continue

            x = df_data[feature].values 
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
                        df_temp = df_temp.set_index('bins')
                         
                        ### produce morphing templates for shape systematics
                        if np.any(h != 0):
                            if np.sqrt(np.sum(hvar))/np.sum(h) < 0.1: # only consider systematics if sigma_N/N < 10%

                                df = dm.get_dataframe(label, c)
                                syst_gen = st.SystematicTemplateGenerator(selection, f'{label}_{n}', 
                                                                          feature, binning, 
                                                                          h, full_cut, i)
                                syst_gen.jet_shape_systematics(df) # don't apply jet cut for jet syst.

                                df = df.query(full_cut)
                                syst_gen.reco_shape_systematics(df)
                                if label == 'ttbar':
                                    syst_gen.theory_shape_systematics(df, dm_syst, f'{full_cut} and {c}', k_theory)

                                df_temp = pd.concat([df_temp, syst_gen.get_syst_dataframe()], axis=1)

                                #df = df.query(full_cut)

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

            data[i] = dict(bins = binning, templates = templates)

        # write the templates and morphing templates to file
        pickle.dump(data, outfile)
        outfile.close()
