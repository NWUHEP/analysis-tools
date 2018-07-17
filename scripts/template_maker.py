import pickle

import numpy as np
import pandas as pd
from skhep.modeling import bayesian_blocks

import scripts.plot_tools as pt
import scripts.fit_helpers as fh
import scripts.systematic_tools as st

if __name__ == '__main__':

    do_bb_binning = True
    output_path   = f'data/templates/bjet_binned_test'
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
    for selection in selections:
        print(selection)
        feature = fh.features[selection]
        pt.make_directory(f'{output_path}')
        ntuple_dir = f'data/flatuples/single_lepton_test/{selection}_2016'
        outfile = open(f'{output_path}/{selection}_templates.pkl', 'wb')

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

        #print(f'Running over selection {selection}...')
        data = dict()
        for i, bcut in enumerate(['n_bjets == 0', 'n_bjets == 1', 'n_bjets >= 2']):
            if selection not in ['mutau', 'etau'] and bcut == 'n_bjets == 0': 
                continue

            if selection in ['e4j', 'mu4j']:
                jet_cut = 'n_jets >= 4'
            else:
                jet_cut = 'n_jets >= 2'


            df_model = {}
            for l in labels:
                df_model[l] = dm.get_dataframe(l).query(jet_cut + ' and ' + bcut)

            ### calculate binning based on ttbar sample
            df_ttbar = df_model['ttbar']
            if df_model['ttbar'].shape[0] == 0: continue

            x = df_ttbar[feature].values # what's going on here \0/
            if do_bb_binning:
                print('Calculating Bayesian block binning...')
                binning = bayesian_blocks(x[:30000], p0=0.0001)

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
            templates = dict(data = dict(val = h, var = h))
            for label, df in df_model.items():
                if label in ['ttbar', 't', 'wjets']: # divide ttbar and tW samples into 21 decay modes
                    dvals = dict()
                    dvars = dict()
                    for n, c in mc_conditions.items():
                        idecay = int(c.split()[-1])
                        if label == 'wjets' and (idecay < 16): 
                            continue

                        x = df.query(c)[feature].values
                        w = df.query(c)['weight'].values
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
                        dvals[n] = h
                        dvars[n] = hvar

                    # make sure the columns are properly ordered
                    if label == 'wjets':
                        dlabels = ['we', 'wmu', 'wtau_e', 'wtau_mu', 'wtau_h', 'wh']
                    else:
                        dlabels = decay_map['decay']
                    templates[label] = dict(val = pd.DataFrame(dvals)[dlabels], var = pd.DataFrame(dvars)[dlabels])

                else: # 1 template per background
                    x = df[feature].values
                    w = df['weight'].values

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

            #store[f'{selection}/{i}] = templates
            #store.close()

            ### produce morphing templates for shape systematics
            df_sys = pd.DataFrame(dict(bins=binning[:-1]))

            ### jet systematics ###
            df_ttbar_no_jetcut = dm.get_dataframes(['ttbar', 't', 'wjets'], concat=True) # we need to get the dataframe without the jet cuts
            h_nominal, _ = np.histogram(df_ttbar[feature], bins=binning, weights=df_ttbar.weight)

            # jes
            df_sys['jes_up'], df_sys['jes_down'] = st.jet_scale(df_ttbar_no_jetcut, feature, binning, 'jes', jet_cut + ' and ' + bcut)
            st.template_overlays(h_nominal, h_nominal*df_sys['jes_up'], h_nominal*df_sys['jes_down'], binning, 'jes', selection, feature, i)

            # jer
            df_sys['jer_up'], df_sys['jer_down'] = st.jet_scale(df_ttbar_no_jetcut, feature, binning, 'jer', jet_cut + ' and ' + bcut)
            st.template_overlays(h_nominal, h_nominal*df_sys['jer_up'], h_nominal*df_sys['jer_down'], binning, 'jer', selection, feature, i)

            # b tag eff
            df_sys['btag_up'], df_sys['btag_down'] = st.jet_scale(df_ttbar_no_jetcut, feature, binning, 'btag', jet_cut + ' and ' + bcut)
            st.template_overlays(h_nominal, h_nominal*df_sys['btag_up'], h_nominal*df_sys['btag_down'], binning, 'btag', selection, feature, i)

            # mistag eff
            df_sys['mistag_up'], df_sys['mistag_down'] = st.jet_scale(df_ttbar_no_jetcut, feature, binning, 'mistag', jet_cut + ' and ' + bcut)
            st.template_overlays(h_nominal, h_nominal*df_sys['mistag_up'], h_nominal*df_sys['mistag_down'], binning, 'mistag', selection, feature, i)

            # pileup
            df_sys['pileup_up'], df_sys['pileup_down'] = st.pileup_morph(df_ttbar, feature, binning)
            st.template_overlays(h_nominal, h_nominal*df_sys['pileup_up'], h_nominal*df_sys['pileup_down'], binning, 'pileup', selection, feature, i)

            ### lepton energy scale ###
            if selection in ['mumu', 'emu', 'mu4j']:
                scale = 0.01
                df_sys['mu_es_up'], df_sys['mu_es_down'] = st.les_morph(df_ttbar, feature, binning, scale)
                st.template_overlays(h_nominal, h_nominal*df_sys['mu_es_up'], h_nominal*df_sys['mu_es_down'], binning, 'mu_es', selection, feature, i)

            if selection in ['ee', 'emu', 'e4j']:
                scale = 0.01
                df_sys['el_es_up'], df_sys['el_es_down'] = st.les_morph(df_ttbar, feature, binning, scale)
                st.template_overlays(h_nominal, h_nominal*df_sys['el_es_up'], h_nominal*df_sys['el_es_down'], binning, 'el_es', selection, feature, i)

            if selection in ['etau', 'mutau']:
                scale = 0.01
                df_sys['tau_es_up'], df_sys['tau_es_down'] = st.les_morph(df_ttbar, feature, binning, scale)
                st.template_overlays(h_nominal, h_nominal*df_sys['tau_es_up'], h_nominal*df_sys['tau_es_down'], binning, 'tau_es', selection, feature, i)

            # theory systematics
            dm_syst = pt.DataManager(input_dir     = f'data/flatuples/single_lepton_ttbar_syst/{selection}_2016',
                                     dataset_names = datasets_ttbar_syst,
                                     selection     = selection,
                                     scale         = 35.9e3,
                                     cuts          = pt.cuts[selection] + ' and ' + jet_cut + ' and ' + bcut
                                     )

            df_ttbar = dm.get_dataframe('ttbar').query(jet_cut + ' and ' + bcut)

            # isr
            df_sys['isr_up'], df_sys['isr_down'] = st.theory_systematics(df_ttbar, dm_syst, feature, binning, 'isr')
            st.template_overlays(h_nominal, h_nominal*df_sys['isr_up'], h_nominal*df_sys['isr_down'], binning, 'isr', selection, feature, i)

            # fsr
            df_sys['fsr_up'], df_sys['fsr_down'] = st.theory_systematics(df_ttbar, dm_syst, feature, binning, 'fsr')
            st.template_overlays(h_nominal, h_nominal*df_sys['fsr_up'], h_nominal*df_sys['fsr_down'], binning, 'fsr', selection, feature, i)

            # ME-PS (hdamp)
            df_sys['hdamp_up'], df_sys['hdamp_down'] = st.theory_systematics(df_ttbar, dm_syst, feature, binning, 'hdamp')
            st.template_overlays(h_nominal, h_nominal*df_sys['hdamp_up'], h_nominal*df_sys['hdamp_down'], binning, 'hdamp', selection, feature, i)

            # UE tune
            df_sys['tune_up'], df_sys['tune_down'] = st.theory_systematics(df_ttbar, dm_syst, feature, binning, 'tune')
            st.template_overlays(h_nominal, h_nominal*df_sys['tune_up'], h_nominal*df_sys['tune_down'], binning, 'tune', selection, feature, i)

            # PDF scale (average over MC replicas)
            df_sys['pdf_up'], df_sys['pdf_down'] = st.theory_systematics(df_ttbar, dm_syst, feature, binning, 'pdf')
            st.template_overlays(h_nominal, h_nominal*df_sys['pdf_up'], h_nominal*df_sys['pdf_down'], binning, 'pdf', selection, feature, i)

            # QCD scale (mu_R and mu_F variation)
            df_sys['qcd_up'], df_sys['qcd_down'] = st.theory_systematics(df_ttbar, dm_syst, feature, binning, 'qcd')
            st.template_overlays(h_nominal, h_nominal*df_sys['qcd_up'], h_nominal*df_sys['qcd_down'], binning, 'qcd', selection, feature, i)

            data[i] = dict(bins = binning, templates = templates, systematics = df_sys)

        # write the templates and morphing templates to file
        pickle.dump(data, outfile)
        outfile.close()
