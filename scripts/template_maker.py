import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from skhep.modeling import bayesian_blocks

import scripts.plot_tools as pt
import scripts.fit_helpers as fh
import scripts.systematic_tools as st
from scripts.blt_reader import jec_source_names, btag_source_names

np.set_printoptions(precision=2)

def get_binning(df, feature, do_bbb=True):
    '''
    Gets binning for histogram templates.  By default uses Bayesian Block Binning.

    Parameters:
    ===========
    dm: data manager instance
    feature: --
    cuts: --
    do_bbb: use Bayesian Block binning
    '''

    x = df[feature].values
    if do_bbb:
        # only consider 30k data point (more than that takes too long)
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

    else:
        hist_lut  = dm._lut_features.loc[feature]
        binning   = np.linspace(hist_lut.xmin, hist_lut.xmax, hist_lut.n_bins)

    hist, _ = np.histogram(df[feature], bins = binning)

    return hist, binning

def make_templates(df, binning):

    # get nominal template
    x = df[feature].values
    w = df['weight'].values
    h, _ = np.histogram(x, bins = binning, weights = w)
    hvar, _ = np.histogram(x, bins = binning, weights = w**2)

    # save templates, statistical errors, and systematic variations
    df_template = pd.DataFrame(dict(bins=binning[:-1], val=h, var=hvar))
    df_template = df_template.set_index('bins')

    return df_template
    
def make_morphing_templates(df, label, syst_gen, cat_items):
    
    dataset    = label[0]
    decay_mode = label[1]
    selection  = syst_gen._selection

    # calculate jet systematic morphing w/o jet cut 
    syst_gen.jes_systematics(df, cat_items.jet_cut)
    syst_gen.btag_systematics(df, cat_items.jet_cut)

    # apply jet cut for all other systematics
    df = df.query(cat_items.jet_cut)
    syst_gen.misc_systematics(df)
    if selection in ['mumu', 'emu', 'mutau', 'mu4j']:
        syst_gen.muon_systematics(df)

    if selection in ['ee', 'emu', 'etau', 'e4j']:
        syst_gen.electron_systematics(df)

    # tau misid 
    if selection in ['etau', 'mutau']:
        if decay_mode in [7, 8, 12, 15] or dataset == 'zjets_alt':
            syst_gen.tau_systematics(df)
        elif decay_mode in [16, 17, 18, 19, 20, 21]:
            syst_gen.tau_j_misid_systematics(df)
        elif decay_mode in [1, 3, 6, 10, 11, 13]:
            syst_gen.tau_e_misid_systematics(df)
    #elif selection in ['e4j', 'mu4j']: # add (1 - eff) for l+jet categories
    #    if decay_mode in [7, 8, 12, 15]:
    #        syst_gen.tau_systematics(df)
        
    # theory systematics
    if dataset in ['ttbar', 'zjets_alt']:
        syst_gen.theory_systematics(df, dataset, cat_items.njets)

        if dataset == 'ttbar':
            syst_gen.top_pt_systematics(df)

            if cat_items.cut == None:
                cut = f'gen_cat == {decay_mode} and {cat_items.jet_cut}'
            else:
                cut = f'gen_cat == {decay_mode} and {cat_items.jet_cut} and {cat_items.cut}'

    # ww pt
    if dataset == 'ww' and selection == 'emu':
        syst_gen.ww_pt_systematics(df)

    return syst_gen.get_syst_dataframe()

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
                           #'ttbar_inclusive_hdampup', 'ttbar_inclusive_hdampdown',
                           #'ttbar_inclusive_tuneup', 'ttbar_inclusive_tunedown',

                 	   'ttbar_inclusive_isrup_ext2', 'ttbar_inclusive_isrdown_ext2',
                 	   'ttbar_inclusive_fsrup_ext1', 'ttbar_inclusive_fsrdown_ext1',    
                 	   'ttbar_inclusive_fsrup_ext2', 'ttbar_inclusive_fsrdown_ext2',    
                 	   'ttbar_inclusive_hdampup_ext1', 'ttbar_inclusive_hdampdown_ext1',
                 	   'ttbar_inclusive_tuneup_ext1', 'ttbar_inclusive_tunedown_ext1',
                          ]

    # sigal samples are split according the decay of the W bosons
    decay_map = pd.read_csv('data/decay_map.csv').set_index('id')

    # features to keep in memory
    feature_list = [
                    'lepton1_pt', 'lepton2_pt',
                    'lead_lepton_pt','trailing_lepton_pt',
                    'lead_lepton_flavor', 'trailing_lepton_flavor',
                    'dilepton1_mass', 'dilepton1_delta_phi', 'lepton1_mt',
                    'n_jets', 'n_bjets', 'tau_decay_mode',
                    
                    'n_pu', 'gen_cat',
                    'lepton1_reco_weight', 'lepton2_reco_weight',
                    'lepton1_id_weight', 'lepton2_id_weight',
                    'trigger_weight', 'pileup_weight', 'top_pt_weight',
                    'z_pt_weight', 'ww_pt_weight', 'event_weight',

                    'qcd_weight_nominal', 'qcd_weight_nom_up', 'qcd_weight_nom_down',
                    'qcd_weight_up_nom', 'qcd_weight_up_up', 'qcd_weight_down_nom',
                    'qcd_weight_down_down', 'pdf_var', 'alpha_s_err',

                    'ww_pt_scale_up', 'ww_pt_scale_down',
                    'ww_pt_resum_up', 'ww_pt_resum_down',

                    'lepton1_id_var', 'lepton2_id_var',
                    'lepton1_reco_var', 'lepton2_reco_var',
                    'trigger_var', 'el_trigger_syst_tag', 'el_trigger_syst_probe',

                    'n_jets_jer_up', 'n_jets_jer_down',
                    'n_bjets_jer_up', 'n_bjets_jer_down',
                    'n_bjets_ctag_up', 'n_bjets_ctag_down',
                    'n_bjets_mistag_up', 'n_bjets_mistag_down'
                   ]

    feature_list += [f'n_jets_jes_{n}_up' for n in jec_source_names]
    feature_list += [f'n_jets_jes_{n}_down' for n in jec_source_names]
    feature_list += [f'n_bjets_jes_{n}_up' for n in jec_source_names]
    feature_list += [f'n_bjets_jes_{n}_down' for n in jec_source_names]
    feature_list += [f'n_bjets_btag_{n}_up' for n in btag_source_names]
    feature_list += [f'n_bjets_btag_{n}_down' for n in btag_source_names]

    selections = ['ee', 'mumu', 'emu', 'etau', 'mutau', 'e4j', 'mu4j']
    #selections = ['etau']
    pt.make_directory(f'{args.output}', clear=False)
    for selection in selections:
        print(f'Running over category {selection}...')
        feature    = fh.features[selection]
        ntuple_dir = f'{args.input}/{selection}_2016'
        outfile    = open(f'{args.output}/{selection}_templates.pkl', 'wb')

        # get the data dataframe
        if selection in ['ee', 'etau', 'e4j']:
            data_labels = ['electron']
        elif selection in ['mumu', 'mutau', 'mu4j']:
            data_labels = ['muon']
        elif selection == 'emu':
            data_labels = ['electron', 'muon']

        datasets = [d for l in data_labels for d in pt.dataset_dict[l]]
        dm = pt.DataManager(input_dir     = ntuple_dir,
                            dataset_names = datasets,
                            selection     = selection,
                            scale         = 35.9e3,
                            cuts          = pt.cuts[selection],
                            features      = feature_list[:11]
                           )
        df_data = dm.get_dataframe('data')

        # prepare signal and background datasets
        labels = pt.selection_dataset_dict[selection]
        datasets = [d for l in labels for d in pt.dataset_dict[l]]
        dm = pt.DataManager(input_dir     = ntuple_dir,
                            dataset_names = datasets,
                            selection     = selection,
                            scale         = 35.9e3,
                            cuts          = pt.cuts[selection],
                            features      = feature_list
                           )

        data = dict()
        selection_categories = [c for c, citems in pt.categories.items() if selection in citems.selections]
        for category in tqdm(selection_categories,
                             desc       = 'binning jet categories...',
                             unit_scale = True,
                             ncols      = 75,
                            ):
            cat_items = pt.categories[category]
            if cat_items.cut is None:
                full_cut = f'{cat_items.jet_cut}'
            else:
                full_cut = f'{cat_items.cut} and {cat_items.jet_cut}'

            # generate the data template
            h, binning = get_binning(df_data.query(full_cut), feature) 

            ### get signal and background templates
            templates = dict(data = dict(val = h, var = h))
            for label in labels:
                if label in ['ttbar', 't', 'ww', 'wjets']: 
                    # divide ttbar, tW, and ww samples into 21 decay modes
                    # w+jets sample into 6 decay modes

                    mode_dict = dict()
                    for idecay, decay_data in decay_map.iterrows():
                        if label == 'wjets' and (idecay < 16):
                            continue

                        df = dm.get_dataframe(label, f'{full_cut} and gen_cat == {idecay}')
                        df_template = make_templates(df, binning)

                        # produce morphing templates for shape systematics if
                        # any bin has more events than 5% of the total error in
                        # that bin
                        hval, hvar = df_template['val'], df_template['var']
                        if np.all(hval < 0.1*np.sqrt(h)): 
                            mode_dict[decay_data.decay] = df_template
                            continue

                        if cat_items.cut is None:
                            df = dm.get_dataframe(label, f'gen_cat == {idecay}')
                        else:
                            df = dm.get_dataframe(label, f'{cat_items.cut} and gen_cat == {idecay}')

                        # initialize systematics generator
                        syst_gen = st.SystematicTemplateGenerator(selection, feature, binning, df_template['val'].values)
                        df_syst = make_morphing_templates(df, (label, idecay), syst_gen, cat_items)
                        mode_dict[decay_data.decay] = pd.concat([df_template, df_syst], axis=1)
                        #mode_dict[decay_data.decay] = df_template

                    templates[label] = mode_dict

                else: # 1 template per background
                    if label not in dm._dataframes.keys():
                        continue

                    df = dm.get_dataframe(label, full_cut)
                    df_template = make_templates(df, binning)

                    # produce morphing templates for shape systematics if
                    # any bin has more events than 5% of the total error in
                    # that bin (excludes empty templates obviously)
                    hval, hvar = df_template['val'], df_template['var']
                    if label != 'zjets_alt' or np.all(hval < 0.1*np.sqrt(h)): 
                        templates[label] = df_template
                        continue

                    if cat_items.cut is None:
                        df = dm.get_dataframe(label)
                    else:
                        df = dm.get_dataframe(label, cat_items.cut)

                    syst_gen = st.SystematicTemplateGenerator(selection, feature, binning, df_template['val'].values)
                    df_syst = make_morphing_templates(df, (label, None), syst_gen, cat_items)
                    templates[label] = pd.concat([df_template, df_syst], axis=1)
                    #templates[label] = df_template

            data[category] = dict(bins = binning, templates = templates)

        # Additional ttbar-specific systematics
        dm = pt.DataManager(input_dir     = f'local_data/flatuples/ttbar_systematics_new/{selection}_2016',
                            dataset_names = datasets_ttbar_syst,
                            selection     = selection,
                            scale         = 35.9e3,
                            cuts          = pt.cuts[selection],
                            features      = feature_list
                           )

        for category in tqdm(selection_categories,
                             desc       = 'producing ttbar systematics',
                             unit_scale = True,
                             ncols      = 75,
                            ):
            cat_items = pt.categories[category]
            if cat_items.cut is None:
                full_cut = f'{cat_items.jet_cut}'
            else:
                full_cut = f'{cat_items.cut} and {cat_items.jet_cut}'

            for idecay, decay_data in decay_map.iterrows():
                binning = data[category]['bins']
                total_err = np.sqrt(data[category]['templates']['data']['var'])
                df_syst = data[category]['templates']['ttbar'][decay_data.decay]

                if np.any(df_syst['val'] > 0.5*total_err):
                    st.ttbar_systematics(dm, df_syst,
                                         f'{full_cut} and gen_cat == {idecay}',
                                         idecay, feature, binning,
                                         smooth = None
                                        )

        # write the templates and morphing templates to file
        pickle.dump(data, outfile)
        outfile.close()
