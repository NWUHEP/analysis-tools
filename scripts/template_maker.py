import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from skhep.modeling import bayesian_blocks

import scripts.plot_tools as pt
import scripts.fit_helpers as fh
import scripts.systematic_tools as st

np.set_printoptions(precision=2)

def get_binning(dm, feature, cuts, dataset='data', do_bbb=True):
    '''
    Gets binning for histogram templates.  By default uses Bayesian Block Binning.

    Parameters:
    ===========
    dm: data manager instance
    feature: --
    cuts: --
    do_bbb: use Bayesian Block binning
    '''

    df = dm.get_dataframe(dataset, cuts)
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

    return binning


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
    
def make_morphing_templates(df, selection, label, feature, binning, h_nominal, cat_items):
    
    dataset = label[0]
    decay_mode = label[1]

    # initialize systematics generator
    syst_gen = st.SystematicTemplateGenerator(selection, feature, binning, h_nominal)

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
        elif decay_mode in [1, 3, 6, 10, 11, 13]: #
            syst_gen.tau_e_misid_systematics(df)
        
    # theory systematics
    if dataset in ['ttbar', 'zjets_alt']:
        syst_gen.theory_systematics(df, dataset, cat_items.njets)

        if dataset == 'ttbar':
            syst_gen.top_pt_systematics(df)

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
                           'ttbar_inclusive_hdampup', 'ttbar_inclusive_hdampdown',
                           'ttbar_inclusive_tuneup', 'ttbar_inclusive_tunedown',
                          ]

    # sigal samples are split according the decay of the W bosons
    decay_map = pd.read_csv('data/decay_map.csv').set_index('id')

    #selections = ['ee', 'mumu', 'emu', 'etau', 'mutau', 'e4j', 'mu4j']
    selections = ['etau', 'mutau']
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
            if cat_items.cut is None:
                full_cut = f'{cat_items.jet_cut}'
            else:
                full_cut = f'{cat_items.cut} and {cat_items.jet_cut}'

            ### calculate binning based on data sample

            df_data = dm.get_dataframe('data', full_cut)
            if df_data.shape[0] == 0:
                continue

            # generate the data template
            binning = get_binning(dm, feature, full_cut) 
            h, b = np.histogram(df_data[feature], bins = binning)

            ### get signal and background templates
            templates = dict(data = dict(val = h, var = h))
            for label in labels:
                if label in ['ttbar', 't', 'ww', 'wjets']: 
                    # divide ttbar and tW samples into 21 decay modes and
                    # w+jets sample into 6 decay modes

                    mode_dict = dict()
                    count = 0
                    for idecay, decay_data in decay_map.iterrows():
                        count += 1
                        if label == 'wjets' and (idecay < 16):
                            continue

                        df = dm.get_dataframe(label, f'{full_cut} and gen_cat == {idecay}')
                        df_template = make_templates(df, binning)

                        # if template is empty, don't generate morphing templates
                        if df_template['val'].sum() == 0:
                            mode_dict[decay_data.decay] = df_template
                            continue
                         
                        # produce morphing templates for shape systematics
                        hval, herr = np.sum(df_template['val']), np.sqrt(np.sum(df_template['var']))
                        #print(selection, category, label, decay_data.decay, hval, herr, herr/hval)
                        if herr/hval > 0.1: 
                            mode_dict[decay_data.decay] = df_template
                            continue

                        if cat_items.cut is None:
                            df = dm.get_dataframe(label, f'gen_cat == {idecay}')
                        else:
                            df = dm.get_dataframe(label, f'{cat_items.cut} and gen_cat == {idecay}')

                        df_syst = make_morphing_templates(df, selection, (label, idecay), 
                                                          feature, binning, 
                                                          df_template['val'].values, 
                                                          cat_items
                                                          )
                        mode_dict[decay_data.decay] = pd.concat([df_template, df_syst], axis=1)

                    templates[label] = mode_dict

                else: # 1 template per background
                    if label not in dm._dataframes.keys():
                        continue

                    df = dm.get_dataframe(label, full_cut)
                    df_template = make_templates(df, binning)

                    # if template is empty, don't generate morphing templates
                    if np.all(df_template['val'] == 0):
                        templates[label] = df_template
                        continue

                    # produce morphing templates for shape systematics
                    hval, herr = np.sum(df_template['val']), np.sqrt(np.sum(df_template['var']))
                    #print(selection, category, label, hval, herr, herr/hval)
                    if label != 'zjets_alt' or herr/hval > 0.1: 
                        templates[label] = df_template
                        continue

                    if cat_items.cut is None:
                        df = dm.get_dataframe(label)
                    else:
                        df = dm.get_dataframe(label, cat_items.cut)


                    df_syst = make_morphing_templates(df, selection, (label, None), 
                                                      feature, binning, 
                                                      df_template['val'].values, 
                                                      cat_items
                                                      )

                    if label == 'fakes_ss':
                        label = 'fakes'

                    templates[label] = pd.concat([df_template, df_syst], axis=1)

            data[category] = dict(bins = binning, templates = templates)

        # write the templates and morphing templates to file
        pickle.dump(data, outfile)
        outfile.close()
