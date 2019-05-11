#!/usr/bin/env python

import os, sys
import shutil
import argparse
from copy import deepcopy
import multiprocessing as mp
from multiprocessing import Pool, current_process
from itertools import product

import numpy as np
import pandas as pd
import ROOT as r

from tqdm import tqdm, trange
import scripts.plot_tools as pt
import scripts.blt_reader as br

def pickle_ntuple(input_file, tree_name, output_path, event_range, ix):
    process = mp.current_process()

    # get the tree, convert to dataframe, and save df to pickle
    root_file = r.TFile(input_file)
    tree = root_file.Get(tree_name)
    selection, dataset = tree_name.replace('bltTree_', '').split('/')

    # strip 'fakes' suffix
    if '_fakes' in selection:
        selection = selection.replace('_fakes', '')

    ntuple = br.fill_ntuple(tree, selection, dataset, 
                            event_range = event_range, 
                            job_id = (process._identity[0], ix[0], ix[1])
                            )

    df = pd.DataFrame(ntuple)
    df = df.query('weight != 0') # remove deadweight
    if df.shape[0] > 0:
        df.to_pickle(f'{output_path}/{dataset}_{ix[0]}.pkl')
    root_file.Close()

    return

if __name__ == '__main__':

    # parse arguments #
    parser = argparse.ArgumentParser(description='Convert hgcal root ntuples to dataframes')
    parser.add_argument('input',
                        help='input rootfile',
                        type=str
                        )
    parser.add_argument('output',
                        help='output directory',
                        type=str
                        )
    parser.add_argument('-p', '--nprocesses',
                        help    = 'number of concurrent processes (will be less than number of available cores)',
                        default = 8,
                        type    = int
                        )
    parser.add_argument('-n', '--nevents',
                        help    = 'number of events to run per process',
                        default = 100000,
                        type    = int
                        )
    parser.add_argument('-m', '--nmax',
                        help    = 'maximum number of events to be processed',
                        default = sys.maxsize,
                        type    = int
                        )
    parser.add_argument('-a', '--append',
                        help    = 'Run in append mode (existing datasets will not be overwritten)',
                        default = True,
                        type    = bool
                        )
    args = parser.parse_args()


    ### Configuration ###
    selections  = ['ee', 'mumu', 'emu', 'mutau', 'etau', 'mu4j', 'e4j']
    #selections  = ['emu']
    do_data     = True
    do_mc       = True
    do_syst     = False
    period      = 2016

    # configure datasets to run over
    data_labels  = ['muon', 'electron']
    mc_labels    = ['ttbar', 'zjets_alt', 'diboson', 'ww', 't', 'wjets']

    dataset_list = []
    if do_data:
        dataset_list.extend(d for l in data_labels for d in pt.dataset_dict[l])
    if do_mc:
        dataset_list.extend(d for l in mc_labels for d in pt.dataset_dict[l])

    # for ttbar systematics
    if do_syst:
        dataset_list = [
                        'ttbar_inclusive_isrup', 'ttbar_inclusive_isrdown',
                        'ttbar_inclusive_fsrup', 'ttbar_inclusive_fsrdown',
                        'ttbar_inclusive_hdampup', 'ttbar_inclusive_hdampdown',
                        'ttbar_inclusive_tuneup', 'ttbar_inclusive_tunedown',
                        #'ttbar_inclusive_herwig'
                        ]

    #dataset_list = [
    #                #'zjets_m-50_alt', 'zjets_m-50', 
    #                #'z1jets_m-50', 'z2jets_m-50', 'z3jets_m-50', 'z4jets_m-50',
    #                #'ttbar_lep', 
    #                #'muon_2016H'
    #                ]

    ### Initialize multiprocessing queue and processes
    pool = Pool(processes = min(12, args.nprocesses))
    for selection in selections:
        output_path = f'{args.output}/{selection}_{period}'
        pt.make_directory(output_path, clear=(not args.append))
        event_count = {}
        for dataset in dataset_list:

            # get the root file and check that the tree exists
            root_file = r.TFile(args.input)
            if not root_file.Get(selection).GetListOfKeys().Contains(f'bltTree_{dataset}'):
                continue

            # get the tree and make sure that it's not empty
            tree_name = f'{selection}/bltTree_{dataset}'
            tree      = root_file.Get(f'{selection}/bltTree_{dataset}')
            n_entries  = tree.GetEntriesFast()
            if n_entries == 0: 
                continue

            # get event counts
            ecount = root_file.Get(f'TotalEvents_{dataset}')
            if ecount:
                event_count[dataset] = [ecount.GetBinContent(i+1) for i in range(ecount.GetNbinsX())]
                event_count[dataset][4] = n_entries
            else:
                print(f'Could not find dataset {dataset} in root file...')
                continue
            root_file.Close()

            # split dataset up according to configuration
            if dataset.split('_')[0] not in ['electron', 'muon']:
                max_event = min(n_entries, args.nmax)
            else:
                max_event = n_entries

            event_ranges = [i for i in range(0, max_event, args.nevents)]
            if event_ranges[-1] < max_event:
                event_ranges.append(max_event)

            # start pool process
            tmp_path = f'{output_path}/{dataset}'
            pt.make_directory(tmp_path, clear=True)
            for i, ievt in enumerate(event_ranges[:-1]):
                result = pool.apply_async(pickle_ntuple, 
                                          args = (args.input, tree_name, tmp_path, 
                                                  (ievt, event_ranges[i+1]), (i, len(event_ranges)-1)
                                                 )
                                          )
            #print(result.get(timeout=1))

        # special case: fakes
        if selection in ['mutau', 'mu4j', 'etau', 'e4j'] and do_data:
            for dataset in [d for l in data_labels for d in pt.dataset_dict[l]]:
                root_file = r.TFile(args.input)
                tree_name = f'{selection}_fakes/bltTree_{dataset}'
                tree      = root_file.Get(tree_name)
                n_entries = tree.GetEntriesFast()
                event_count[f'{dataset}_fakes'] = 10*[1.,]
                if n_entries == 0: 
                    continue
                root_file.Close()

                # split dataset up according to configuration
                if dataset.split('_')[0] not in ['electron', 'muon']:
                    max_event = min(n_entries, args.nmax)
                else:
                    max_event = n_entries

                event_ranges = [i for i in range(0, max_event, args.nevents)]
                if event_ranges[-1] < max_event:
                    event_ranges.append(max_event)

                # start pool process
                tmp_path = f'{output_path}/{dataset}_fakes'
                pt.make_directory(tmp_path, clear=True)
                for i, ievt in enumerate(event_ranges[:-1]):
                    result = pool.apply_async(pickle_ntuple, 
                                              args = (args.input, tree_name, tmp_path, 
                                                      (ievt, event_ranges[i+1]), (i, len(event_ranges)-1)
                                                     )
                                              )

        df_ecounts = pd.DataFrame(event_count)
        file_name = f'{output_path}/event_counts.csv'
        if args.append and os.path.isfile(file_name):
            df_ecounts_old = pd.read_csv(file_name)
            df_ecounts_old = df_ecounts_old.drop([c for c in df_ecounts_old.columns if c in df_ecounts.columns], axis=1)
            df_ecounts = pd.concat([df_ecounts, df_ecounts_old], axis=1)
            df_ecounts.to_csv(file_name)
        else:
            df_ecounts.to_csv(file_name)

    pool.close()
    pool.join()

    # concatenate pickle files when everything is done
    for selection in tqdm(selections, 
                          desc     = 'concatenating input files...',
                          total    = len(selections),
                          position = args.nprocesses+1
                          ):
        input_path = f'{args.output}/{selection}_{period}'
        for dataset in os.listdir(input_path):
            dataset_path = f'{input_path}/{dataset}'
            if not os.path.isdir(dataset_path):
                continue

            file_list = os.listdir(dataset_path)
            if len(file_list) > 0: 
                df_concat = pd.concat([pd.read_pickle(f'{dataset_path}/{filename}') for filename in file_list])
                df_concat.to_pickle(f'{input_path}/ntuple_{dataset}.pkl')
        
            shutil.rmtree(dataset_path) 
