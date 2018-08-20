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
    ntuple = br.fill_ntuple(tree, selection, dataset, event_range, 
                            job_id = (process._identity[0], ix[0], ix[1])
                            )
    df     = pd.DataFrame(ntuple)
    df     = df.query('weight != 0') # remove deadweight
    df.to_pickle(f'{output_path}/{dataset}_{ix[0]}.pkl')
    del df
    root_file.Close()

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
                        help='number of concurrent processes (will be less than number of available cores)',
                        default=8,
                        type=int
                        )
    parser.add_argument('-n', '--nevents',
                        help='number of events to run per process',
                        default=100000,
                        type=int
                        )
    args = parser.parse_args()


    ### Configuration ###
    selections  = ['mumu', 'ee']#, 'emu', 'mutau', 'etau', 'mu4j', 'e4j']
    do_data     = True
    do_mc       = True
    do_syst     = False
    period      = 2016

    # configure datasets to run over
    dataset_list = []
    data_labels  = ['muon']#, 'electron']
    #mc_labels    = ['zjets', 'ttbar', 'diboson', 't', 'wjets']
    mc_labels    = ['zjets_alt', 'zjets']
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

    dataset_list.append('ttbar_lep')

    ### Initialize multiprocessing queue and processes
    pool = Pool(processes = min(12, args.nprocesses))
    for selection in selections:
        output_path = f'{args.output}/{selection}_{period}'
        pt.make_directory(output_path, clear=True)
        event_count = {}
        for dataset in dataset_list:
            root_file = r.TFile(args.input)
            tree_name = f'{selection}/bltTree_{dataset}'
            tree      = root_file.Get(f'{selection}/bltTree_{dataset}')
            n_entries  = tree.GetEntriesFast()

            ecount    = root_file.Get(f'TotalEvents_{dataset}')
            if ecount:
                event_count[dataset] = [ecount.GetBinContent(i+1) for i in range(ecount.GetNbinsX())]
            else:
                print(f'Could not find dataset {dataset} in root file...')
                continue
            root_file.Close()

            # start pool process
            event_ranges = [i for i in range(0, n_entries, args.nevents)]
            event_ranges[-1] = n_entries
            tmp_path = f'{output_path}/{dataset}'
            pt.make_directory(tmp_path, clear=True)
            for i, ievt in enumerate(event_ranges[:-1]):
                result = pool.apply_async(pickle_ntuple, 
                              args=(args.input, tree_name, tmp_path, (ievt, event_ranges[i+1]), (i, len(event_ranges)-1))
                                          )
            #print(result.get(timeout=1))
            
        df = pd.DataFrame(event_count)
        df.to_csv(f'{output_path}/event_counts.csv')

    pool.close()
    pool.join()

    # concatenate pickle files when everything is done
    print('Concatenating output files')
    for selection, dataset in tqdm(product(selections, dataset_list)):
        input_path = f'{args.output}/{selection}_{period}/{dataset}'
        file_list = os.listdir(input_path)
        if len(file_list) == 0: 
            continue
        df_concat = pd.concat([pd.read_pickle(f'{input_path}/{filename}') for filename in file_list])
        df_concat.to_pickle(f'{args.output}/{selection}_{period}/ntuple_{dataset}.pkl')
        
        shutil.rmtree(input_path) 
