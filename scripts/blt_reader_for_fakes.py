#!/usr/bin/env python

import os, sys
from copy import deepcopy
import multiprocessing as mp

import numpy as np
import pandas as pd
import ROOT as r

from tqdm import tqdm, trange

'''
Script for getting blt data out of ROOT files and into CSV format.
'''

def make_directory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)

    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')

def fill_event_vars(tree):

    out_dict = dict(
                    run_number     = tree.runNumber,
                    event_number   = tree.evtNumber,
                    lumi           = tree.lumiSection,
                    trigger_status = tree.triggerStatus,
 
                    n_pu           = tree.nPU,
                    n_pv           = tree.nPV,
                    n_muons        = tree.nMuons,
                    n_electrons    = tree.nElectrons,
                    n_jets         = tree.nJets,
                    n_fwdjets      = tree.nFwdJets,
                    n_bjets        = tree.nBJets,
 
                    met_mag        = tree.met,
                    met_phi        = tree.metPhi,
                    ht_mag         = tree.ht,
                    ht_phi         = tree.htPhi,
 
                    event_weight        = tree.eventWeight
                   )

    #if dataset in ['zjets_m-50', 'zjets_m-10to50'] and 0 < tree.nPartons < 5:
    #    out_dict['weight'] = 0.
    #else:
    #    out_dict['weight'] = tree.eventWeight
    out_dict['weight'] = tree.eventWeight

    return out_dict

def fill_dilepton_vars(tree):

    lep1, lep2, lep3 = tree.leptonOneP4, tree.leptonTwoP4, tree.leptonThreeP4
    dilepton1 = lep1 + lep2
    trilepton = dilepton1 + lep3

    out_dict = dict(
                    lepton1_pt      = lep1.Pt(),
                    lepton1_eta     = lep1.Eta(),
                    lepton1_phi     = lep1.Phi(),
                    lepton1_d0      = tree.leptonOneD0,
                    lepton1_dz      = tree.leptonOneDZ,
                    lepton1_q       = np.sign(tree.leptonOneFlavor),
                    lepton1_flavor  = np.abs(tree.leptonOneFlavor),
                    lepton1_iso     = tree.leptonOneIso,
                    lepton1_reliso  = tree.leptonOneIso/lep1.Pt(),
                    lepton1_mother  = tree.leptonOneMother,
 
                    lepton2_pt      = lep2.Pt(),
                    lepton2_eta     = lep2.Eta(),
                    lepton2_phi     = lep2.Phi(),
                    lepton2_d0      = tree.leptonTwoD0,
                    lepton2_dz      = tree.leptonTwoDZ,
                    lepton2_q       = np.sign(tree.leptonTwoFlavor),
                    lepton2_flavor  = np.abs(tree.leptonTwoFlavor),
                    lepton2_iso     = tree.leptonTwoIso,
                    lepton2_reliso  = tree.leptonTwoIso/lep2.Pt(),
                    lepton2_mother  = tree.leptonTwoMother,

                    lepton3_pt      = lep3.Pt(),
                    lepton3_eta     = lep3.Eta(),
                    lepton3_phi     = lep3.Phi(),
                    lepton3_d0      = tree.leptonThreeD0,
                    lepton3_dz      = tree.leptonThreeDZ,
                    lepton3_q       = np.sign(tree.leptonThreeFlavor),
                    lepton3_flavor  = np.abs(tree.leptonThreeFlavor),
                    lepton3_iso     = tree.leptonThreeIso,
                    lepton3_reliso  = tree.leptonThreeIso/lep2.Pt(),

                    dilepton1_delta_eta = abs(lep1.Eta() - lep2.Eta()),
                    dilepton1_delta_phi = abs(lep1.DeltaPhi(lep2)),
                    dilepton1_delta_r   = lep1.DeltaR(lep2),
                    dilepton1_mass      = dilepton1.M(),
                    dilepton1_pt        = dilepton1.Pt(),
                    dilepton1_eta       = dilepton1.Eta(),
                    dilepton1_phi       = dilepton1.Phi(),
                    dilepton1_pt_over_m = dilepton1.Pt()/dilepton1.M(),
                    dilepton1_pt_diff   = (lep1.Pt() - lep2.Pt()),
                    dilepton1_pt_asym   = (lep1.Pt() - lep2.Pt())/(lep1.Pt() + lep2.Pt()),

                    dilepton_probe_mass      = trilepton.M(),
                    dilepton_probe_delta_eta = abs(dilepton1.Eta() - lep3.Eta()),
                    dilepton_probe_delta_phi = abs(dilepton1.DeltaPhi(lep3)),
                    dilepton_probe_delta_r   = dilepton1.DeltaR(lep3),
                    dilepton_probe_pt_asym   = (dilepton1.Pt() - lep3.Pt())/(dilepton1.Pt() + lep3.Pt())
                   )

    # calculate vertex information
    #pv, sv = tree.rPV, tree.rDimuon
    #lxy, txy = calculate_trans_lifetime(dilepton, pv, sv)
    #out_dict['lxy'] = lxy
    #out_dict['txy'] = txy

    return out_dict


def fill_jet_vars(tree):
    # aliases
    jet1, jet2 = tree.jetOneP4, tree.jetTwoP4
    dijet      = jet1 + jet2

    out_dict = dict(
                    jet1_pt         = jet1.Pt(),
                    jet1_eta        = jet1.Eta(),
                    jet1_phi        = jet1.Phi(),
                    jet1_e          = jet1.E(),
                    jet1_tag        = tree.jetOneTag,
                    #jet1_flavor     = tree.jetOneFlavor,
 
                    jet2_pt         = jet2.Pt(),
                    jet2_eta        = jet2.Eta(),
                    jet2_phi        = jet2.Phi(),
                    jet2_e          = jet2.E(),
                    jet2_tag        = tree.jetTwoTag,
                    #jet2_flavor     = tree.jetTwoFlavor,
 
                    jet_delta_eta   = abs(jet1.Eta() - jet2.Eta()),
                    jet_delta_phi   = abs(jet1.DeltaPhi(jet2)),
                    jet_delta_r     = jet1.DeltaR(jet2),
 
                    dijet_mass      = dijet.M(),
                    dijet_pt        = dijet.Pt(),
                    dijet_eta       = dijet.Eta(),
                    dijet_phi       = dijet.Phi(),
                    dijet_pt_over_m = dijet.Pt()/dijet.M() if dijet.M() > 0 else -1,
                   )

    return out_dict

def fill_tau_vars(tree):
    
    out_dict = dict(
                    #tau_charged    = tree.tauChHadMult,
                    #tau_photon     = tree.tauPhotonMult,
                    tau_decay_mode = tree.tauDecayMode,
                    tau_mva        = tree.tauMVA,
                   )

    return out_dict

def fill_jet_lepton_vars(tree):

    # aliases
    lep1, lep2  = tree.leptonOneP4, tree.leptonTwoP4
    lep3 = tree.leptonThreeP4
    jet1, jet2  = tree.jetOneP4, tree.jetTwoP4
    dilepton    = lep1 + lep2
    dijet       = jet1 + jet2
    lepton1_j1  = lep1 + jet1
    lepton2_j1  = lep2 + jet1
    lepton1_j2  = lep1 + jet2
    lepton2_j2  = lep2 + jet2
    dilepton_j1 = dilepton + jet1
    dilepton_j2 = dilepton + jet2
    fourbody    = dilepton + dijet

    out_dict = dict(
                    lepton1_j1_mass       = lepton1_j1.M(),
                    lepton1_j1_pt         = lepton1_j1.Pt(),
                    lepton1_j1_delta_eta  = abs(lep1.Eta() - jet1.Eta()),
                    lepton1_j1_delta_phi  = abs(lep1.DeltaPhi(jet1)),
                    lepton1_j1_delta_r    = lep1.DeltaR(jet1),

                    lepton2_j1_mass       = lepton2_j1.M(),
                    lepton2_j1_pt         = lepton2_j1.Pt(),
                    lepton2_j1_delta_eta  = abs(lep2.Eta() - jet1.Eta()),
                    lepton2_j1_delta_phi  = abs(lep2.DeltaPhi(jet1)),
                    lepton2_j1_delta_r    = lep2.DeltaR(jet1),

                    lepton1_j2_mass       = lepton1_j2.M(),
                    lepton1_j2_pt         = lepton1_j2.Pt(),
                    lepton1_j2_delta_eta  = abs(lep1.Eta() - jet2.Eta()),
                    lepton1_j2_delta_phi  = abs(lep1.DeltaPhi(jet2)),
                    lepton1_j2_delta_r    = lep1.DeltaR(jet2),

                    lepton2_j2_mass       = lepton2_j2.M(),
                    lepton2_j2_pt         = lepton2_j2.Pt(),
                    lepton2_j2_delta_eta  = abs(lep2.Eta() - jet2.Eta()),
                    lepton2_j2_delta_phi  = abs(lep2.DeltaPhi(jet2)),
                    lepton2_j2_delta_r    = lep2.DeltaR(jet2),

                    dilepton_j1_mass      = dilepton_j1.M(),
                    dilepton_j1_pt        = dilepton_j1.Pt(),
                    dilepton_j1_delta_eta = abs(dilepton.Eta() - jet1.Eta()),
                    dilepton_j1_delta_phi = abs(dilepton.DeltaPhi(jet1)),
                    dilepton_j1_delta_r   = dilepton.DeltaR(jet1),

                    dilepton_j2_mass      = dilepton_j2.M(),
                    dilepton_j2_pt        = dilepton_j2.Pt(),
                    dilepton_j2_delta_eta = abs(dilepton.Eta() - jet2.Eta()),
                    dilepton_j2_delta_phi = abs(dilepton.DeltaPhi(jet2)),
                    dilepton_j2_delta_r   = dilepton.DeltaR(jet2),

                    # four body variables
                    four_body_mass       = fourbody.M(),
                    four_body_delta_eta  = abs(dijet.Eta() - dilepton.Eta()),
                    four_body_delta_phi  = abs(dijet.DeltaPhi(dilepton)),
                    four_body_delta_r    = dijet.DeltaR(dilepton),

                    mumuj1_j2_delta_eta  = abs(dilepton_j1.Eta() - jet2.Eta()),
                    mumuj1_j2_delta_phi  = abs(dilepton_j1.DeltaPhi(jet2)),
                    mumuj1_j2_delta_r    = dilepton_j1.DeltaR(jet2),
                    mumuj2_j1_delta_eta  = abs(dilepton_j2.Eta() - jet1.Eta()),
                    mumuj2_j1_delta_phi  = abs(dilepton_j2.DeltaPhi(jet1)),
                    mumuj2_j1_delta_r    = dilepton_j2.DeltaR(jet1),
                   )


    #if tree.nBJets > 0:
    #    if tree.leptonOneFlavor < 0:
    #        out_dict['lepton_minus_cos_theta'] = calculate_cos_theta(dilepton_j1, dilepton, lep1)
    #        out_dict['lepton_plus_cos_theta']  = calculate_cos_theta(dilepton_j1, dilepton, lep2)
    #    else:
    #        out_dict['lepton_minus_cos_theta'] = calculate_cos_theta(dilepton_j1, dilepton, lep2)
    #        out_dict['lepton_plus_cos_theta']  = calculate_cos_theta(dilepton_j1, dilepton, lep1)
    #else:
    #    out_dict['lepton_plus_cos_theta']  = 0.
    #    out_dict['lepton_minus_cos_theta'] = 0.

    return out_dict

def fill_gen_particle_vars(tree):
    # aliases
    gen1, gen2 = tree.genOneP4, tree.genTwoP4
    digen = gen1 + gen2
    out_dict = dict(
                    gen_cat   = tree.genCategory,
                    gen1_pt   = gen1.Pt(),
                    gen1_eta  = gen1.Eta(),
                    gen1_phi  = gen1.Phi(),
                    gen1_mass = gen1.M(),
                    gen1_id   = tree.genOneId,
                    gen2_pt   = gen2.Pt(),
                    gen2_eta  = gen2.Eta(),
                    gen2_phi  = gen2.Phi(),
                    gen2_mass = gen2.M(),
                    gen2_id   = tree.genTwoId,

                    digen_delta_eta = abs(gen1.Eta() - gen2.Eta()),
                    digen_delta_phi = abs(gen1.DeltaPhi(gen2)),
                    digen_delta_r   = gen1.DeltaR(gen2),
                    digen_mass      = digen.M(),
                    digen_pt        = digen.Pt(),
                    digen_eta       = digen.Eta(),
                    digen_phi       = digen.Phi(),
                   )

    return out_dict


def fill_ntuple(tree, name, selection):
    n = tree.GetEntriesFast()
    for i in range(n,):
        tree.GetEntry(i)
        entry = {}
        entry.update(fill_event_vars(tree))
        entry.update(fill_jet_vars(tree))
        entry.update(fill_jet_lepton_vars(tree))
        entry.update(fill_dilepton_vars(tree))

        entry.update(fill_gen_particle_vars(tree))

        yield entry

def pickle_ntuple(tree, dataset_name, output_path, selection):

    # get the tree, convert to dataframe, and save df to pickle
    ntuple = fill_ntuple(tree, dataset_name, selection)
    df     = pd.DataFrame(ntuple)
    #df     = df.query('weight != 0')
    df.to_pickle('{0}/ntuple_{1}.pkl'.format(output_path, dataset_name))

    print(f'{dataset_name} pickled successfully')

if __name__ == '__main__':

    ### Configuration ###
    if len(sys.argv) > 1:
        selection     = sys.argv[1]
    else:
        selection     = 'mumu'

    selection   = ['fakes']
    do_mc       = True
    do_data     = True
    period      = 2016
    infile      = f'data/output_fakes.root'

    #infile       = f'data/bltuples/output_btag_eff.root'
    #output_path  = f'data/flatuples/{selection}_btag_{period}'

    dataset_list = [
                    'muon_2016B', 'muon_2016C', 'muon_2016D', 
                    'muon_2016E', 'muon_2016F', 'muon_2016G', 'muon_2016H',
                    #'electron_2016B', 'electron_2016C', 'electron_2016D', 
                    #'electron_2016E', 'electron_2016F', 'electron_2016G', 'electron_2016H',
                    'ttbar_lep', 'ttbar_semilep',
                    #'w1jets', 'w2jets', 'w3jets', 'w4jets', 
                    'zjets_m-50', #'zjets_m-10to50',
                    #'z1jets_m-50', 'z1jets_m-10to50',
                    #'z2jets_m-50', 'z2jets_m-10to50',
                    #'z3jets_m-50', 'z3jets_m-10to50',
                    #'z4jets_m-50', 'z4jets_m-10to50',
                    'wz_3lnu', 'zz_4l'
                    ]


    ### Initialize multiprocessing queue and processes
    processes   = {}
    files_list  = [] # There needs to be multiple instances of the file to access each of the trees.  Not great...
    event_count = {}
    output_path = f'data/flatuples/fakes_{period}'
    make_directory(output_path, clear=True)
    for dataset in dataset_list:

        froot  = r.TFile(infile)
        files_list.append(froot)

        #ecount = froot.Get(f'{selection}/TotalEvents_{selection}_{dataset}')
        ecount = froot.Get(f'TotalEvents_{dataset}')
        if ecount:
            event_count[dataset] = [ecount.GetBinContent(i+1) for i in range(ecount.GetNbinsX())]
        else:
            print(f'Could not find dataset {dataset} in root file...')
            continue

        #tree = froot.Get(f'{selection}/bltTree_{dataset}')
        tree = froot.Get(f'muonFakes/bltTree_{dataset}')
        p = mp.Process(target=pickle_ntuple, args=(tree, dataset, output_path, selection))
        p.start()
        processes[dataset] = p

    fname = f'{output_path}/event_counts.csv'
    df = pd.DataFrame(event_count)
    df.to_csv(fname)

    for p in processes.values():
        p.join()

    for f in files_list:
        f.Close()
