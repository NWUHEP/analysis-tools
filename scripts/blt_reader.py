#!/usr/bin/env python

import os, sys

import numpy as np
import pandas as pd
import ROOT as r
import json
from collections import OrderedDict
from multiprocessing import Pool
from memory_profiler import profile

from tqdm import tqdm, trange

'''
Script for getting blt data out of ROOT files and into CSV format.
'''

def make_directory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)

    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')

def calculate_cos_theta(ref_p4, boost_p4, target_p4):
    '''
    Calculate cos(theta) of target in rest frame of boost relative with ref
    defining the x axis in boost's reference frame.

    Paramters:
    ==========
    ref_p4    : TLorentzVector of reference particle
    boost_p4  : TLorentzVector of particle defining the rest frame
    target_p4 : TLorentzVector of particle whose cos(theta) value we would like to calculate
    '''
                                                                                  
    boost = -1*boost_p4.BoostVector() 
    ref_p4.Boost(boost)  
    ref_v3 = ref_p4.Vect()                                        

    beam_axis = r.TLorentzVector(0, 0, 1, 1)
    beam_axis.Boost(boost)                                                        
    axis_z = (-1*ref_v3).Unit() 
    axis_y = beam_axis.Vect().Cross(ref_v3).Unit()               
    axis_x = axis_y.Cross(axis_z).Unit()                       

    rotation = r.TRotation()
    rotation = rotation.RotateAxes(axis_x, axis_y, axis_z).Inverse()    
                                                                                  
    target_p4.Boost(boost)                                                  
    target_p4.Transform(rotation) 
                                                                                  
    return target_p4.CosTheta() 

def fill_event_vars(tree):

    out_dict = {}
    out_dict['run_number']     = tree.runNumber
    out_dict['event_number']   = tree.evtNumber
    out_dict['lumi']           = tree.lumiSection
    out_dict['trigger_status'] = tree.triggerStatus

    out_dict['n_pu']           = tree.nPU
    out_dict['n_pv']           = tree.nPV
    out_dict['n_muons']        = tree.nMuons
    out_dict['n_electrons']    = tree.nElectrons
    out_dict['n_jets']         = tree.nJets
    out_dict['n_fwdjets']      = tree.nFwdJets
    out_dict['n_bjets']        = tree.nBJets

    out_dict['met_mag']        = tree.met
    out_dict['met_phi']        = tree.metPhi

    if dataset in ['zjets_m-50', 'zjets_m-10to50'] \
        and tree.nPartons > 0 \
        and tree.nPartons < 5:
        out_dict['weight'] = 0.
    else:
        out_dict['weight'] = tree.eventWeight

    return out_dict

def fill_lepton_vars(tree):
    # aliases
    lep1, lep2 = tree.leptonOneP4, tree.leptonTwoP4
    dilepton = lep1 + lep2

    out_dict = {}
    out_dict['lepton1_pt']      = lep1.Pt()
    out_dict['lepton1_eta']     = lep1.Eta()
    out_dict['lepton1_phi']     = lep1.Phi()
    out_dict['lepton1_q']       = tree.leptonOneQ
    out_dict['lepton1_iso']     = tree.leptonOneIso
    out_dict['lepton1_flavor']  = tree.leptonOneFlavor
    out_dict['lepton1_trigger'] = tree.leptonOneTrigger

    out_dict['lepton2_pt']      = lep2.Pt()
    out_dict['lepton2_eta']     = lep2.Eta()
    out_dict['lepton2_phi']     = lep2.Phi()
    out_dict['lepton2_q']       = tree.leptonTwoQ
    out_dict['lepton2_iso']     = tree.leptonTwoIso
    out_dict['lepton2_flavor']  = tree.leptonTwoFlavor
    out_dict['lepton2_trigger'] = tree.leptonTwoTrigger

    out_dict['lepton_delta_eta']   = abs(lep1.Eta() - lep2.Eta())
    out_dict['lepton_delta_phi']   = abs(lep1.DeltaPhi(lep2))
    out_dict['lepton_delta_r']     = lep1.DeltaR(lep2)

    out_dict['dilepton_mass']      = dilepton.M()
    out_dict['dilepton_pt']        = dilepton.Pt()
    out_dict['dilepton_eta']       = dilepton.Eta()
    out_dict['dilepton_phi']       = dilepton.Phi()
    out_dict['dilepton_pt_over_m'] = dilepton.Pt()/dilepton.M()

    return out_dict

def fill_jet_vars(tree):
    # aliases
    bjet, jet         = tree.bjetP4, tree.jetP4
    dijet             = jet + bjet

    out_dict = {}
    out_dict['bjet_pt']         = bjet.Pt()
    out_dict['bjet_eta']        = bjet.Eta()
    out_dict['bjet_phi']        = bjet.Phi()
    out_dict['bjet_e']          = bjet.E()
    out_dict['bjet_d0']         = tree.bjetD0
    out_dict['bjet_tag']        = tree.bjetTag
    #out_dict['bjet_puid']       = tree.bjetPUID
    out_dict['bjet_flavor']     = tree.bjetFlavor

    out_dict['jet_pt']          = jet.Pt()
    out_dict['jet_eta']         = jet.Eta()
    out_dict['jet_phi']         = jet.Phi()
    out_dict['jet_e']           = jet.E()
    out_dict['jet_d0']          = tree.jetD0
    out_dict['jet_tag']         = tree.jetTag
    #out_dict['jet_puid']        = tree.jetPUID
    out_dict['jet_flavor']      = tree.jetFlavor

    out_dict['jet_delta_eta']   = abs(bjet.Eta() - jet.Eta())
    out_dict['jet_delta_phi']   = abs(bjet.DeltaPhi(jet))
    out_dict['jet_delta_r']     = bjet.DeltaR(jet)

    out_dict['dijet_mass']      = dijet.M()
    out_dict['dijet_pt']        = dijet.Pt()
    out_dict['dijet_eta']       = dijet.Eta()
    out_dict['dijet_phi']       = dijet.Phi()
    out_dict['dijet_pt_over_m'] = dijet.Pt()/dijet.M() if dijet.M() > 0 else -1

    return out_dict

def fill_jet_lepton_vars(tree):

    # aliases
    lep1, lep2,       = tree.leptonOneP4, tree.leptonTwoP4
    bjet, jet         = tree.bjetP4, tree.jetP4
    dilepton          = lep1 + lep2
    dijet             = jet + bjet
    lepton1_b         = lep1 + bjet
    lepton2_b         = lep2 + bjet
    dilepton_b        = dilepton + bjet
    dilepton_j        = dilepton + jet
    fourbody          = dilepton + dijet

    out_dict = {}
    out_dict['lepton1_b_mass']       = lepton1_b.M()
    out_dict['lepton1_b_pt']         = lepton1_b.Pt()
    out_dict['lepton1_b_delta_eta']  = abs(lep1.Eta() - bjet.Eta())
    out_dict['lepton1_b_delta_phi']  = abs(lep1.DeltaPhi(bjet))
    out_dict['lepton1_b_delta_r']    = lep1.DeltaR(bjet)

    out_dict['lepton2_b_mass']       = lepton2_b.M()
    out_dict['lepton2_b_pt']         = lepton2_b.Pt()
    out_dict['lepton2_b_delta_eta']  = abs(lep2.Eta() - bjet.Eta())
    out_dict['lepton2_b_delta_phi']  = abs(lep2.DeltaPhi(bjet))
    out_dict['lepton2_b_delta_r']    = lep2.DeltaR(bjet)

    out_dict['dilepton_b_mass']      = dilepton_b.M()
    out_dict['dilepton_b_pt']        = dilepton_b.Pt()
    out_dict['dilepton_b_delta_eta'] = abs(dilepton.Eta() - bjet.Eta())
    out_dict['dilepton_b_delta_phi'] = abs(dilepton.DeltaPhi(bjet))
    out_dict['dilepton_b_delta_r']   = dilepton.DeltaR(bjet)

    out_dict['dilepton_j_mass']      = dilepton_j.M()
    out_dict['dilepton_j_pt']        = dilepton_j.Pt()
    out_dict['dilepton_j_delta_eta'] = abs(dilepton.Eta() - jet.Eta())
    out_dict['dilepton_j_delta_phi'] = abs(dilepton.DeltaPhi(jet))
    out_dict['dilepton_j_delta_r']   = dilepton.DeltaR(jet)

    # four body variables
    out_dict['four_body_mass']       = fourbody.M()
    out_dict['four_body_delta_eta']  = abs(dijet.Eta() - dilepton.Eta())
    out_dict['four_body_delta_phi']  = abs(dijet.DeltaPhi(dilepton))
    out_dict['four_body_delta_r']    = dijet.DeltaR(dilepton)

    out_dict['mumub_j_delta_eta']    = abs(dilepton_b.Eta() - jet.Eta())
    out_dict['mumub_j_delta_phi']    = abs(dilepton_b.DeltaPhi(jet))
    out_dict['mumub_j_delta_r']      = dilepton_b.DeltaR(jet)

    if tree.nBJets > 0:
        if tree.leptonOneQ == -1:
            out_dict['lepton_minus_cos_theta'] = calculate_cos_theta(dilepton_b, dilepton, lep1)
            out_dict['lepton_plus_cos_theta']  = calculate_cos_theta(dilepton_b, dilepton, lep2)
        else:
            out_dict['lepton_minus_cos_theta'] = calculate_cos_theta(dilepton_b, dilepton, lep2)
            out_dict['lepton_plus_cos_theta']  = calculate_cos_theta(dilepton_b, dilepton, lep1)
    else:
        out_dict['lepton_plus_cos_theta']  = 0.
        out_dict['lepton_minus_cos_theta'] = 0.

    return out_dict

def fill_genjet_vars(out_dict):

    out_dict = {}
    # generator level information (all 0 for data)
    gen_bjet = tree.genBJetP4
    out_dict['n_partons']            = (tree.nPartons)
    out_dict['gen_bjet_pt']          = (gen_bjet.Pt())
    out_dict['gen_bjet_eta']         = (gen_bjet.Eta())
    out_dict['gen_bjet_phi']         = (gen_bjet.Phi())
    out_dict['gen_bjet_e']           = (gen_bjet.E())
    out_dict['gen_bjet_tag']         = (tree.genBJetTag)
    #out_dict['gen_dilepton_b_mass'] = ((dilepton + gen_bjet).M())
    
    #out_dict['gen_jet_pt']=(gen_jet.Pt())
    #out_dict['gen_jet_eta']=(gen_jet.Eta())
    #out_dict['gen_jet_phi']=(gen_jet.Phi())
    #out_dict['gen_jet_e']=(gen_bjet.E())
    #out_dict['gen_jet_tag']=(tree.genJetTag)
    #out_dict['gen_dilepton_j_mass']=((dilepton + gen_jet).M())

    return out_dict

@profile
def fill_ntuple(tree, name):
    n = int(np.min([float(tree.GetEntriesFast()), 5e5]))
    for i in trange(n,
                    desc       = name,
                    leave      = True,
                    unit_scale = True,
                    ncols      = 75,
                    total      = n
        ):
        tree.GetEntry(i)
        entry = {}
        entry.update(fill_event_vars(tree))
        entry.update(fill_lepton_vars(tree))
        entry.update(fill_jet_vars(tree))
        entry.update(fill_jet_lepton_vars(tree))
        entry.update(fill_genjet_vars(tree))
        n -= 1;

        yield entry

def pickle_ntuple(ntuple_data):
    # unpack input data
    name        = ntuple_data[0]
    input_file  = ntuple_data[1]
    output_path = ntuple_data[2]

    # get the tree, convert to dataframe, and save df to pickle
    tree   = froot.Get('tree_{0}'.format(name))
    ntuple = fill_ntuple(tree, name)
    df     = pd.DataFrame(ntuple)
    df.to_pickle('{0}/ntuple_{1}.pkl'.format(output_path, name))

if __name__ == '__main__':

    ### Configuration ###
    selection    = 'mumu'
    period       = 2016
    infile       = 'data/bltuples/output_{0}_{1}.root'.format(selection, period)
    output_path  = 'data/flatuples/{0}_{1}'.format(selection, period)

    if period == 2016:
        dataset_list = [
                        #'muon_2016C',
                        'muon_2016B', 'muon_2016C', 'muon_2016D', 
                        'muon_2016E', 'muon_2016F', 'muon_2016G', 'muon_2016H',

                        #'bprime_xb',
                        'ttbar_lep', #'ttbar_semilep',
                        'zjets_m-50', 'zjets_m-10to50',
                        'z1jets_m-50', 'z1jets_m-10to50',
                        'z2jets_m-50', 'z2jets_m-10to50',
                        'z3jets_m-50', 'z3jets_m-10to50',
                        'z4jets_m-50', 'z4jets_m-10to50',
                        't_t', 'tbar_t', 't_tw', 'tbar_tw', #'t_s', 'tbar_s'
                        'ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q', #'zz_2l2nu',

                        ]
    elif period == 2012:
        dataset_list = [
                        'muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D', 
                        #'electron_2012A', 'electron_2012B', 'electron_2012C', 'electron_2012D', 
                        'ttbar_lep', 'ttbar_semilep',
                        'zjets_m-50', 'zjets_m-10to50',
                        'z1jets_m-50', 'z1jets_m-10to50',
                        'z2jets_m-50', 'z2jets_m-10to50',
                        'z3jets_m-50', 'z3jets_m-10to50',
                        'z4jets_m-50', 'z4jets_m-10to50',
                        't_s', 't_t', 't_tw', 'tbar_s', 'tbar_t', 'tbar_tw', 
                        'ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q', 'zz_2l2nu',
                        'bprime_bb_semilep', 'bprime_t-channel', 
                        'fcnc_s-channel', 'fcnc_tt_semilep'
                       ]
    make_directory(output_path, clear=True)

    ### Get input bltuple ###
    print 'Opening file {0}'.format(infile)
    froot  = r.TFile(infile)
    #pool = Pool(processes=4)
    #pool.map(pickle_ntuple, [(d, froot, output_path) for d in dataset_list])

    event_count = {}
    for dataset in tqdm(dataset_list, 
                        desc       = 'Unpacking',
                        unit_scale = True,
                        ncols      = 75,
                        total      = len(dataset_list)
                       ):
        ecount = froot.Get('TotalEvents_{0}'.format(dataset))
        if ecount:
            event_count[dataset] = [ecount.GetBinContent(i+1) for i in range(ecount.GetNbinsX())]
        else:
            print 'Could not find dataset {0} in root file...'.format(dataset)
            continue

        tree   = froot.Get('tree_{0}'.format(dataset))
        ntuple = fill_ntuple(tree, dataset)

        df = pd.DataFrame(ntuple)
        df.to_pickle('{0}/ntuple_{1}.pkl'.format(output_path, dataset))

    fname = '{0}/event_counts.csv'.format(output_path)
    df = pd.DataFrame(event_count)
    df.to_csv(fname)
