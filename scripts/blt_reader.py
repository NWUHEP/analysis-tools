#!/usr/bin/env python

import os, sys

import numpy as np
import pandas as pd
import ROOT as r
import json

'''
Script for getting blt data out of ROOT files and into CSV format.
'''

def make_directory(filePath, clear=True):
    if not os.path.exists(filePath):
        os.system('mkdir -p '+filePath)

    if clear and len(os.listdir(filePath)) != 0:
        os.system('rm '+filePath+'/*')

if __name__ == '__main__':

    ### Configuration ###
    selection    = 'ee'
    period       = 2012
    infile       = 'data/bltuples/output_{0}_{1}.root'.format(selection, period)
    output_path  = 'data/flatuples/{0}_{1}'.format(selection, period)
    dataset_list = [
                    #'muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D', 
                    'electron_2012A', 'electron_2012B', 'electron_2012C', 'electron_2012D', 
                    'ttbar_lep', 'ttbar_semilep',
                    'zjets_m-50', 'zjets_m-10to50',
                    't_s', 't_t', 't_tw', 'tbar_s', 'tbar_t', 'tbar_tw', 
                    'ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q', 'zz_2l2nu',
                    'bprime_xb'
                    ]
    features = [
               'run_number', 'event_number', 'lumi', 'weight',
               'lepton1_pt', 'lepton1_eta', 'lepton1_phi', 
               'lepton1_iso', 'lepton1_q', 'lepton1_flavor',
               'lepton2_pt', 'lepton2_eta', 'lepton2_phi',  
               'lepton2_iso', 'lepton2_q', 'lepton2_flavor',
               'lepton_delta_eta', 'lepton_delta_phi', 'lepton_delta_r',
               'dilepton_mass', 'dilepton_pt', 'dilepton_eta', 'dilepton_phi', 
               'dilepton_pt_over_m',

               'met_mag', 'met_phi',
               'n_jets', 'n_fwdjets', 'n_bjets',
               'bjet_pt', 'bjet_eta', 'bjet_phi', 'bjet_d0',
               'jet_pt', 'jet_eta', 'jet_phi', 'jet_d0', 
               'dijet_mass', 'dijet_pt', 'dijet_eta', 'dijet_phi', 
               'dijet_pt_over_m',

               'dilepton_j_mass', 'dilepton_j_pt', 
               'dilepton_j_delta_eta', 'dilepton_j_delta_phi', 'dilepton_j_delta_r',
               'dilepton_b_mass', 'dilepton_b_pt', 
               'dilepton_b_delta_eta', 'dilepton_b_delta_phi', 'dilepton_b_delta_r',
               'four_body_mass',
               'four_body_delta_phi', 'four_body_delta_eta', 'four_body_delta_r',
               ]
    make_directory(output_path)

    ### Get input bltuple ###
    print 'Opening file {0}'.format(infile)
    froot  = r.TFile(infile)

    event_count = {}
    for dataset in dataset_list:
        ecount  = froot.Get('TotalEvents_{0}'.format(dataset))
        event_count[dataset] = [ecount.GetBinContent(i+1) for i in range(ecount.GetNbinsX())]

        tree    = froot.Get('tree_{0}'.format(dataset))
        n       = tree.GetEntriesFast()
        ntuple = {f:[] for f in features}

        print 'Reading {0} with {1} events...'.format(dataset, n)
        for i in xrange(n):
            tree.GetEntry(i)

            # get and build physics objects
            lep1, lep2, bjet, jet = tree.leptonOneP4, tree.leptonTwoP4, tree.bjetP4, tree.jetP4
            met, met_phi = tree.met, tree.metPhi
            dilepton     = lep1 + lep2
            dijet        = jet + bjet
            fourbody     = dilepton + dijet
            dilepton_b   = dilepton + bjet
            dilepton_j   = dilepton + jet

            # event info
            ntuple['run_number'].append(tree.runNumber)
            ntuple['event_number'].append(tree.evtNumber)
            ntuple['lumi'].append(tree.lumiSection)
            ntuple['weight'].append(tree.eventWeight)

            ### lepton
            ntuple['lepton1_pt'].append(lep1.Pt())
            ntuple['lepton1_eta'].append(lep1.Eta())
            ntuple['lepton1_phi'].append(lep1.Phi())
            ntuple['lepton1_q'].append(tree.leptonOneQ)
            ntuple['lepton1_iso'].append(tree.leptonOneIso)
            ntuple['lepton1_flavor'].append(tree.leptonOneFlavor)
            ntuple['lepton2_pt'].append(lep2.Pt())
            ntuple['lepton2_eta'].append(lep2.Eta())
            ntuple['lepton2_phi'].append(lep2.Phi())
            ntuple['lepton2_q'].append(tree.leptonTwoQ)
            ntuple['lepton2_iso'].append(tree.leptonTwoIso)
            ntuple['lepton2_flavor'].append(tree.leptonTwoFlavor)
            ntuple['lepton_delta_eta'].append(abs(lep1.Eta() - lep2.Eta()))
            ntuple['lepton_delta_phi'].append(abs(lep1.DeltaPhi(lep2)))
            ntuple['lepton_delta_r'].append(lep1.DeltaR(lep2))

            ### dilepton 
            ntuple['dilepton_mass'].append(dilepton.M())
            ntuple['dilepton_pt'].append(dilepton.Pt())
            ntuple['dilepton_eta'].append(dilepton.Eta())
            ntuple['dilepton_phi'].append(dilepton.Phi())
            ntuple['dilepton_pt_over_m'].append(dilepton.Pt()/dilepton.M())

            # jets
            ntuple['bjet_pt'].append(bjet.Pt())
            ntuple['bjet_eta'].append(bjet.Eta())
            ntuple['bjet_phi'].append(bjet.Phi())
            ntuple['bjet_d0'].append(tree.bjetD0)
            ntuple['jet_pt'].append(jet.Pt())
            ntuple['jet_eta'].append(jet.Eta())
            ntuple['jet_phi'].append(jet.Phi())
            ntuple['jet_d0'].append(tree.jetD0)
            ntuple['n_jets'].append(tree.nJets)
            ntuple['n_fwdjets'].append(tree.nFwdJets)
            ntuple['n_bjets'].append(tree.nBJets)

            # MET
            ntuple['met_mag'].append(met)
            ntuple['met_phi'].append(met_phi)

            ntuple['dijet_mass'].append(dijet.M())
            ntuple['dijet_pt'].append(dijet.Pt())
            ntuple['dijet_eta'].append(dijet.Eta())
            ntuple['dijet_phi'].append(dijet.Phi())
            ntuple['dijet_pt_over_m'].append(dijet.Pt()/dijet.M() if dijet.M() > 0 else -1)

            # dilepton + b jet variables
            ntuple['dilepton_b_mass'].append(dilepton_b.M())
            ntuple['dilepton_b_pt'].append(dilepton_b.Pt())
            ntuple['dilepton_b_delta_eta'].append(abs(dilepton.Eta() - bjet.Eta()))
            ntuple['dilepton_b_delta_phi'].append(abs(dilepton.DeltaPhi(bjet)))
            ntuple['dilepton_b_delta_r'].append(dilepton.DeltaR(bjet))

            # dilepton + b jet variables
            ntuple['dilepton_j_mass'].append(dilepton_j.M())
            ntuple['dilepton_j_pt'].append(dilepton_j.Pt())
            ntuple['dilepton_j_delta_eta'].append(abs(dilepton.Eta() - jet.Eta()))
            ntuple['dilepton_j_delta_phi'].append(abs(dilepton.DeltaPhi(jet)))
            ntuple['dilepton_j_delta_r'].append(dilepton.DeltaR(jet))

            # four body variables
            ntuple['four_body_mass'].append(fourbody.M())
            ntuple['four_body_delta_eta'].append(abs(dijet.Eta() - dilepton.Eta()))
            ntuple['four_body_delta_phi'].append(abs(dijet.DeltaPhi(dilepton)))
            ntuple['four_body_delta_r'].append(dijet.DeltaR(dilepton))

        df = pd.DataFrame(ntuple)
        df.to_csv('{0}/ntuple_{1}.csv'.format(output_path, dataset), index=False)

    df = pd.DataFrame(event_count)
    df.to_csv('{0}/event_counts.csv'.format(output_path))
