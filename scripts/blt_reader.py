#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import ROOT as r
import json

'''
Script for getting blt data out of ROOT files and into CSV format.
'''

if __name__ == '__main__':

    ### Configuration ###
    infile       = 'data/bltuples/output_mumu_2012.root'
    selection    = 'mumu'
    period       = 2012
    dataset_list = [
                    'muon_2012A', 'muon_2012B', 'muon_2012C', 'muon_2012D', 
                    'ttbar_lep', 
                    'zjets_m-50', 'zjets_m-10to50',
                    't_s', 't_t', 't_tw', 'tbar_s', 'tbar_t', 'tbar_tw', 
                    'ww', 'wz_2l2q', 'wz_3lnu', 'zz_2l2q', 'zz_2l2nu'
                    ]

    ### Get input bltuple ###
    print 'Opening file {0}'.format(infile)
    froot  = r.TFile(infile)

    event_count = {}
    for dataset in dataset_list:
        ecount  = froot.Get('TotalEvents_{0}'.format(dataset))
        event_count[dataset] = [ecount.GetBinContent(i+1) for i in range(ecount.GetNbinsX())]

        tree    = froot.Get('tree_{0}'.format(dataset))
        n       = tree.GetEntriesFast()
        ntuple  = {
                   'run_number':[], 'event_number':[], 'lumi':[], 'weight':[],
                   'muon1_pt':[], 'muon1_eta':[], 'muon1_phi':[], 'muon1_iso':[], 
                   'muon2_pt':[], 'muon2_eta':[], 'muon2_phi':[], 'muon2_iso':[], 
                   'muon_delta_eta':[], 'muon_delta_phi':[], 'muon_delta_r':[],
                   'dimuon_mass':[], 'dimuon_pt':[], 'dimuon_eta':[], 'dimuon_phi':[], 
                   'n_jets':[], 'n_fwdjets':[], 'n_bjets':[],
                   'bjet_pt':[], 'bjet_eta':[], 'bjet_phi':[], 'bjet_d0':[],
                   'jet_pt':[], 'jet_eta':[], 'jet_phi':[], 'jet_d0':[], 
                   'dijet_mass':[], 'dijet_pt':[], 'dijet_eta':[], 'dijet_phi':[], 
                   'delta_phi':[], 'delta_eta':[], 'four_body_mass':[],
                   #'dimuon_b_mass':[], 'dimuon_b_pt':[],
                   'met_mag':[], 'met_phi':[],
                   }

        print 'Reading {0} with {1} events...'.format(dataset, n)
        for i in xrange(n):
            tree.GetEntry(i)

            # get and build physics objects
            mu1, mu2, bjet, jet = tree.muonOneP4, tree.muonTwoP4, tree.bjetP4, tree.jetP4
            met, met_phi        = tree.met, tree.metPhi
            dimuon              = mu1 + mu2
            dijet               = jet + bjet
            quadbody            = dimuon + dijet
            tribody             = dimuon + bjet

            # event info
            ntuple['run_number'].append(tree.runNumber)
            ntuple['event_number'].append(tree.evtNumber)
            ntuple['lumi'].append(tree.lumiSection)
            ntuple['weight'].append(1)

            ### muon
            ntuple['muon1_pt'].append(mu1.Pt())
            ntuple['muon1_eta'].append(mu1.Eta())
            ntuple['muon1_phi'].append(mu1.Phi())
            ntuple['muon1_iso'].append(tree.muonOneIso)
            ntuple['muon2_pt'].append(mu2.Pt())
            ntuple['muon2_eta'].append(mu2.Eta())
            ntuple['muon2_phi'].append(mu2.Phi())
            ntuple['muon2_iso'].append(tree.muonTwoIso)
            ntuple['muon_delta_eta'].append(abs(mu1.Eta() - mu2.Eta()))
            ntuple['muon_delta_phi'].append(mu1.DeltaPhi(mu2))
            ntuple['muon_delta_r'].append(mu1.DeltaR(mu2))

            ### dimuon and dijet
            ntuple['dimuon_mass'].append(dimuon.M())
            ntuple['dimuon_pt'].append(dimuon.Pt())
            ntuple['dimuon_eta'].append(dimuon.Eta())
            ntuple['dimuon_phi'].append(dimuon.Phi())
            ntuple['dijet_mass'].append(dijet.M())
            ntuple['dijet_pt'].append(dijet.Pt())
            ntuple['dijet_eta'].append(dijet.Eta())
            ntuple['dijet_phi'].append(dijet.Phi())

            # four body variables
            ntuple['delta_phi'].append(dijet.DeltaPhi(dimuon))
            ntuple['delta_eta'].append(abs(dijet.Eta() - dimuon.Eta()))
            ntuple['four_body_mass'].append(quadbody.M())

            # dimuon + b jet variables

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

        df = pd.DataFrame(ntuple)
        df.to_csv('data/flatuples/{0}_{1}/ntuple_{2}.csv'.format(selection, period, dataset), index=False)

    df = pd.DataFrame(event_count)
    df.to_csv('data/flatuples/{0}_{1}/event_counts.csv'.format(selection, period))
