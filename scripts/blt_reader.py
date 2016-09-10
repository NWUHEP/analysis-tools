#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
import ROOT as r
import json

'''
Simple script for getting data out of ROOT files and into CSV format.
'''

if __name__ == '__main__':

    if len(sys.argv) > 1:
        datatype = sys.argv[1]
    else:
        datatype = 'test'

    infile = 'data/bltuples/output_{0}.root'.format(datatype)
    rfiles = []
    froot  = r.TFile(infile)

    print 'Opening file {0}'.format(infile)
    if datatype == 'muon_2012':
        datasets = ['data_muon_2012A', 'data_muon_2012B', 'data_muon_2012C', 'data_muon_2012D'] 
    elif datatype[:9] == 'muon_2016':
        datasets = ['data_muon_2016B', 'data_muon_2016C', 'data_muon_2016D']#, 'data_muon_2016E', 'data_muon_2016F']
    elif datatype == 'test':
        datasets = ['data_muon']
    else:
        print 'Specified datatype is not valid!!!'
        exit()

    ntuple  = {
            'muon1_pt':[], 'muon1_eta':[], 'muon1_phi':[], 'muon1_iso':[], 
            'muon2_pt':[], 'muon2_eta':[], 'muon2_phi':[], 'muon2_iso':[], 
            'muon_delta_eta':[], 'muon_delta_phi':[], 'muon_delta_r':[],
            'dimuon_mass':[], 'dimuon_pt':[], 'dimuon_eta':[], 'dimuon_phi':[], 
            'dijet_mass':[], 'dijet_pt':[], 'dijet_eta':[], 'dijet_phi':[], 
            'delta_phi':[], 'delta_eta':[], 'four_body_mass':[],
            'bjet_pt':[], 'bjet_eta':[], 'bjet_phi':[], 'bjet_d0':[],
            'jet_pt':[], 'jet_eta':[], 'jet_phi':[], 'jet_d0':[], 
            'n_jets':[], 'n_fwdjets':[], 'n_bjets':[],
            'met_mag':[], 'met_phi':[],
            'run_number':[], 'event_number':[], 'lumi':[]
            }

    for dataset in datasets:
        tree    = froot.Get(dataset)
        n       = tree.GetEntriesFast()

        print 'Reading {0} with {1} events...'.format(dataset, n)
        for i in xrange(n):
            tree.GetEntry(i)

            # event info
            ntuple['run_number'].append(tree.runNumber)
            ntuple['event_number'].append(tree.evtNumber)
            ntuple['lumi'].append(tree.lumiSection)

            # get and build physics objects
            mu1, mu2, bjet, jet = tree.muonOneP4, tree.muonTwoP4, tree.bjetP4, tree.jetP4
            met, met_phi = tree.met, tree.metPhi
            dimuon = mu1 + mu2
            dijet = jet + bjet

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
            ntuple['four_body_mass'].append((dijet + dimuon).M())

            # jets
            ntuple['bjet_pt'].append(bjet.Pt())
            ntuple['bjet_eta'].append(bjet.Eta())
            ntuple['bjet_phi'].append(bjet.Phi())
            ntuple['bjet_d0'].append(tree.bjetD0)
            ntuple['jet_pt'].append(jet.Pt())
            ntuple['jet_eta'].append(jet.Eta())
            ntuple['jet_phi'].append(jet.Phi())
            ntuple['jet_d0'].append(tree.jet_d0)
            ntuple['n_jets'].append(tree.nJets)
            ntuple['n_fwdjets'].append(tree.nFwdJets)
            ntuple['n_bjets'].append(tree.nBJets)

            # MET
            ntuple['met_mag'].append(met)
            ntuple['met_phi'].append(met_phi)

    df = pd.DataFrame(ntuple)
    df.to_csv('data/ntuple_dimuon_{0}.csv'.format(datatype), index=False)
