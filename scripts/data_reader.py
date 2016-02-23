#!/usr/bin/env python

import pandas as pd
import ROOT as r

'''
Simple script for getting data out of ROOT files and into CSV format.
'''

if __name__ == '__main__':
    filenames = {
            '1b1f': 'data/amumuFile_MuMu2012ABCD_sasha_54b.root',
            '1b1c': 'data/amumuFile_MuMu2012ABCD_sasha_56b.root',
            '1b0f': 'data/amumuFile_MuMu2012ABCD_sasha_57b.root',
            '1b1c_inclusive': 'data/amumuFile_MuMu2012ABCD_sasha_58b.root'
            }

    rfiles = []
    varnames = ['id', 
            'mu1_pt', 'mu1_eta', 'mu1_phi',
            'mu2_pt', 'mu2_eta', 'mu2_phi',
            'jet1_pt', 'jet1_eta', 'jet1_phi',
            'jet2_pt', 'jet2_eta', 'jet2_phi',
            'met_rho', 'met_phi']

    for cat, name in filenames.iteritems():
        outfile = file('dimuon_mass_{0}.csv'.format(cat), 'w')
        froot = r.TFile(name)
        tree  = froot.Get('amumuTree_DATA')
        n     = tree.GetEntriesFast()
        for i in xrange(n):
            tree.GetEntry(i)
            outfile.write('{0}\n'.format(tree.x))

        outfile.close()

