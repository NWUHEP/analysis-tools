#!/usr/bin/env

from __future__ import division

import numpy as np
import ROOT as r

if __name__ == '__main__':

    data_file = r.TFile('data/data_pileup.root')
    h_pileup_data = data_file.Get('pileup')

    mc_file = r.TFile('data/bltuples/output_mumu_2016.root')
    h_pileup_mc = r.TH1F('h_pileup_mc', '', 500, 0., 50.)
    mc_tree = mc_file.Get('tree_zjets_m-50')
    mc_tree.Draw('nPU>>h_pileup_mc') 

    scale_factors = []
    data_sum = h_pileup_data.Integral()
    mc_sum = h_pileup_mc.Integral()
    for ibin in range(500):
        x_mc = h_pileup_mc.GetBinContent(ibin+1)/mc_sum
        x_data = h_pileup_data.GetBinContent(ibin+1)/data_sum
        if x_mc > 0.:
            scale_factors.append(x_data/x_mc)
        else:
            scale_factors.append(0.)

    h_pileup_data.DrawNormalized()
    h_pileup_mc.DrawNormalized()

    out_file = r.TFile('data/pileup_sf_2016_full.root', 'RECREATE')
    sf_graph = r.TGraph(500, np.linspace(0, 50, 500), np.array(scale_factors))
    sf_graph.SetName('pileup_sf')
    sf_graph.SetTitle('pileup_sf')
    out_file.Add(sf_graph)
    out_file.Add(h_pileup_mc)
    out_file.Add(h_pileup_data)
    out_file.Write()
    out_file.Close()

