#!/usr/bin/env

from __future__ import division

import numpy as np
import ROOT as r

if __name__ == '__main__':

    data_file = r.TFile('data/pileup_2016_BCD.root')
    pileup_hist = data_file.Get('pileup')
    mc_data = np.array([
        0.000829312873542 , 0.00124276120498  , 0.00339329181587  ,
        0.00408224735376  , 0.00383036590008  , 0.00659159288946  ,
        0.00816022734493  , 0.00943640833116  , 0.0137777376066   ,
        0.017059392038    , 0.0213193035468   , 0.0247343174676   ,
        0.0280848773878   , 0.0323308476564   , 0.0370394341409   ,
        0.0456917721191   , 0.0558762890594   , 0.0576956187107   ,
        0.0625325287017   , 0.0591603758776   , 0.0656650815128   ,
        0.0678329011676   , 0.0625142146389   , 0.0548068448797   ,
        0.0503893295063   , 0.040209818868    , 0.0374446988111   ,
        0.0299661572042   , 0.0272024759921   , 0.0219328403791   ,
        0.0179586571619   , 0.0142926728247   , 0.00839941654725  ,
        0.00522366397213  , 0.00224457976761  , 0.000779274977993 ,
        0.000197066585944 , 7.16031761328e-05 , 0.0               , 0.0 , 0.0 ,
        0.0               , 0.0               , 0.0               , 0.0 , 0.0 ,
        0.0               , 0.0               , 0.0               , 0.0
       ])

    scale_factors = []
    data_sum = pileup_hist.Integral()
    for ibin in range(500):
        bin_content = pileup_hist.GetBinContent(ibin+1)/data_sum
        jbin = int(np.floor(ibin/10))
        if mc_data[jbin] > 0.:
            scale_factors.append(10*bin_content/mc_data[jbin])
        else:
            scale_factors.append(0.)


    out_file = r.TFile('data/pileup_sf_2016_BCD.root', 'RECREATE')
    sf_graph = r.TGraph(500, np.linspace(0, 50, 500), np.array(scale_factors))
    sf_graph.SetName('pileup')
    sf_graph.SetTitle('pileup')
    out_file.Add(sf_graph)
    out_file.Write()
    out_file.Close()

