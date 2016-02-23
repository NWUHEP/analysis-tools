#!/usr/bin/env python

import sys
import os
from math import sqrt

# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
sys.argv.append( '-b-' )
import ROOT as r
from ROOT import gROOT, gStyle, gPad, RooFit, TCanvas, TLatex
gROOT.SetBatch(True)
sys.argv.remove( '-b-' )

#from configLimits import *
from fitBuilder import NameFixer

# ______________________________________________________________________________
tlatex = TLatex()
tlatex.SetNDC()
tlatex.SetTextFont(42)
tlatex.SetTextSize(0.03)

vladimir = False

donotdelete = []

def getWorkspace(fname, wsname):
    '''Open a .root file, retrieve a RooWorkspace'''
    
    if not fname.endswith('.root'):
        raise RuntimeError('Input not a .root file', fname)
    
    tfile = r.TFile.Open(fname)
    if not tfile:
        raise RuntimeError('Cannot open file', fname)
        
    ws = tfile.Get(wsname)
    if not ws:
        raise RuntimeError('Cannot get workspace', wsname)
    return ws

def addText(ws):
    tlatex.DrawLatex(0.18, 0.84, 'Nevts [12,70]: %.0f' % (ws.data('data_obs').sumEntries('12<=x && x<=70')))
    tlatex.DrawLatex(0.18, 0.80, 'Nevts [26,32]: %.0f' % (ws.data('data_obs').sumEntries('26<=x && x<=32')))
    
    tlatex.DrawLatex(0.64, 0.88, '#color[600]{Signal fit}')
    tlatex.DrawLatex(0.64, 0.84, '#color[600]{#mu = %.2f +/- %.2f}' % (ws.var('mean').getVal(), ws.var('mean').getError()))
    tlatex.DrawLatex(0.64, 0.80, '#color[600]{#sigma = %.2f +/- %.2f}' % (ws.var('sigma').getVal(), ws.var('sigma').getError()))
    #tlatex.DrawLatex(0.64, 0.76, '#color[600]{N_{s} = %.2f +/- %.2f}' % (ws.var('nsig').getVal(), ws.var('nsig').getError()))
    tlatex.DrawLatex(0.64, 0.76, '#color[600]{Signif = %.2f#sigma}' % (ws.var('signif').getVal()))
    
    argset = r.RooArgSet(ws.var('x'))
    s0 = ws.pdf('gauss').createIntegral(argset).getVal()
    s1 = ws.pdf('gauss').createIntegral(argset,RooFit.Range('xsignal')).getVal()
    #s2 = ws.pdf('gauss').createIntegral(argset,RooFit.NormSet(argset),RooFit.Range('xsignal')).getVal()
    b0 = ws.pdf('pol').createIntegral(argset).getVal()
    b1 = ws.pdf('pol').createIntegral(argset,RooFit.Range('xsignal')).getVal()
    #b2 = ws.pdf('pol').createIntegral(argset,RooFit.NormSet(argset),RooFit.Range('xsignal')).getVal()
    tlatex.DrawLatex(0.64, 0.72, '#color[600]{N_{s} [26,32] = %.2f +/- %.2f}' % (ws.var('nsig').getVal()*s1/s0, ws.var('nsig').getError()*s1/s0))
    tlatex.DrawLatex(0.64, 0.68, '#color[600]{N_{b} [26,32] = %.2f +/- %.2f}' % (ws.var('nbkg').getVal()*b1/b0, ws.var('nbkg').getError()*b1/b0))

if __name__=='__main__':
    
    # Set style
    gROOT.LoadMacro('tdrstyle.C')
    gROOT.ProcessLine('setTDRStyle()')
    gStyle.SetLabelSize(0.04, 'Z')

    # Configuration
    events = {
        'amumu_1b1f': 'data/amumuFile_MuMu2012ABCD_sasha_54b.root',
        #'amumu_1b1c': 'data/amumuFile_MuMu2012ABCD_sasha_56b.root',
    }

    x0      = 28
    xmin    = 12 # (xmax - xmin) must be even
    xmax    = 70
    
    # Run
    canvas = TCanvas()
    
    for cat,filename in events.iteritems():
        tfile = r.TFile.Open(filename)
        ttree = tfile.Get('amumuTree_DATA')
        
        ws = r.RooWorkspace('amumu', 'amumu')
    
        factory = ws.factory('{0}[{1}, {2}, {3}]'.format('x', x0, xmin, xmax))
        factory.SetTitle('M(#mu#mu) [GeV]')
        
        nbinsx = (xmax - xmin)/2
        factory.setRange(xmin, xmax)
        factory.setBins(int(nbinsx))

        data = r.RooDataSet('data_obs', 'data_obs', ttree, r.RooArgSet(factory), 'x>0', '__WEIGHT__')
        getattr(ws,'import')(data)
        
        fframe = factory.frame()
        fframe.SetTitle('BIG FITS')
        data.plotOn(fframe)

        ### Signal gaussian initialization ####
        mean=30 
        mean_lo=26 
        mean_hi=32 
        sigma=1 
        sigma_lo=0.50 
        sigma_hi=10.0
        model_sig = ws.factory('Gaussian::gauss(x, mean[{0}, {1}, {2}], sigma[{3}, {4}, {5}])'.format(mean, mean_lo, mean_hi, sigma, sigma_lo, sigma_hi))

        #mean    = r.RooRealVar('mean', 'mean', 30, 26, 34) 
        #sigma   = r.RooRealVar('sigma', 'sigma', 2, 0.5, 4) 

        ### BG Polynomial initialization ###
        a0=1 
        a0_lo=0 
        a0_hi=100 
        a1=1e-2 
        a1_lo=0 
        a1_hi=200 
        a2=1e-4 
        a2_lo=0 
        a2_hi=2
        model_bg = ws.factory('Polynomial::pol(x, {{a0[{0}, {1}, {2}], a1[{3}, {4}, {5}], a2[{6}, {7}, {8}]}}, 0)'.format(a0, a0_lo, a0_hi, a1, a1_lo, a1_hi, a2, a2_lo, a2_hi))

        ### Combined Signal+BG ###
        nbkg_lo = 0
        nbkg_hi = 1e3
        nsig_lo = 0
        nsig_hi = 100
        model_comb = ws.factory('SUM::model(nbkg[{0}, {1}]*pol, nsig[{2}, {3}]*gauss)'.format(nbkg_lo, nbkg_hi, nsig_lo, nsig_hi))

        nsig = ws.var('nsig')
        nsig.setVal(0.) 

        nsig.setConstant(True)
        result0 = model_bg.fitTo(data, RooFit.Save(True), 
                RooFit.Extended(True), RooFit.Minimizer('Minuit2','Migrad'), 
                RooFit.Hesse(True), RooFit.Minos(True), RooFit.PrintLevel(-1)) 
        result0.Print()

        nsig.setConstant(False)
        result = model_comb.fitTo(data, RooFit.Save(True), 
                RooFit.Extended(True), RooFit.Minimizer('Minuit2','Migrad'), 
                RooFit.Hesse(True), RooFit.Minos(True), RooFit.PrintLevel(-1)) 
        result.Print()
        
        signif = sqrt(2.0*abs(result.minNll() - result0.minNll()))
        print 'L_0 = {0}, L_1 = {1}'.format(result0.minNll(), result.minNll())
        ws.factory('signif[{0}]'.format(signif))
        
        model_comb.plotOn(fframe, RooFit.Components('gauss'), RooFit.LineStyle(2), RooFit.Invisible())
        model_comb.plotOn(fframe, RooFit.Components('pol'), RooFit.LineStyle(2))
        model_comb.plotOn(fframe)
        model_bg.plotOn(fframe, RooFit.LineStyle(3))

        intsig = ws.pdf('gauss').createInegral(r.RooArgSet(factory)).getVal()
        intbkg = ws.pdf('pol').createIntegral(r.RooArgSet(factory)).getVal()

        nsig_func = ws.factory('nsig_func[0]')
        nsig_func.setVal(ws.var('nsig').getVal()/intsig)
        nsig_func.setError(ws.var('nsig').getError()/intsig)

        nbkg_func = ws.factory('nbkg_func[0]')
        nbkg_func.setVal(ws.var('nbkg').getVal()/intbkg)
        nbkg_func.setError(ws.var('nbkg').getError()/intbkg)

        print 'sig = {0}'.format(intsig)
        print 'bkg = {0}'.format(intbkg)
        print 'f_sig = {0} +/- {1}'.format(nsig_func.getVal(), nsig_func.getError())
        print 'f_bkg = {0} +/- {1}'.format(nbkg_func.getVal(), nbkg_func.getError())

        fframe.SetMaximum(fframe.GetMaximum()*1.3)
        print 'ymax =', fframe.GetMaximum()
        gPad.SetLeftMargin(0.15);
        fframe.GetYaxis().SetTitleOffset(1.2)
        fframe.Draw()
        addText(ws)
        
        ## ______________________________________________________________________
        ## Print
        gPad.Print('figures/fit_{0}.png'.format(cat))
        gPad.Print('figures/fit_{0}.pdf'.format(cat))
         
        ## ______________________________________________________________________
        ## Fix the parameter names
        nameFixer = NameFixer(ws, cat, 'CMS_amumu')
        nameFixer.Fix('sig', 'bg')
         
        ## ______________________________________________________________________
        ## Output
        donotdelete.append(fframe)
        
        #rootname = 'ws_%s.root' % cat
        #ws.writeToFile(rootname)
        #print 'I wrote the workspace: %s' % rootname
