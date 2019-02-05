#!/usr/bin/env python

import os, sys
from copy import deepcopy
import multiprocessing as mp

import numpy as np
import pandas as pd
import ROOT as r

from tqdm import tqdm, trange
import scripts.plot_tools as pt

'''
Script for getting blt data out of ROOT files and into CSV format.
'''

jec_source_names = [
                    "abs_scale", "abs_stat",
                    "fragmentation", "abs_mpf_bias", 
                    "pileup_data_mc", "pileup_pt_bb", "pileup_pt_ec1", "pileup_pt_ref",
                    "relative_bal", "relative_jer_ec1",
                    "relative_pt_bb","relative_pt_ec1",
                    "relative_stat_fsr", "relative_stat_ec",
                    "single_pion_ecal", "single_pion_hcal",
                    "time_pt_eta", "flavor_qcd",
                    "total"
                   ]

btag_source_names = [
                     "bfragmentation", "btempcorr", "cb",       
                     "cfragmentation", "dmux", "gluonsplitting",
                     "jes", "jetaway", "ksl", "l2c",            
                     "ltothers", "mudr", 
                     "mupt", "sampledependence", "pileup",      
                     "ptrel", "statistic" 
                    ]

def calculate_cos_theta(ref_p4, boost_p4, target_p4):
    '''
    !!! THIS NEED TO BE FIXED SO THAT THE INPUTS ARE NOT MODIFIED !!!
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

def calculate_trans_lifetime(p4, reference_vertex, decay_vertex):

    dvtx = decay_vertex - reference_vertex
    p2   = r.TVector2(p4.Px(), p4.Py())
    r2   = r.TVector2(dvtx.X(), dvtx.Y())
    lxy  = (p2*r2)/p4.Pt()
    txy  = (100./3.)*(lxy*p4.M())/p4.Pt()

    return lxy, txy

def calculate_mt(lep_p4, met_p2):

    lep_p2 = r.TVector2(lep_p4.Px(), lep_p4.Py())
    delta_phi = met_p2.DeltaPhi(lep_p2)

    return np.sqrt(2*lep_p4.Pt()*met_p2.Mod()*(1 - np.cos(delta_phi))), delta_phi

def calculate_zeta_vars(lep1_p4, lep2_p4, met_p2):
    '''
    This variable is used for reducing W+jet contamination in Z->tautau events.
    It is described in arXiv:0508051.  
    '''

    # first calculate the bisector vector of the angle between the lepton and the tau
    lep1_p2 = r.TVector2(lep1_p4.Px(), lep2_p4.Py()) 
    lep2_p2 = r.TVector2(lep2_p4.Px(), lep2_p4.Py())
    zeta = lep1_p2.Unit() + lep2_p2.Unit()

    # calculate projection vectors
    dilepton    = lep1_p2 + lep2_p2
    p_vis_zeta  = zeta * dilepton
    p_miss_zeta = zeta * met_p2
    
    return p_vis_zeta, p_miss_zeta

def fill_event_vars(tree, dataset):

    out_dict = dict(
                    run_number     = tree.runNumber,
                    event_number   = tree.evtNumber,
                    lumi           = tree.lumiSection,
                    trigger_status = tree.triggerStatus,
 
                    n_pu           = tree.nPU,
                    n_pv           = tree.nPV,
                    n_muons        = tree.nMuons,
                    n_electrons    = tree.nElectrons,
                    n_taus         = tree.nTaus,
                    n_jets         = tree.nJets,
                    n_fwdjets      = tree.nFwdJets,
                    n_bjets        = tree.nBJets,
                    n_bjets_raw    = tree.nBJetsRaw,
 
                    met_mag        = tree.met,
                    met_phi        = tree.metPhi,
                    ht_mag         = tree.ht,
                    ht_phi         = tree.htPhi,

                    # jet counting for systematics
                    n_jets_jer_up       = tree.nJetsJERUp,
                    n_jets_jer_down     = tree.nJetsJERDown,
                    n_bjets_jer_up      = tree.nBJetsJERUp,
                    n_bjets_jer_down    = tree.nBJetsJERDown,
                    n_bjets_ctag_up     = tree.nBJetsCTagUp,
                    n_bjets_ctag_down   = tree.nBJetsCTagDown,
                    n_bjets_mistag_up   = tree.nBJetsMistagUp,
                    n_bjets_mistag_down = tree.nBJetsMistagDown,

                    lepton1_reco_weight = tree.leptonOneRecoWeight,
                    lepton2_reco_weight = tree.leptonTwoRecoWeight,
                    lepton1_id_weight   = tree.leptonOneIDWeight,
                    lepton2_id_weight   = tree.leptonTwoIDWeight,
                    trigger_weight      = tree.triggerWeight,
                    pileup_weight       = tree.puWeight,
                    top_pt_weight       = tree.topPtWeight,
                    event_weight        = tree.eventWeight,

                    lepton1_reco_var = tree.leptonOneRecoVar,
                    lepton2_reco_var = tree.leptonTwoRecoVar,
                    lepton1_id_var   = tree.leptonOneIDVar,
                    lepton2_id_var   = tree.leptonTwoIDVar,

                   )

    if dataset in ['zjets_m-50', 'zjets_m-10to50'] and 0 < tree.nPartons < 5:
        out_dict['weight'] = 0.
    else:
        out_dict['weight'] = tree.eventWeight

    # factorized JEC systematic
    if tree.runNumber == 1: # checks if dataset is real data or MC

        for i, n in enumerate(jec_source_names):
            out_dict[f'n_jets_jes_{n}_up']    = tree.nJetsJESUp[i]
            out_dict[f'n_jets_jes_{n}_down']  = tree.nJetsJESDown[i]
            out_dict[f'n_bjets_jes_{n}_up']   = tree.nBJetsJESUp[i]
            out_dict[f'n_bjets_jes_{n}_down'] = tree.nBJetsJESDown[i]

        for i, n in enumerate(btag_source_names):
            out_dict[f'n_bjets_btag_{n}_up']   = tree.nBJetsBTagUp[i]
            out_dict[f'n_bjets_btag_{n}_down'] = tree.nBJetsBTagDown[i]

        # generator weights and systematics
        out_dict['gen_weight'] = tree.genWeight
        out_dict['n_partons']  = tree.nPartons
        if dataset in ['ttbar_inclusive', 'zjets_m-10to50_alt', 'zjets_m-50_alt']:
            out_dict['qcd_weight_nominal']   = tree.qcdWeights[0]
            out_dict['qcd_weight_nom_up']    = tree.qcdWeights[1]
            out_dict['qcd_weight_nom_down']  = tree.qcdWeights[2]
            out_dict['qcd_weight_up_nom']    = tree.qcdWeights[3]
            out_dict['qcd_weight_up_up']     = tree.qcdWeights[4]
            out_dict['qcd_weight_up_down']   = tree.qcdWeights[5]
            out_dict['qcd_weight_down_nom']  = tree.qcdWeights[6]
            out_dict['qcd_weight_down_up']   = tree.qcdWeights[7]
            out_dict['qcd_weight_down_down'] = tree.qcdWeights[8]
            out_dict['pdf_var']              = tree.pdfWeight 
            out_dict['alpha_s_err']          = tree.alphaS

    return out_dict

def fill_dilepton_vars(tree):

    lep1, lep2 = tree.leptonOneP4, tree.leptonTwoP4
    dilepton1  = lep1 + lep2
    met_p2 = r.TVector2(tree.met*np.cos(tree.metPhi), tree.met*np.sin(tree.metPhi))

    lep1_mt, lep1_met_dphi = calculate_mt(lep1, met_p2)
    lep2_mt, lep2_met_dphi = calculate_mt(lep2, met_p2)

    p_vis_zeta, p_miss_zeta = calculate_zeta_vars(lep1, lep2, met_p2)

    if lep1.Pt() > lep2.Pt():
        lead_lepton_pt     = lep1.Pt()
        lead_lepton1_phi   = lep1.Phi()
        lead_lepton_mt     = lep1_mt
        lead_lepton_met_dphi = lep1_met_dphi
        lead_lepton_mother = tree.leptonOneMother
        lead_lepton_flavor = tree.leptonOneFlavor

        trailing_lepton_pt     = lep2.Pt()
        trailing_lepton_phi    = lep2.Phi()
        trailing_lepton_mt     = lep2_mt
        trailing_lepton_met_dphi = lep2_met_dphi
        trailing_lepton_mother = tree.leptonTwoMother
        trailing_lepton_flavor = tree.leptonTwoFlavor
    else:
        lead_lepton_pt     = lep2.Pt()
        lead_lepton1_phi   = lep2.Phi()
        lead_lepton_mt     = lep2_mt
        lead_lepton_met_dphi = lep2_met_dphi
        lead_lepton_mother = tree.leptonTwoMother
        lead_lepton_flavor = tree.leptonTwoFlavor

        trailing_lepton_pt     = lep1.Pt()
        trailing_lepton_phi    = lep1.Phi()
        trailing_lepton_mt     = lep1_mt
        trailing_lepton_met_dphi = lep1_met_dphi
        trailing_lepton_mother = tree.leptonOneMother
        trailing_lepton_flavor = tree.leptonOneFlavor

    out_dict = dict(
                    lepton1_pt             = lep1.Pt(),
                    lepton1_eta            = lep1.Eta(),
                    lepton1_phi            = lep1.Phi(),
                    lepton1_mt             = lep1_mt,
                    lepton1_met_dphi       = abs(lep1_met_dphi),
                    lepton1_d0             = tree.leptonOneD0,
                    lepton1_dz             = tree.leptonOneDZ,
                    lepton1_q              = np.sign(tree.leptonOneFlavor),
                    lepton1_flavor         = np.abs(tree.leptonOneFlavor),
                    lepton1_iso            = tree.leptonOneIso,
                    lepton1_reliso         = tree.leptonOneIso/lep1.Pt(),
                    lepton1_mother         = tree.leptonOneMother,
 
                    lepton2_pt             = lep2.Pt(),
                    lepton2_eta            = lep2.Eta(),
                    lepton2_phi            = lep2.Phi(),
                    lepton2_mt             = lep2_mt,
                    lepton2_met_dphi       = abs(lep2_met_dphi),
                    lepton2_d0             = tree.leptonTwoD0,
                    lepton2_dz             = tree.leptonTwoDZ,
                    lepton2_q              = np.sign(tree.leptonTwoFlavor),
                    lepton2_flavor         = np.abs(tree.leptonTwoFlavor),
                    lepton2_iso            = tree.leptonTwoIso,
                    lepton2_reliso         = tree.leptonTwoIso/lep2.Pt(),
                    lepton2_mother         = tree.leptonTwoMother,

                    lead_lepton_pt         = lead_lepton_pt,
                    lead_lepton_mt         = lead_lepton_mt,
                    lead_lepton_mother     = lead_lepton_mother,
                    lead_lepton_flavor     = lead_lepton_flavor,
                    trailing_lepton_pt     = trailing_lepton_pt,
                    trailing_lepton_mt     = trailing_lepton_mt,
                    trailing_lepton_mother = trailing_lepton_mother,
                    trailing_lepton_flavor = trailing_lepton_flavor,
 
                    dilepton1_delta_eta    = abs(lep1.Eta() - lep2.Eta()),
                    dilepton1_delta_phi    = abs(lep1.DeltaPhi(lep2)),
                    dilepton1_delta_r      = lep1.DeltaR(lep2),
                    dilepton1_mass         = dilepton1.M(),
                    dilepton1_pt           = dilepton1.Pt(),
                    dilepton1_eta          = dilepton1.Eta(),
                    dilepton1_phi          = dilepton1.Phi(),
                    dilepton1_pt_over_m    = dilepton1.Pt()/dilepton1.M(),
                    dilepton1_pt_diff      = (lep1.Pt() - lep2.Pt()),
                    dilepton1_pt_asym      = (lep1.Pt() - lep2.Pt())/(lep1.Pt() + lep2.Pt()),

                    p_vis_zeta             = p_vis_zeta,
                    p_miss_zeta            = p_miss_zeta,
                    delta_p_zeta           = p_miss_zeta - 0.85*p_vis_zeta
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
                    jet1_flavor     = tree.jetOneFlavor,
 
                    jet2_pt         = jet2.Pt(),
                    jet2_eta        = jet2.Eta(),
                    jet2_phi        = jet2.Phi(),
                    jet2_e          = jet2.E(),
                    jet2_tag        = tree.jetTwoTag,
                    jet2_flavor     = tree.jetTwoFlavor,
 
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

def fill_lepton4j_vars(tree):

    # aliases
    #met = r.TVector2(tree.met*np.cos(tree.metPhi), tree.met*np.sin(tree.metPhi))
    lep = tree.leptonOneP4
    jet1, jet2  = tree.jetOneP4, tree.jetTwoP4
    jet3, jet4  = tree.jetThreeP4, tree.jetFourP4
    lepton_j1  = lep + jet1
    lepton_j2  = lep + jet2
    lepton_j3  = lep + jet3
    lepton_j4  = lep + jet4
    
    w_cand = jet1 + jet2
    htop_cand = w_cand + jet3

    met_p2 = r.TVector2(tree.met*np.cos(tree.metPhi), tree.met*np.sin(tree.metPhi))
    lep_mt, lep_met_dphi = calculate_mt(lep, met_p2)

    out_dict = dict(
                    lepton1_pt      = lep.Pt(),
                    lepton1_mt      = lep_mt,
                    lepton1_eta     = lep.Eta(),
                    lepton1_phi     = lep.Phi(),
                    lepton1_q       = np.sign(tree.leptonOneFlavor),
                    lepton1_flavor  = np.abs(tree.leptonOneFlavor),
                    lepton1_iso     = tree.leptonOneIso,
                    lepton1_reliso  = tree.leptonOneIso/lep.Pt(),

                    jet1_pt     = jet1.Pt(),
                    jet1_phi    = jet1.Phi(),
                    jet1_eta    = jet1.Eta(),
                    jet1_e      = jet1.E(),
                    jet1_tag    = tree.jetOneTag,
                    jet1_flavor = tree.jetOneFlavor,
 
                    jet2_pt     = jet2.Pt(),
                    jet2_eta    = jet2.Eta(),
                    jet2_phi    = jet2.Phi(),
                    jet2_e      = jet2.E(),
                    jet2_tag    = tree.jetTwoTag,
                    jet2_flavor = tree.jetTwoFlavor,

                    jet3_pt     = jet3.Pt(),
                    jet3_eta    = jet3.Eta(),
                    jet3_phi    = jet3.Phi(),
                    jet3_e      = jet3.E(),
                    jet3_tag    = tree.jetThreeTag,
                    jet3_flavor = tree.jetThreeFlavor,
 
                    jet4_pt     = jet4.Pt(),
                    jet4_eta    = jet4.Eta(),
                    jet4_phi    = jet4.Phi(),
                    jet4_e      = jet4.E(),
                    jet4_tag    = tree.jetFourTag,
                    jet4_flavor = tree.jetFourFlavor,

                    w_pt         = w_cand.Pt(),
                    w_eta        = w_cand.Eta(),
                    w_phi        = w_cand.Phi(),
                    w_mass       = w_cand.M(),
                    w_delta_eta  = abs(jet1.Eta() - jet2.Eta()),
                    w_delta_phi  = abs(jet1.DeltaPhi(jet2)),
                    w_delta_r    = jet1.DeltaR(jet2),

                    htop_pt         = htop_cand.Pt(),
                    htop_eta        = htop_cand.Eta(),
                    htop_phi        = htop_cand.Phi(),
                    htop_mass       = htop_cand.M(),
                    htop_delta_eta  = abs(w_cand.Eta() - jet3.Eta()),
                    htop_delta_phi  = abs(w_cand.DeltaPhi(jet3)),
                    htop_delta_r    = w_cand.DeltaR(jet3),
 
                    lepton_j1_mass       = lepton_j1.M(),
                    lepton_j1_pt         = lepton_j1.Pt(),
                    lepton_j1_delta_eta  = abs(lep.Eta() - jet1.Eta()),
                    lepton_j1_delta_phi  = abs(lep.DeltaPhi(jet1)),
                    lepton_j1_delta_r    = lep.DeltaR(jet1),

                    lepton_j2_mass       = lepton_j2.M(),
                    lepton_j2_pt         = lepton_j2.Pt(),
                    lepton_j2_delta_eta  = abs(lep.Eta() - jet2.Eta()),
                    lepton_j2_delta_phi  = abs(lep.DeltaPhi(jet2)),
                    lepton_j2_delta_r    = lep.DeltaR(jet2),

                    lepton_j3_mass       = lepton_j3.M(),
                    lepton_j3_pt         = lepton_j3.Pt(),
                    lepton_j3_delta_eta  = abs(lep.Eta() - jet3.Eta()),
                    lepton_j3_delta_phi  = abs(lep.DeltaPhi(jet3)),
                    lepton_j3_delta_r    = lep.DeltaR(jet3),

                    lepton_j4_mass       = lepton_j4.M(),
                    lepton_j4_pt         = lepton_j4.Pt(),
                    lepton_j4_delta_eta  = abs(lep.Eta() - jet4.Eta()),
                    lepton_j4_delta_phi  = abs(lep.DeltaPhi(jet4)),
                    lepton_j4_delta_r    = lep.DeltaR(jet4),

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

def fill_ntuple(tree, selection, dataset, event_range=None, job_id=(1, 1, 1)):

    if type(event_range) == None:
        entries = range(0, tree.GetEntriesFast())
    else:
        entries = range(event_range[0], event_range[1], 1)

    for i in tqdm(entries, 
                  position = job_id[0],
                  desc     = f'[{selection}|{dataset}] {job_id[1]+1}/{job_id[2]} |',
                  ncols    = 0,
                  ascii    = True,
                  #leave    = True
                  ):
        tree.GetEntry(i)
        entry = {}
        entry.update(fill_event_vars(tree, dataset))

        if selection in ['ee', 'mumu', 'emu', 'etau', 'mutau']:
            entry.update(fill_jet_vars(tree))
            entry.update(fill_jet_lepton_vars(tree))
            entry.update(fill_dilepton_vars(tree))

            if selection in ['etau', 'mutau']:
                entry.update(fill_tau_vars(tree))

        elif selection in ['e4j', 'mu4j']:
            entry.update(fill_lepton4j_vars(tree))

        entry.update(fill_gen_particle_vars(tree))
        yield entry

def pickle_ntuple(tree, dataset, output_path, selection):

    # get the tree, convert to dataframe, and save df to pickle
    ntuple = fill_ntuple(tree, dataset, selection)
    df     = pd.DataFrame(ntuple)
    df     = df.query('weight != 0')
    df.to_pickle(f'{output_path}/ntuple_{dataset_name}.pkl')

if __name__ == '__main__':

    ### Configuration ###
    infile      = 'local_data/bltuples/output_z_cr.root'
    output_dir  = 'local_data/flatuples/z_cr'
    selections  = ['mumu']#, 'ee', 'emu', 'mutau', 'etau', 'mu4j', 'e4j']
    do_data     = True
    do_mc       = True
    do_syst     = False
    period      = 2016

    # configure datasets to run over
    dataset_list = []
    data_labels  = ['muon']#, 'electron']
    #mc_labels    = ['zjets', 'ttbar', 'diboson', 't', 'wjets']
    mc_labels    = ['zjets', 'zjets_alt']
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
    processes   = {}
    files_list  = [] # There needs to be multiple instances of the file to access each of the trees.  Not great...
    event_count = {}
    for selection in selections:
        output_path = f'{output_dir}/{selection}_{period}'
        pt.make_directory(output_path, clear=True)
        for dataset in dataset_list:

            root_file = r.TFile(infile)
            files_list.append(root_file)

            #ecount = root_file.Get(f'{selection}/TotalEvents_{selection}_{dataset}')
            ecount = root_file.Get(f'TotalEvents_{dataset}')
            if ecount:
                event_count[dataset] = [ecount.GetBinContent(i+1) for i in range(ecount.GetNbinsX())]
            else:
                print(f'Could not find dataset {dataset} in root file...')
                continue

            #tree = root_file.Get(f'{selection}/bltTree_{dataset}')
            tree = root_file.Get(f'{selection}/bltTree_{dataset}')
            p = mp.Process(target=pickle_ntuple, args=(tree, dataset, output_path, selection))
            p.start()
            processes[f'{dataset}_{selection}'] = p

        # special case: fakes
        if selection in ['mutau', 'mu4j', 'etau', 'e4j'] and do_data:
            for dataset in [d for l in data_labels for d in pt.dataset_dict[l]]:

                root_file = r.TFile(infile)
                files_list.append(root_file)

                event_count[f'{dataset}_fakes'] = 10*[1.,]

                #tree = root_file.Get(f'{selection}/bltTree_{dataset}')
                tree = root_file.Get(f'{selection}_fakes/bltTree_{dataset}')
                p = mp.Process(target=pickle_ntuple, 
                               args=(tree, f'{dataset}_fakes', output_path, selection))
                p.start()
                processes[f'{dataset}_{selection}_fakes'] = p

        fname = f'{output_path}/event_counts.csv'
        df = pd.DataFrame(event_count)
        df.to_csv(fname)

    for p in processes.values():
        p.join()

    for f in files_list:
        f.Close()
