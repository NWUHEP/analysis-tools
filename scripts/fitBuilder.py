#!/usr/bin/env python

class FitBuilder:
    def __init__(self, ws, cat):
        self.ws     = ws
        self.cat    = cat
        
        self.BuildDict = {
            "Gauss": self.BuildGaussian,
            "Pol": self.BuildPolynomial,
            "Gauss+Pol": self.BuildGaussPlusPol,
        }
    
    def Build(self, funcName, **kargs):
        return self.BuildDict[funcName](**kargs)
    
    def BuildGaussian(self, mean=30, mean_lo=14, mean_hi=68, sigma=1, sigma_lo=0.50, sigma_hi=10.0):
        self.ws.factory("Gaussian::gauss(x, mean[%s,%s,%s], sigma[%s,%s,%s])" % (str(mean), str(mean_lo), str(mean_hi), str(sigma), str(sigma_lo), str(sigma_hi)))
        return self.ws.pdf("gauss")
    
    def BuildPolynomial(self, a0=1, a0_lo=0, a0_hi=100, a1=1e-2, a1_lo=0, a1_hi=200, a2=1e-4, a2_lo=0, a2_hi=2):
        self.ws.factory("Polynomial::pol(x, {a0[%s,%s,%s], a1[%s,%s,%s], a2[%s,%s,%s]}, 0)" % (str(a0), str(a0_lo), str(a0_hi), str(a1), str(a1_lo), str(a1_hi), str(a2), str(a2_lo), str(a2_hi)))
        return self.ws.pdf("pol")
    
    def BuildGaussPlusPol(self, mean=30, mean_lo=14, mean_hi=68, sigma=1, sigma_lo=0.01, sigma_hi=10.0, 
                          a0=1, a0_lo=0, a0_hi=100, a1=1e-2, a1_lo=0, a1_hi=100, a2=1e-6, a2_lo=0, a2_hi=1,
                          nbkg_lo=0, nbkg_hi=1000, nsig_lo=0, nsig_hi=100):

        gauss   = self.BuildGaussian(mean, mean_lo, mean_hi, sigma, sigma_lo, sigma_hi)
        pol     = self.BuildPolynomial(a0, a0_lo, a0_hi, a1, a1_lo, a1_hi, a2, a2_lo, a2_hi)
        
        # Extended
        self.ws.factory("SUM::model(nbkg[%s,%s]*pol, nsig[%s,%s]*gauss)" % (nbkg_lo, nbkg_hi, nsig_lo, nsig_hi))
        return self.ws.pdf("model")

class NameFixer:
    def __init__(self, ws, cat, prefix=""):
        self.ws = ws
        self.cat = cat
        self.prefix = prefix

    def Fix(self, sig, bg):
        # sig
        mean = "mean"
        meanNew = "meanNew"
        sigma = "sigma"
        sigmaNew = "sigmaNew"
        mShift = self.prefix+"_"+sig+"_mShift_"+self.cat
        sigmaShift = self.prefix+"_"+sig+"_sigmaShift_"+self.cat
        self.ws.var(mean).setConstant(True)
        self.ws.var(sigma).setConstant(True)
        self.ws.factory("{0}[{1},{2},{3}]".format(mShift, 1, 0, 10))
        self.ws.factory("{0}[{1},{2},{3}]".format(sigmaShift, 1, 0, 10))
        self.ws.factory("PROD::{0}({1},{2})".format(meanNew, mean, mShift))
        self.ws.factory("PROD::{0}({1},{2})".format(sigmaNew, sigma, sigmaShift))
        
        gauss = "gauss"
        gaussNew = self.prefix+"_"+sig+"_"+self.cat
        gaussNorm = "nsig"
        gaussNormNew = gaussNew+"_norm"
        self.ws.factory("EDIT::{0}({1},{2}={3},{4}={5})".format(gaussNew, gauss, mean, meanNew, sigma, sigmaNew))
        self.ws.factory("{0}[{1},{2},{3}]".format(gaussNormNew, self.ws.function(gaussNorm).getVal(), self.ws.function(gaussNorm).getMin(), self.ws.function(gaussNorm).getMax()))
        self.ws.var(gaussNormNew).setError(self.ws.var(gaussNorm).getError())
        
        # bg
        a0 = "a0"
        a0New = self.prefix+"_"+bg+"_a0_"+self.cat
        a1 = "a1"
        a1New = self.prefix+"_"+bg+"_a1_"+self.cat
        a2 = "a2"
        a2New = self.prefix+"_"+bg+"_a2_"+self.cat
        self.ws.factory("{0}[{1},{2},{3}]".format(a0New, self.ws.function(a0).getVal(), self.ws.function(a0).getMin(), self.ws.function(a0).getMax()))
        self.ws.factory("{0}[{1},{2},{3}]".format(a1New, self.ws.function(a1).getVal(), self.ws.function(a1).getMin(), self.ws.function(a1).getMax()))
        self.ws.factory("{0}[{1},{2},{3}]".format(a2New, self.ws.function(a2).getVal(), self.ws.function(a2).getMin(), self.ws.function(a2).getMax()))
        self.ws.var(a0New).setError(self.ws.var(a0).getError())
        self.ws.var(a1New).setError(self.ws.var(a1).getError())
        self.ws.var(a2New).setError(self.ws.var(a2).getError())
        
        pol = "pol"
        polNew = self.prefix+"_"+bg+"_"+self.cat
        polNorm = "nbkg"
        polNormNew = polNew+"_norm"
        self.ws.factory("EDIT::{0}({1},{2}={3},{4}={5},{6}={7})".format(polNew, pol, a0, a0New, a1, a1New, a2, a2New))
        self.ws.factory("{0}[{1},{2},{3}]".format(polNormNew, self.ws.function(polNorm).getVal(), self.ws.function(polNorm).getMin(), self.ws.function(polNorm).getMax()))
        self.ws.var(polNormNew).setError(self.ws.var(polNorm).getError())

        return
