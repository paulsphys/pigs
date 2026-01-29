import sys
import numpy as np

def binproc(estimatorvalues,binSize):
   """Given an array of estimator values, bin them and return the mean and average error"""
   numBins = int(1.0*len(estimatorvalues)/binSize)
   slices = np.linspace(0, len(estimatorvalues),numBins+1,dtype=int)
   Binnedvalues = np.add.reduceat(estimatorvalues, slices[:-1]) / np.diff(slices)
   average = np.mean(Binnedvalues) 
   error = np.std(Binnedvalues)/np.sqrt(numBins-1)
   return average,error
