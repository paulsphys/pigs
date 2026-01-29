'''
A path-integral quantum Monte Carlo program to compute the energy of the simple
harmonic oscillator in one spatial dimension.
'''

import numpy as np
from constants import *
#import emcee
#import matplotlib.pyplot as plt
from paths import *
from outproc import *
from pimc import *
# ------------------------------------------------------------------------------------------- 
def SHOEnergyExact(T):
    '''The exact SHO energy when \hbar \omega/ k_B = 1.''' 
    return 0.5/np.tanh(0.5/T)

# ------------------------------------------------------------------------------------------- 
def HarmonicOscillator(R):
    '''Simple harmonic oscillator potential with m = 1 and \omega = 1.'''
    return 0.5*np.dot(R,R);
def AnharmonicOscillator(R):
    c3 = 0.5238
    #c3 = 0.2381
    c4 = 0.0907
    return 0.5*np.dot(R,R) + 0.5*c3*np.dot(np.dot(R,R),R) + 0.5*c4*np.dot(R,R)*np.dot(R,R)
def Free(R):
    return 0
# ------------------------------------------------------------------------------------------- 
def main():

    print('Simulation Parameters:')
    print('N      = %d' % numParticles)
    print('tau    = %6.4f' % tau)
    print('lambda = %6.4f' % lam)
    print('Imaginary time Length = %4.2f\n' % beta)

    # fix the random seed
    np.random.seed(seed)

    # initialize main data structure
    beads = np.zeros([numTimeSlices,numParticles])

    # random initial positions (classical state) 
    for tslice in range(numTimeSlices):
        for ptcl in range(numParticles):
            beads[tslice,ptcl] = 0.5*(-1.0 + 2.0*np.random.random())
    
    # setup the paths
    BoxLength = 1
    Path = Paths(beads,tau,lam,BoxLength)
    Path.SetPotential(Free)
    for ptcl in range(numParticles):
        Path.beads[:,ptcl] = Path.PutPathInBC(Path.beads[:,ptcl])
    
    # compute the energy via path-integral Monte Carlo
    PotEnergy,KinEnergy = PIMC(numMCSteps,Path)
    
    # Do some simple binning statistics
    numBins = int(1.0*len(KinEnergy)/binSize)
    slices = np.linspace(0, len(KinEnergy),numBins+1,dtype=int)
    binnedKinEnergy = np.add.reduceat(KinEnergy, slices[:-1]) / np.diff(slices)
    binnedPotEnergy = np.add.reduceat(PotEnergy, slices[:-1]) / np.diff(slices)
    #acf = emcee.autocorr.function_1d(binnedEnergy)
    #np.save('essacf.npy',np.array(acf))
    #plt.plot(acf)
    #plt.show()
    #print(emcee.autocorr.integrated_time(binnedEnergy))
    # output the final result
    print('Energy = %8.4f +/- %6.4f' % (np.mean(binnedKinEnergy), (np.std(binnedKinEnergy)/np.sqrt(numBins-1))))
    print('Energy = %8.4f +/- %6.4f' % (np.mean(binnedPotEnergy), (np.std(binnedPotEnergy)/np.sqrt(numBins-1))))
    #print('Eexact = %8.4f' % SHOEnergyExact(T))
    #return np.mean(binnedEnergy)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

