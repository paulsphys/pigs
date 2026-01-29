#Set all the constants for the pimc code
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s','--seed',type=int,default=1173,help="The seed of the random number generator")
parser.add_argument('-N',type=int,required=True, help='The number of particles to simulate')
parser.add_argument('--beta','-T',type=float,required = True, help="The inverse temperature of the simulation")
parser.add_argument('--NumTimeSlices','-P',type=int,default=20,help="The number of beads in a wordline")
parser.add_argument('--NumMCSteps','-S',type=int,default=40000,help="Number of Monte-Carlo steps")
parsed_args = vars(parser.parse_args())

beta = parsed_args['beta']
lam = 0.5 # \hbar^2/2m k_B
omega = 1
xmin = 0
numParticles = parsed_args['N']
numMCSteps = parsed_args['NumMCSteps']
numTimeSlices = parsed_args['NumTimeSlices']
tau = beta/(numTimeSlices)
binSize = 10
seed = parsed_args['seed']
