from moves import *
from constants import *
from outproc import binproc

def PIMC(numSteps,Path):
    '''Perform a path integral Monte Carlo simulation of length numSteps.'''
    observableSkip = 10
    equilSkip = 1000
    numAccept = {'CenterOfMass':0,'Staging':0, 'End Staging':0}
    loopcounter = []
    KinEnergyTrace = []
    PotEnergyTrace = []
    CalcEThermo = Path.Energy()
    StdEThermo = 1
    steps = 0
    #print(CalcEThermo)
    #while np.abs(StdEThermo/CalcEThermo) > 0.01 or steps < equilSkip + 100: 
    for steps in range(0,numSteps):
        # for each particle try a center-of-mass move
        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles):
                numAccept['CenterOfMass'] += CenterOfMassMove(Path,ptcl)

        # for each particle try a staging move
        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles): 
            numAccept['Staging'] += StagingMove(Path,ptcl)

        for ptcl in np.random.randint(0,Path.numParticles,Path.numParticles): 
            numAccept['End Staging'] += EndStagingMove(Path,ptcl)
        
        # measure the energy
        if steps % observableSkip == 0 and steps > equilSkip:
            KinEnergyTrace.append(Path.KineticEnergy())
            PotEnergyTrace.append(Path.PotentialEnergy())
            if (len(KinEnergyTrace) > 5):
                CalcEThermo,StdEThermo = binproc(PotEnergyTrace,1)
        steps += 1        
    print('Acceptance Ratios:')
    print('Center of Mass: %4.3f' % ((1.0*numAccept['CenterOfMass'])/(numSteps*Path.numParticles)))
    print('Staging:        %1.3f\n' % ((1.0*numAccept['Staging'])/(numSteps*Path.numParticles)))
    #print('Average loopcounter: %1.3f\n' % np.mean(np.array(loopcounter)))
    return np.array(PotEnergyTrace),np.array(KinEnergyTrace)

