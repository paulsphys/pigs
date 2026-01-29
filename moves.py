import numpy as np

def SampleWindingSector(Path,StartBead,EndBead,StageLength):
    maxWind = 10
    numWind = 2*maxWind + 1
    windingsectors = np.arange(-maxWind,maxWind+1,1)
    weight = np.zeros(len(windingsectors))
    for i in range(len(windingsectors)):
        w = windingsectors[i]
        weight[i] = Path.TwoPointFreeAction(StartBead,EndBead+w*Path.Length,StageLength)
    weight = weight/np.sum(weight)
    wind = np.random.choice(windingsectors,p=weight)
    return wind

def CenterOfMassMove(Path,ptcl):
    '''
    Attempts a center of mass update, displacing an entire particle
    worldline.
    '''
    delta = 0.75
    shift = delta*(-1.0 + 2.0*np.random.random())

    # Store the positions on the worldline
    oldbeads = np.copy(Path.beads[:,ptcl])

    # Calculate the potential action
    oldAction = 0.0
    for tslice in range(Path.numTimeSlices):
        oldAction += Path.PotentialAction(tslice,ptcl)

    # Displace the worldline
    for tslice in range(Path.numTimeSlices):
        Path.beads[tslice,ptcl] = oldbeads[tslice] + shift

    # Compute the new action
    newAction = 0.0
    for tslice in range(Path.numTimeSlices):
        newAction += Path.PotentialAction(tslice,ptcl)
    #print("com: ",newAction - oldAction)
    # Accept the move, or reject and restore the bead positions
    if np.random.random() < np.exp(-(newAction - oldAction)):
        Path.beads[:,ptcl] = Path.PutPathInBC(Path.beads[:,ptcl])
        return True
    else:
        Path.beads[:,ptcl] = np.copy(oldbeads)
        return False

# ------------------------------------------------------------------------------------------- 
def StagingMove(Path,ptcl):
    """
    Perform staging on the bulk of the worldlines.
    """

    m = int(0.8*Path.numTimeSlices)

    # Choose the start and end of the stage
    # For PIGS we need to pick a point in the path 
    alpha_start = np.random.randint(0,Path.numTimeSlices-m)
    alpha_end = (alpha_start + m) % Path.numTimeSlices

    # Record the positions of the beads to be updated and store the action
    oldbeads = np.zeros(m-1)
    oldAction = 0.0
    for a in range(1,m):
        tslice = (alpha_start + a) % Path.numTimeSlices
        oldbeads[a-1] = Path.beads[tslice,ptcl]
        oldAction += Path.PotentialAction(tslice,ptcl)

    wind = SampleWindingSector(Path, Path.beads[alpha_start,ptcl], Path.beads[alpha_end,ptcl], m) 
    #print(wind)
    # Generate new positions and accumulate the new action
    newAction = 0.0;
    for a in range(1,m):
        tslice = (alpha_start + a) % Path.numTimeSlices
        tslicem1 = (tslice - 1) % Path.numTimeSlices
        tau1 = (m-a)*Path.tau
        avex = (tau1*Path.beads[tslicem1,ptcl] +
                Path.tau*(Path.beads[alpha_end,ptcl] + wind*Path.Length)) / (Path.tau + tau1)
        sigma2 = 2.0*Path.lam / (1.0 / Path.tau + 1.0 / tau1)
        val = avex + np.sqrt(sigma2)*np.random.randn()  
        Path.beads[tslice,ptcl] = val
        newAction += Path.PotentialAction(tslice,ptcl)
    #print("Before",Path.beads)
    #Path.plotConfig()
    #Path.unwrapped[alpha_end,ptcl] = Path.beads[alpha_end,ptcl] + wind*Path.Length
    Path.beads[:,ptcl] = Path.PutPathInBC(Path.beads[:,ptcl])
    #print("After",Path.beads)
    #Path.plotConfig()
    # Perform the Metropolis step, if we rejct, revert the worldline
    #print("Staging: ",newAction-oldAction)
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        for a in range(1,m):
            tslice = (alpha_start + a) % Path.numTimeSlices
            Path.beads[tslice,ptcl] = oldbeads[a-1]
        return False
# ------------------------------------------------------------------------------------------- 
def EndStagingMove(Path,ptcl):
    """
    Perform staging on the head or tail of the worldlines.
    """ 
    if (np.random.rand() < 0.5):
        beadIndex = 0
        startbead = True
    else:
        beadIndex = Path.numTimeSlices - 1
        startbead = False

    oldAction = Path.PotentialAction(beadIndex,ptcl)
    if(startbead):
        gaussmean = Path.beads[beadIndex + 1,Path.nextLink[beadIndex,ptcl]]
    else:
        gaussmean = Path.beads[beadIndex - 1,Path.prevLink[beadIndex,ptcl]]
   
    newBead = np.random.normal(loc = gaussmean, scale = np.sqrt(2*Path.lam*Path.tau))
    #Put it in BC
    #newBead = (newBead + Path.Length/2) % Path.Length - Path.Length/2
    
    while (not newBead <= Path.Length/2): 
        if newBead > Path.Length/2:
            newBead -= Path.Length
        elif newBead < Path.Length/2:
            newBead += Path.Length
    
    oldbead = Path.beads[beadIndex,ptcl]
    Path.beads[beadIndex,ptcl] = newBead
    newAction = Path.PotentialAction(beadIndex,ptcl)
    if np.random.random() < np.exp(-(newAction - oldAction)):
        return True
    else:
        Path.beads[beadIndex,ptcl] = oldbead
        return False
