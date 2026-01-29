import numpy as np
import scipy.special as scp

class Paths:
    '''The set of worldlines, action and estimators.'''
    def __init__(self,beads,tau,lam,L):
        self.tau = tau
        self.lam = lam
        self.beads = np.copy(beads)
        self.numTimeSlices = len(beads)
        self.numParticles = len(beads[0])
        self.nextLink = np.transpose(np.array([i*np.ones(self.numTimeSlices,dtype=np.int64) for i in range(self.numParticles)]))
        self.prevLink = np.transpose(np.array([i*np.ones(self.numTimeSlices,dtype=np.int64) for i in range(self.numParticles)]))
        self.Length = L
        self.rc2 = 4*4
        self.g = 1
        self.flag = True

    def SetPotential(self,externalPotentialFunction):
        '''The potential function. '''
        self.VextHelper = externalPotentialFunction

    def Vext(self,R):
        '''The external potential energy.'''
        return self.VextHelper(R)
    
    def deltaSeparation(self,sep1,sep2):
        delta = sep2-sep1
        while (delta >= 0.5*self.Length):
            delta -= self.Length
        while (delta < -0.5*self.Length):
            delta += self.Length

        return delta

    def PotentialAction(self,tslice,ptcl):
        '''The potential action.'''
        Uext = 0.0
        Uint = 0.0
        totU = 0.0
        Uext += self.Vext(self.beads[tslice,ptcl]) 
        Uext = self.tau*Uext
        totU += Uext
        if(tslice == 0 or tslice == self.numTimeSlices - 1):
            totU -= np.log(self.trial(self.beads[tslice,ptcl]))
        #print(ptcl,self.numParticles)
        tslicep1 = (tslice + 1) % self.numTimeSlices
        for ptcl2 in range(ptcl+1,self.numParticles):
            sep = self.beads[tslice,ptcl] - self.beads[tslice,ptcl2]                
            if (sep**2 < self.rc2):
                sep2 = self.beads[tslicep1,ptcl] - self.beads[tslicep1,ptcl2]
                Uint += self.PairProductPotential(sep,sep2)
        #print(Uint)
        totU += Uint
        return totU
   
    def PairProductPotential(self,sep1,sep2):
        erfCO = 7.0;
        l0 = 2.0*np.sqrt(self.lam*self.tau);
        li = 4.0*self.lam/self.g;
        xi = l0/li;
        xiSqOver2 = (0.5)*xi*xi;
        xiSqrtPIOver2 = np.sqrt(np.pi/2.0)*xi;
        dxt = self.deltaSeparation(sep1, sep2)/l0;
        yt = (abs(sep1)+abs(sep2))/l0;
        
        if(xi + yt < erfCO):
            erfVal = scp.erfc( (xi+yt)/np.sqrt(2.0) )
            expVal = np.exp( xiSqOver2 + (0.5)*dxt*dxt + xi*yt )
            W = 1.0 - xiSqrtPIOver2*expVal*erfVal
            
        else:
            expVal = np.exp((-0.5)*(yt*yt-dxt*dxt))
            W = 1.0 - (xi/(xi+yt))*expVal
        return (-1.0)*np.log(W);

    def dWdxi(self, yt, dxt):
    
        expVal = np.exp((0.5)*(dxt*dxt-yt*yt) )
        erfCO = 7.0;
        l0 = 2.0*np.sqrt(self.lam*self.tau);
        li = 4.0*self.lam/self.g;
        xi = l0/li;
        if(xi + yt < erfCO):
            erfVal = scp.erfc( (xi+yt)/np.sqrt(2.0) )
            expVal2 = np.exp( (0.5)*(xi+yt)*(xi+yt) )
            dW = (-1.0)*xi*expVal*(np.sqrt(0.5*np.pi)*(xi +yt + (1.0/xi) )*expVal2*erfVal - 1.0 )
        else:
            dW = (-1.0)*xi*expVal*( (xi + yt + (1.0/xi) )/( xi + yt ) - 1.0 )
    
        return dW
    
    def dWdyt(self, yt, dxt):
        expVal = np.exp( (0.5)*(dxt*dxt-yt*yt) );
        erfCO = 7.0;
        l0 = 2.0*np.sqrt(self.lam*self.tau);
        li = 4.0*self.lam/self.g;
        xi = l0/li;

        if(xi + yt < erfCO):
            erfVal = scp.erfc( (xi+yt)/np.sqrt(2.0) )
            expVal2 = np.exp( (0.5)*(xi+yt)*(xi+yt) )
            dW = xi*expVal*( 1 - np.sqrt(0.5*np.pi)*xi*expVal2*erfVal )
        else:
            dW = xi*expVal*( 1 - xi/(xi+yt) )

        return dW

    def dWddxt(self, yt, dxt):

        erfCO = 7.0
        l0 = 2.0*np.sqrt(self.lam*self.tau)
        li = 4.0*self.lam/self.g
        xi = l0/li
        if (xi + yt < erfCO):
            erfVal = scp.erfc( (xi+yt)/np.sqrt(2.0) )
            expVal = np.exp( (0.5)*( dxt*dxt + xi*( xi + 2.0*yt)) )
            dW = (-1.0)*np.sqrt(0.5*np.pi)*xi*dxt*expVal*erfVal    
        else:
            expVal = np.exp( (0.5)*(dxt*dxt-yt*yt) )
            dW = (-1.0)*dxt*expVal*( xi/( xi + yt ) )
        return dW


    def KineticEnergy(self):
        M = self.numTimeSlices
        N = self.numParticles

        sum_sq = 0.0
        for tslice in range(M - 1):
            for ptcl in range(N):
                dR = self.beads[tslice+1, ptcl] - self.beads[tslice, ptcl]
                dR = self.DistInBC(dR)
                sum_sq += dR * dR

        KE = (N * (M - 1)) / (2.0 * M * self.tau) - sum_sq / (4.0 * self.lam * self.tau * self.tau * M)

        return KE

    def PairProductEnergy(self, sep1, sep2):
        
        l0 = 2.0*np.sqrt(self.lam*self.tau)
        li = 4.0*self.lam/self.g
        xi = l0/li
        dxt = self.deltaSeparation(sep1, sep2)/l0;
        yt = (abs(sep1)+abs(sep2))/l0;
    
        W = np.exp(-self.PairProductPotential(yt,dxt))
        dWdt = ((1.0)/(2.0*self.tau)) * ( (-1.0)*yt*self.dWdyt(yt,dxt) - dxt*self.dWddxt(yt,dxt) + xi*self.dWdxi(yt,dxt) )

        return ((-1.0)/W)*dWdt
    def PotentialEnergy(self):
        '''The operator potential energy estimator.'''
        totU = 0.0
        Vext = 0.0
        Epair = 0.0
        for tslice in range(self.numTimeSlices):
            tslicep1 = (tslice + 1) % self.numTimeSlices
            for ptcl in range(self.numParticles):
                R = self.beads[tslice,ptcl]
                Vext = Vext + self.Vext(R)
        Vext = Vext/(self.numTimeSlices)
        totU += Vext
        for tslice in range(self.numTimeSlices - 1):
            for i in range(self.numParticles):
                for j in range(i + 1, self.numParticles):
                    sep1 = self.beads[tslice, i] - self.beads[tslice, j]
                    sep2 = (self.beads[tslice + 1, i] - self.beads[tslice + 1, j])
                    if sep1**2 < self.rc2:
                        Epair += self.PairProductEnergy(sep1, sep2)
        Epair /= (self.numTimeSlices)
        totU += Epair
        for tslice in [0, self.numTimeSlices-1]:
            for ptcl in range(self.numParticles):
                totU -= np.log(self.trial(self.beads[tslice, ptcl]))

        return totU

    def Energy(self):
        '''The total energy.'''
        return self.KineticEnergy() + self.PotentialEnergy()

    def TwoPointFreeAction(self,bead1,bead2,seperation):
        act = np.exp(-(bead1-bead2)**2/(4*self.lam*self.tau*seperation))/np.sqrt(4*np.pi*self.lam*self.tau*seperation)
        return act

    def trial(self,bead):
        return 1

    def PutPathInBC(self,pathsection):
        while (not np.all(np.abs(pathsection) <= self.Length/2)): 
            OutsideRightBoxMask = pathsection > self.Length/2
            OutsideLeftBoxMask = pathsection < -self.Length/2
            pathsection[OutsideRightBoxMask] = pathsection[OutsideRightBoxMask] - self.Length 
            pathsection[OutsideLeftBoxMask] = pathsection[OutsideLeftBoxMask] + self.Length
        return pathsection
    
    def DistInBC(self,dist):
        return dist - self.Length * np.floor(dist*(1/self.Length) + 0.5)

    def plotConfig(self):
        """
        Plots a configuration, useful for debugging
        """
        
        import matplotlib.pyplot as plt
        for ptcl in range(self.numParticles):
            for tslice in range(self.numTimeSlices):
                nextslice = (tslice + 1) % self.numTimeSlices
                bead = self.beads[tslice,ptcl]
                nextbead = self.beads[nextslice, self.nextLink[tslice,ptcl]]
                if(nextslice == 0) : nextslice = self.numTimeSlices;
                plt.plot([bead,nextbead],[tslice,nextslice],color = 'r', ls='None',marker='o')
                
        
        plt.ylabel('Imaginary time')
        plt.xlabel('r')
        plt.xlim(-self.Length/2,self.Length/2)
        plt.show()

# ------------------------------------------------------------------------------------------- 
