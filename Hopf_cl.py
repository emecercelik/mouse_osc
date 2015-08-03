# "Pattern generators with sensory feedback for the control of
# quadruped locomotion", Ludovic Righetti and Auke Jan Ijspeert
import numpy as np
import matplotlib.pyplot as plt

class Hopf:
    def __init__(self,numOsc=4,h=1e-1,alpha=5.,beta=50.,mu=1.,w_stance=10.,w_swing=1.,b=10.,F=300,\
                 feedback=0,gait=0):
        self.h=h
        self.numOsc=numOsc
        self.alpha=alpha
        self.beta=beta
        self.mu=mu
        self.w_stance=w_stance # Changes the stance phase
        self.w_swing=w_swing
        self.b=b # Changes the shape of periodic signal
        self.F=F
        self.feedback=feedback
                
        # Coupling matrices for different gaits
        self.gait=gait
        self.gaitType=['Trot','Pace','Bound','Walk']
        if self.numOsc==4:
            if self.gait==0:
                self.K=np.array([[0,-1,-1,1],[-1,0,1,-1],[-1,1,0,-1],[1,-1,-1,0]]) #Trot
            elif self.gait==1:
                self.K=np.array([[0,-1,1,-1],[-1,0,-1,1],[1,-1,0,-1],[-1,1,-1,0]]) #Pace
            elif self.gait==2:
                self.K=np.array([[0,1,-1,-1],[1,0,-1,-1],[-1,-1,0,1],[-1,-1,1,0]]) #Bound
            else:
                self.K=np.array([[0,-1,1,-1],[-1,0,-1,1],[-1,1,0,-1],[1,-1,-1,0]]) #Walk
        else:
            self.K=1-np.eye(numOsc)

        

        self.x=np.zeros((self.numOsc,1))+np.random.rand(self.numOsc,1)
        self.y=np.zeros((self.numOsc,1))+np.random.rand(self.numOsc,1)
        self.u=np.zeros((self.numOsc,1))
        self.r=np.sqrt(self.x**2+self.y**2)
        self.w=self.w_stance/(np.exp(-self.b*self.y)+1)+self.w_swing/(np.exp(self.b*self.y)+1)
        self.xRec=np.array(self.x)
        self.yRec=np.array(self.y)
        self.rRec=np.array(self.r)
        
    def iterate(self,Record=0):
        if self.feedback==1:
            for ii in range(self.numOsc):
                if self.y[ii]<0.25*self.r[ii] and self.y[ii]>-0.25*self.r[ii]:
                    self.u[ii]=-self.w[ii]*self.x[ii]-self.K.dot(self.y)[ii]
                elif (self.y[ii]>0. and self.x[ii]<0.) or (self.y[ii]<0. and self.x[ii]>0.):
                    self.u[ii]=-np.sign(self.y[ii])*self.F
            else:
                self.u[ii]=0.
        self.x+=self.h*(self.alpha*(self.mu-self.r**2)*self.x-self.w*self.y)
        self.y+=self.h*(self.beta*(self.mu-self.r**2)*self.y+self.w*self.x+self.K.dot(self.y)+self.u)
        self.w=self.w_stance/(np.exp(-self.b*self.y)+1)+self.w_swing/(np.exp(self.b*self.y)+1)
        self.r=np.sqrt(self.x**2+self.y**2)

        self.Record=Record
        if self.Record==1:
            self.xRec=np.hstack((self.xRec,self.x))
            self.yRec=np.hstack((self.yRec,self.y))
            self.rRec=np.hstack((self.rRec,self.r))

    def plot(self):
        if self.Record==1:
            timeStop=self.xRec.shape[1]-1
            time=np.arange(0,timeStop+1)
            pltxAxisStep=(timeStop)/10
            ####Plot1##############################
            plt.figure(1,figsize=(15,10))
            plt.suptitle('Hopf Oscillator\n'+\
            'h={0:.3f}, alpha={1:.2f}, beta={2:.2f},'.format(self.h,self.alpha,self.beta)+\
            ' mu={0:.2f}, w_stance={1:.2f}, w_swing={2:.2f}, b={3:.2f}'\
            .format(self.mu,self.w_stance,self.w_swing,self.b)+\
            '\nGait Type={0:s}'.format(self.gaitType[self.gait]))
            plt.subplot(421)
            plt.plot(time,self.xRec[0,:],'b',label='x1')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.xlabel('Time step')
            plt.ylabel('Value of state variables')
            plt.legend()
            plt.subplot(423)
            plt.plot(time,self.xRec[1,:],'k',label='x2')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.legend()
            plt.subplot(425)
            plt.plot(time,self.xRec[2,:],'g',label='x3')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.legend()
            plt.subplot(427)
            plt.plot(time,self.xRec[3,:],'m',label='x4')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.legend()
            plt.xlabel('Time step')
            plt.ylabel('Value of state variables')
            plt.subplot(422)
            plt.plot(time,self.yRec[0,:],'b',label='y1')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.legend()
            plt.subplot(424)
            plt.plot(time,self.yRec[1,:],'k',label='y2')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.legend()
            plt.subplot(426)
            plt.plot(time,self.yRec[2,:],'g',label='y3')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.legend()
            plt.subplot(428)
            plt.plot(time,self.yRec[3,:],'m',label='y4')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.legend()
            ####Plot2##############################
            plt.figure(2,figsize=(15,10))
            plt.suptitle('Hopf Oscillator\n'+\
            'h={0:.3f}, alpha={1:.2f}, beta={2:.2f},'.format(self.h,self.alpha,self.beta)+\
            ' mu={0:.2f}, w_stance={1:.2f}, w_swing={2:.2f}, b={3:.2f}'\
            .format(self.mu,self.w_stance,self.w_swing,self.b)+\
            '\nGait Type={0:s}'.format(self.gaitType[self.gait]))
            plt.subplot(211)
            plt.plot(time,self.xRec[0,:],'b',label='x1')
            plt.plot(time,self.xRec[1,:],'k',label='x2')
            plt.plot(time,self.xRec[2,:],'g',label='x3')
            plt.plot(time,self.xRec[3,:],'m',label='x4')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.xlabel('Time step')
            plt.ylabel('Value of state variables')
            plt.legend()
            plt.subplot(212)
            plt.plot(time,self.yRec[0,:],'b',label='y1')
            plt.plot(time,self.yRec[1,:],'k',label='y2')
            plt.plot(time,self.yRec[2,:],'g',label='y3')
            plt.plot(time,self.yRec[3,:],'m',label='y4')
            plt.xticks(np.arange(0,timeStop,pltxAxisStep))
            plt.xlabel('Time step')
            plt.ylabel('Value of state variables')
            plt.legend()
            ####Plot3##############################
            plt.figure(3)
            plt.plot(self.rRec[0,:])
            plt.plot(self.rRec[1,:])
            plt.plot(self.rRec[2,:])
            plt.plot(self.rRec[3,:])
            plt.show()
        else:
            print('No records!')
    def output(self):
        output=[]
        temp_y=(self.y+1)*0.5
        for i in range(self.numOsc):
            output.append(float(temp_y[i]))
        return output

kwargs={'numOsc':4,'h':1e-3,'alpha':5.,'beta':50.,'mu':1.,'w_stance':10.,'w_swing':4.,'b':10.,'F':300,\
                 'feedback':0,'gait':0}

osc=Hopf(**kwargs)
xRec=np.array(osc.x)
yRec=np.array(osc.y)
rRec=np.array(osc.r)
stopTime=20000
timeStop=stopTime
for t in range(stopTime):
    osc.iterate(1)

osc.plot()
