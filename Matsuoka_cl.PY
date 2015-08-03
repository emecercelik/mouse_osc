## Matsuoka Oscillator 

## Some parameters makes this oscillator stable
## This can be useful for some joints
## Relationship between parameters for oscillation
## 1+tau/T<a<1+b
## 'Analysis of neural oscillator', Matsuoka
## We can restrict the output inside certain values
## by using g function and c parameter

import numpy as np
import matplotlib.pyplot as plt

class Matsuoka:
    def __init__(self,numOsc=4,h=1e-2,tau=1e-2,T=1e-1,a=10.5,b=20.5,c=0.08,aa=3):
        self.h=h
        self.tau=tau
        self.T=T
        self.a=a
        self.b=b
        self.c=c
        self.A=aa*np.array([[0,-1,-1,1],[-1,0,1,-1],[-1,1,0,-1],[1,-1,-1,0]])
        self.numOsc=numOsc
        self.x=np.zeros((self.numOsc,1))+np.array([[0.1],[0.1],[0.2],[0.2]])#np.random.rand(numOsc,1)
        self.v=np.zeros((self.numOsc,1))+np.array([[0.1],[0.1],[0.2],[0.2]])#np.random.rand(numOsc,1)
        self.y=np.zeros((self.numOsc,1))+np.array([[0.1],[0.1],[0.2],[0.2]])#np.random.rand(numOsc,1)
        self.g=lambda x:max(0.,x)
        self.Record=0
        self.time=[]
        for i in range(self.numOsc):
            exec('self.yRec{0:d}=[]'.format(i))
    def iterate(self,Record=0):
        self.x+=self.h*(-self.x+self.c-self.A.dot(self.y)-self.b*self.v)/self.tau
        self.v+=self.h*(-self.v+self.y)/self.T
        self.Record=Record
        for i in range(self.numOsc):
            self.y[i]=self.g(self.x[i])
            if Record==1:
                exec('self.yRec{0:d}.append(float(self.y[{0:d}]))'.format(i))

    def plot(self):
        if self.Record==1:
            plt.plot(self.yRec0,'ro',linewidth=1.,label='1st output')
            plt.plot(self.yRec1,'--',linewidth=3.,label='2nd output')
            plt.plot(self.yRec2,'c*',label='3rd output')
            plt.plot(self.yRec3,'g',label='4th output')
            plt.title('Outputs of {0:d} neurons of Matsuoka Oscillator, h={1:.3f}\n'.format(self.numOsc,self.h)+\
              'tau={0:.3f}, T={1:.2f}, a={2:.2f}, b={3:.2f}, c={4:.2f}, '\
              .format(self.tau,self.T,self.a,self.b,self.c)+\
               'A matrix[{0:.1f},{1:.1f},{2:.1f},{3:.1f}]'.format(self.A[0,0],self.A[0,1],self.A[0,2],self.A[0,3])   )
            plt.xlabel('Time [step]')
            plt.ylabel('Value of outputs')
            plt.legend()
            plt.show()
        else:
            print('No records!')
    def output(self):
        output=[]
        for i in range(self.numOsc):
            output.append(float(self.y[i]))
        return output

kwargs={'numOsc':4,'h':1e-3,'tau':1e-2,'T':1e-1,'a':10.5,\
        'b':20.5,'c':0.08,'aa':3}

osc=Matsuoka(**kwargs)

stopTime=2000
for t in range(stopTime):
    osc.iterate(1)
