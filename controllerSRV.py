# =================================================================================================================================================
#                                       Import modules
##import sys
##sys.path.append('/home/ercelik/opt1/nest/lib/python3.4/site-packages/')
##import nest
import pickle
import random
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

#import Matsuoka_cl as OSC
import Hopf_cl as OSC

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

def ActFunc(n,maxim,minim,value,off):
    return np.exp((-n/(3*(maxim-minim)))*(value-off)**2)

def NeuronFunc(x):
    return (1.+np.tanh(x-1.6))*.5 # 4.=9 input

def InputFunc(x,dim,nAct,maxAcc,minAcc):
    rangeAcc=maxAcc-minAcc
    off=np.arange(minAcc+rangeAcc/(2.*nAct),maxAcc,rangeAcc/nAct)
    res=np.array([[ActFunc(nAct,maxAcc,minAcc,x[j],off[i]) for i in range(len(off))] for j in range(len(x))])
    res=res.reshape(res.size,1)
    for i in range(res.size):
        if res[i]<1e-2:
            res[i]=.0
    return res


bpy.context.scene.game_settings.fps=60.
dt=1./bpy.context.scene.game_settings.fps


#nest.sli_func('synapsedict info')
# =================================================================================================================================================
#                                       Creating muscles


#~ servo_ids = {}
#~ servo_ids["forearm.L"] = setVelocityServo(reference_object_name = "obj_forearm.L",  attached_object_name = "obj_upper_arm.L",  maxV = 10.0)
PP=120.

servo_ids = {}
servo_ids["wrist.L"]      = setPositionServo(reference_object_name = "obj_wrist.L",      attached_object_name = "obj_forearm.L", P = PP)
servo_ids["wrist.R"]      = setPositionServo(reference_object_name = "obj_wrist.R",      attached_object_name = "obj_forearm.R", P = PP)
servo_ids["forearm.L"]    = setPositionServo(reference_object_name = "obj_forearm.L",    attached_object_name = "obj_upper_arm.L", P = PP)
servo_ids["forearm.R"]    = setPositionServo(reference_object_name = "obj_forearm.R",    attached_object_name = "obj_upper_arm.R", P = PP)

servo_ids["upper_arm.L"]  = setPositionServo(reference_object_name = "obj_upper_arm.L",  attached_object_name = "obj_shoulder.L", P = PP)
servo_ids["upper_arm.R"]  = setPositionServo(reference_object_name = "obj_upper_arm.R",  attached_object_name = "obj_shoulder.R", P = PP)
servo_ids["shin_lower.L"] = setPositionServo(reference_object_name = "obj_shin_lower.L", attached_object_name = "obj_shin.L", P = PP)
servo_ids["shin_lower.R"] = setPositionServo(reference_object_name = "obj_shin_lower.R", attached_object_name = "obj_shin.R", P = PP)

servo_ids["shin.L"]       = setPositionServo(reference_object_name = "obj_shin.L",       attached_object_name = "obj_thigh.L", P = PP)
servo_ids["shin.R"]       = setPositionServo(reference_object_name = "obj_shin.R",       attached_object_name = "obj_thigh.R", P = PP)
servo_ids["thigh.L"]       = setPositionServo(reference_object_name = "obj_thigh.L",     attached_object_name = "obj_hips", P = PP)
servo_ids["thigh.R"]       = setPositionServo(reference_object_name = "obj_thigh.R",     attached_object_name = "obj_hips", P = PP)


# =================================================================================================================================================
#                                       Network creation

#np.random.seed(np.random.randint(0,10000))

ax_avg=0.1 #Global parameter to calculate avg acceleration
ay_avg=0.1
az_avg=0.1

ax=.1 #Global parameter to record acceleration
ay=.1
az=.1

# PID Parameters
nJoint=12 #number of joints that is controlled
kP=np.array([.2 for i in range(nJoint)]) #coefficients of PID controller
kI=np.array([.03 for i in range(nJoint)])
kD=np.array([.1 for i in range(nJoint)])

e_old=np.array([0. for i in range(nJoint)]) #The error one step before
E=np.array([0. for i in range(nJoint)]) #Integrated error
uu=np.array([0. for i in range(nJoint)]) # Control input initialization

inpp=[] # Record inputs of servos to be plotted
outp=[] # Record outputs of servos to be plotted 
cont_inp=[] # Reecord control inputs to servos to be plotted

# Joints
#Joint names
joints=["wrist.L","wrist.R","forearm.L","forearm.R","upper_arm.L","upper_arm.R",\
        "shin_lower.L","shin_lower.R","shin.L","shin.R","thigh.L","thigh.R"]
numRec=3 # The number of joint to be plotted


############ Matsuoka Osc param ###########
numOsc=4 # Number of Oscillator

h=1e-2 # bigger h increases the amplitude thats why i decrease c value
tau=0.01
T=0.1
a=10.5
b=20.5
c=0.08
A=3*np.array([[0,-1,-1,1],[-1,0,1,-1],[-1,1,0,-1],[1,-1,-1,0]])

# Coefficient of A matrix is an important parameter to determine
# the phase and frequency
# h has to be selected properly for the solution of diff equ
# tau and T has to be coherent with h and they change freq
# c changes amplitude of the oscillation

# Initials of oscillators
x=np.zeros((numOsc,1))+np.array([[0.1],[0.1],[0.2],[0.2]])
v=np.zeros((numOsc,1))+np.array([[0.1],[0.1],[0.2],[0.2]])
y=np.zeros((numOsc,1))+np.array([[0.1],[0.1],[0.2],[0.2]])

g=lambda x:max(0,x)

yRec0=[]
yRec1=[]
yRec2=[]
yRec3=[]

##kwargs_mats={'numOsc':4,'h':1e-2,'tau':1e-2,'T':1e-1,'a':10.5,\
##        'b':20.5,'c':0.08,'aa':3}
##osc=OSC.Matsuoka(**kwargs_mats)
kwargs_hopf={'numOsc':4,'h':1e-2,'alpha':5.,'beta':50.,'mu':1.,'w_stance':10.,'w_swing':10.,'b':10.,'F':300,\
                 'feedback':0,'gait':0}
osc=OSC.Hopf(**kwargs_hopf)

ff=lambda x,a: min(x,a)
###########################################

# =================================================================================================================================================
#                                       Evolve function
def evolve():
    # Global variable definitions
    global ax,ay,az,ax_avg,ay_avg,az_avg
    global inpp,outp,PP
    global joints,nJoint,numRec
    global h,tau,T,a,b,c,x,v,y,g,numOsc
    global yRec0,yRec1,yRec2,yRec3,ff
    
    print("Step:", i_bl, "  Time:{0:.2f}".format(t_bl),'   Acc:{0:8.2f}  {1:8.2f}  {2:8.2f}'.format(ax,ay,az),\
          'Acc:{0:8.2f} {1:8.2f} {2:8.2f}'.format(ax_avg,ay_avg,az_avg))
    # ------------------------------------- Visual ------------------------------------------------------------------------------------------------
    #visual_array     = getVisual(camera_name = "Meye", max_dimensions = [256,256])
    #scipy.misc.imsave("test_"+('%05d' % (i_bl+1))+".png", visual_array)
    # ------------------------------------- Olfactory ---------------------------------------------------------------------------------------------
    olfactory_array  = getOlfactory(olfactory_object_name = "obj_nose", receptor_names = ["smell1", "plastic1"])
    # ------------------------------------- Taste -------------------------------------------------------------------------------------------------
    taste_array      = getTaste(    taste_object_name =     "obj_mouth", receptor_names = ["smell1", "plastic1"], distance_to_object = 1.0)
    # ------------------------------------- Vestibular --------------------------------------------------------------------------------------------
    vestibular_array = getVestibular(vestibular_object_name = "obj_head")
    #print (vestibular_array)
    # ------------------------------------- Sensory -----------------------------------------------------------------------------------------------
    # ------------------------------------- Proprioception ----------------------------------------------------------------------------------------
    #~ spindle_FLEX = getMuscleSpindle(control_id = muscle_ids["forearm.L_FLEX"])
    #~ spindle_EXT  = getMuscleSpindle(control_id = muscle_ids["forearm.L_EXT"])
    # ------------------------------------- Neural Simulation -------------------------------------------------------------------------------------
    
    
    ax=vestibular_array[3] # Get instant accelerations
    ay=vestibular_array[4]
    az=vestibular_array[5]

    ax_avg=(ax_avg*(i_bl)+ax)/(i_bl+1) # Calculate avg accelerations
    ay_avg=(ay_avg*(i_bl)+ay)/(i_bl+1)
    az_avg=(az_avg*(i_bl)+az)/(i_bl+1)


    # ------------------------------------- Muscle Activation -------------------------------------------------------------------------------------
    
    speed_ = 6.0 # Speed of the mouse (ang. freq of joint patterns)

    # Joint signals to be applied
    act_tmp         = 0.5 + 0.5*np.sin(speed_*t_bl)
    anti_act_tmp    = 1.0 - act_tmp
    act_tmp_p1      = 0.5 + 0.5*np.sin(speed_*t_bl - np.pi*0.5)
    anti_act_tmp_p1 = 1.0 - act_tmp_p1
    act_tmp_p2      = 0.5 + 0.5*np.sin(speed_*t_bl + np.pi*0.5)
    anti_act_tmp_p2 = 1.0 - act_tmp_p2

    
    ### Oscillator
##    x=x+h*(-x+c-A.dot(y)-b*v)/tau
##    v=v+h*(-v+y)/T
##    for i in range(numOsc):
##        y[i]=float(g(x[i]))
    osc.iterate(1)
    y=osc.output()


    joints=["wrist.L","wrist.R","forearm.L","forearm.R",\
            "upper_arm.L","upper_arm.R","shin_lower.L","shin_lower.R",\
            "shin.L","shin.R","thigh.L","thigh.R"]
    # Reference value of joints
##    r=np.array([0.4,0.4,0.8*act_tmp,0.8*anti_act_tmp,1.0*act_tmp_p1,1.0*anti_act_tmp_p1,0.8*anti_act_tmp,\
##                0.8*act_tmp,0.5*anti_act_tmp_p1,0.5*act_tmp_p1,0.5*anti_act_tmp,0.5*act_tmp])
    
##    r=np.array([0.4,0.4,0.8*y[0],0.8*y[2],\
##                y[0],y[2],0.8*y[3], 0.8*y[1],\
##                0.5*y[3],0.5*y[1],0.5*y[3],0.5*y[1]])

    
    r=np.array([0.4,0.4,0.8*y[0],0.8*y[2],\
                y[0],y[2],0.8*y[3], 0.8*y[1],\
                0.5*y[3],0.5*y[1],0.5*y[3],0.5*y[1]])    
    
##    r=np.array([0.6,0.6,np.mod(0.8*y[0]+0.4,0.8),np.mod(0.8*y[2]+0.4,0.8),\
##                y[0],y[2],np.mod(y[3]+0.5,1), np.mod(y[1]+0.5,1),\
##                y[3],y[1],0.5,0.5])#np.mod(y[3]+0.3,1),np.mod(y[1]+0.3,1)

    positions=np.array([getMuscleSpindle(control_id = servo_ids[joints[i]])[0] for i in range(len(joints))])
    #print(positions)
    
    
    for i in range(nJoint):
        controlActivity(control_id = servo_ids[joints[i]], control_activity = r[i])
        # Apply the reference 


    # Get actual positions of joints after the control input
    positions=np.array([getMuscleSpindle(control_id = servo_ids[joints[i]])[0] for i in range(len(joints))])
    #print(positions)    

    
    numRec=3
    inpp.append(r[numRec]) # Record reference
    outp.append(positions[numRec]) # record positions of servos

    if i_bl==250:
        data=np.hstack((np.array(inpp),np.array(outp),joints[numRec],PP,speed_))
        PickleIt(data,'inpOut') # Save the data to be plotted
    if np.mod(i_bl,250)==0.:
        osc.plot()

