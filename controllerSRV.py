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


#####################################################################################################################
#####################################################################################################################
    
##    controlActivity(control_id = muscle_ids["wrist.L_FLEX"], control_activity = .4) #act4)# fL
##    controlActivity(control_id = muscle_ids["wrist.L_EXT"] , control_activity = .6) #anti_act4)# fL
##    controlActivity(control_id = muscle_ids["wrist.R_FLEX"], control_activity = .4)
##    controlActivity(control_id = muscle_ids["wrist.R_EXT"] , control_activity = .6)
##    controlActivity(control_id = muscle_ids["forearm.L_FLEX"], control_activity = 0.8*act_tmp)#act5)#fAL) #   #fL
##    controlActivity(control_id = muscle_ids["forearm.L_EXT"] , control_activity = 1.-0.8*act_tmp) #act5)#anti_fAL) # #fL
##    controlActivity(control_id = muscle_ids["forearm.R_FLEX"], control_activity = 0.8*anti_act_tmp) #
##    controlActivity(control_id = muscle_ids["forearm.R_EXT"] , control_activity = 1.-0.8*anti_act_tmp) #
##    controlActivity(control_id = muscle_ids["upper_arm.L_FLEX"], control_activity = 1.0*act_tmp_p1)#act6)#uAL) # #fL
##    controlActivity(control_id = muscle_ids["upper_arm.L_EXT"] , control_activity = 1.-1.0*act_tmp_p1)#act6)#anti_uAL) # #fL
##    controlActivity(control_id = muscle_ids["upper_arm.R_FLEX"], control_activity = 1.0*anti_act_tmp_p1) #
##    controlActivity(control_id = muscle_ids["upper_arm.R_EXT"] , control_activity = 1.-1.0*anti_act_tmp_p1) #
##    controlActivity(control_id = muscle_ids["shin_lower.L_FLEX"], control_activity = 0.8*act6)#0.8*anti_act_tmp)
##    controlActivity(control_id = muscle_ids["shin_lower.L_EXT"] , control_activity = anti_act6)#1-0.8*anti_act_tmp)
##    controlActivity(control_id = muscle_ids["shin_lower.R_FLEX"], control_activity = 0.8*act3)#0.8*act_tmp)#
##    controlActivity(control_id = muscle_ids["shin_lower.R_EXT"] , control_activity = anti_act3)#1.-0.8*act_tmp)#
##    controlActivity(control_id = muscle_ids["shin.L_FLEX"], control_activity = act5)#0.5*anti_act_tmp_p1)
##    controlActivity(control_id = muscle_ids["shin.L_EXT"] , control_activity = anti_act5)#1.-0.5*anti_act_tmp_p1)
##    controlActivity(control_id = muscle_ids["shin.R_FLEX"], control_activity = act2)#0.5*act_tmp_p1)#
##    controlActivity(control_id = muscle_ids["shin.R_EXT"] , control_activity = anti_act2)#1.-0.5*act_tmp_p1)#
##    controlActivity(control_id = muscle_ids["thigh.L_FLEX"], control_activity = act4)#0.5*anti_act_tmp)
##    controlActivity(control_id = muscle_ids["thigh.L_EXT"] , control_activity = anti_act4)#1.-0.5*anti_act_tmp)
##    controlActivity(control_id = muscle_ids["thigh.R_FLEX"], control_activity = act1)#0.5*act_tmp)#
##    controlActivity(control_id = muscle_ids["thigh.R_EXT"] , control_activity = anti_act1)#1.-0.5*act_tmp)#


##    


#~ servo_ids = {}
#~ servo_ids["forearm.L"] = setVelocityServo(reference_object_name = "obj_forearm.L",  attached_object_name = "obj_upper_arm.L",  maxV = 10.0)
##PP=120.
##
##servo_ids = {}
##servo_ids["wrist.L"]      = setPositionServo(reference_object_name = "obj_wrist.L",      attached_object_name = "obj_forearm.L", P = PP)
##servo_ids["wrist.R"]      = setPositionServo(reference_object_name = "obj_wrist.R",      attached_object_name = "obj_forearm.R", P = PP)
##servo_ids["forearm.L"]    = setPositionServo(reference_object_name = "obj_forearm.L",    attached_object_name = "obj_upper_arm.L", P = PP)
##servo_ids["forearm.R"]    = setPositionServo(reference_object_name = "obj_forearm.R",    attached_object_name = "obj_upper_arm.R", P = PP)
##
##servo_ids["upper_arm.L"]  = setPositionServo(reference_object_name = "obj_upper_arm.L",  attached_object_name = "obj_shoulder.L", P = PP)
##servo_ids["upper_arm.R"]  = setPositionServo(reference_object_name = "obj_upper_arm.R",  attached_object_name = "obj_shoulder.R", P = PP)
##servo_ids["shin_lower.L"] = setPositionServo(reference_object_name = "obj_shin_lower.L", attached_object_name = "obj_shin.L", P = PP)
##servo_ids["shin_lower.R"] = setPositionServo(reference_object_name = "obj_shin_lower.R", attached_object_name = "obj_shin.R", P = PP)
##
##servo_ids["shin.L"]       = setPositionServo(reference_object_name = "obj_shin.L",       attached_object_name = "obj_thigh.L", P = PP)
##servo_ids["shin.R"]       = setPositionServo(reference_object_name = "obj_shin.R",       attached_object_name = "obj_thigh.R", P = PP)
##servo_ids["thigh.L"]       = setPositionServo(reference_object_name = "obj_thigh.L",     attached_object_name = "obj_hips", P = PP)
##servo_ids["thigh.R"]       = setPositionServo(reference_object_name = "obj_thigh.R",     attached_object_name = "obj_hips", P = PP)


    # Torque-based muscles
    
##    controlActivity(control_id = muscle_ids["wrist.L_FLEX"], control_activity = outV[0])
##    controlActivity(control_id = muscle_ids["wrist.L_EXT"] , control_activity = 1.0-outV[0])
##    controlActivity(control_id = muscle_ids["wrist.R_FLEX"], control_activity = outV[1])
##    controlActivity(control_id = muscle_ids["wrist.R_EXT"] , control_activity = 1.0-outV[1])
##    controlActivity(control_id = muscle_ids["forearm.L_FLEX"], control_activity = outV[2])
##    controlActivity(control_id = muscle_ids["forearm.L_EXT"] , control_activity = 1.0-outV[2])
##    controlActivity(control_id = muscle_ids["forearm.R_FLEX"], control_activity = outV[3])
##    controlActivity(control_id = muscle_ids["forearm.R_EXT"] , control_activity = 1.0-outV[3])
##    controlActivity(control_id = muscle_ids["upper_arm.L_FLEX"], control_activity = outV[4])
##    controlActivity(control_id = muscle_ids["upper_arm.L_EXT"] , control_activity = 1.0-outV[4])
##    controlActivity(control_id = muscle_ids["upper_arm.R_FLEX"], control_activity = outV[5])
##    controlActivity(control_id = muscle_ids["upper_arm.R_EXT"] , control_activity = 1.0-outV[5])
##    controlActivity(control_id = muscle_ids["shin_lower.L_FLEX"], control_activity = outV[6])
##    controlActivity(control_id = muscle_ids["shin_lower.L_EXT"] , control_activity = 1.0-outV[6])
##    controlActivity(control_id = muscle_ids["shin_lower.R_FLEX"], control_activity = outV[7])
##    controlActivity(control_id = muscle_ids["shin_lower.R_EXT"] , control_activity = 1.0-outV[7])
##    controlActivity(control_id = muscle_ids["shin.L_FLEX"], control_activity = outV[8])
##    controlActivity(control_id = muscle_ids["shin.L_EXT"] , control_activity = 1.0-outV[8])
##    controlActivity(control_id = muscle_ids["shin.R_FLEX"], control_activity = outV[9])
##    controlActivity(control_id = muscle_ids["shin.R_EXT"] , control_activity = 1.0-outV[9])
##    controlActivity(control_id = muscle_ids["thigh.L_FLEX"], control_activity = outV[10])
##    controlActivity(control_id = muscle_ids["thigh.L_EXT"] , control_activity = 1.0-outV[10])
##    controlActivity(control_id = muscle_ids["thigh.R_FLEX"], control_activity = outV[11])
##    controlActivity(control_id = muscle_ids["thigh.R_EXT"] , control_activity = 1.0-outV[11])





##    controlActivity(control_id = muscle_ids["wrist.L_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["wrist.L_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["wrist.R_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["wrist.R_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["forearm.L_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["forearm.L_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["forearm.R_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["forearm.R_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["upper_arm.L_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["upper_arm.L_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["upper_arm.R_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["upper_arm.R_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["shin_lower.L_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["shin_lower.L_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["shin_lower.R_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["shin_lower.R_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["shin.L_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["shin.L_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["shin.R_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["shin.R_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["thigh.L_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["thigh.L_EXT"] , control_activity = anti_act_tmp)
##    controlActivity(control_id = muscle_ids["thigh.R_FLEX"], control_activity = act_tmp)
##    controlActivity(control_id = muscle_ids["thigh.R_EXT"] , control_activity = anti_act_tmp)

##    speed_ = 6.0
##    act_tmp         = 0.5 + 0.5*np.sin(speed_*t_bl)
##    anti_act_tmp    = 1.0 - act_tmp
##    act_tmp_p1      = 0.5 + 0.5*np.sin(speed_*t_bl - np.pi*0.5)
##    anti_act_tmp_p1 = 1.0 - act_tmp_p1
##    act_tmp_p2      = 0.5 + 0.5*np.sin(speed_*t_bl + np.pi*0.5)
##    anti_act_tmp_p2 = 1.0 - act_tmp_p2
##
##
##    controlActivity(control_id = servo_ids["wrist.L"], control_activity = 0.4)
##    controlActivity(control_id = servo_ids["wrist.R"], control_activity = 0.4)
##    controlActivity(control_id = servo_ids["forearm.L"], control_activity = 0.8*act_tmp)
##    controlActivity(control_id = servo_ids["forearm.R"], control_activity = 0.8*anti_act_tmp)
##    controlActivity(control_id = servo_ids["upper_arm.L"], control_activity = 1.0*act_tmp_p1)
##    controlActivity(control_id = servo_ids["upper_arm.R"], control_activity = 1.0*anti_act_tmp_p1)
##    controlActivity(control_id = servo_ids["shin_lower.L"], control_activity = 0.8*anti_act_tmp)
##    controlActivity(control_id = servo_ids["shin_lower.R"], control_activity = 0.8*act_tmp)
##    controlActivity(control_id = servo_ids["shin.L"], control_activity = 0.5*anti_act_tmp_p1)
##    controlActivity(control_id = servo_ids["shin.R"], control_activity = 0.5*act_tmp_p1)
##    controlActivity(control_id = servo_ids["thigh.L"], control_activity = 0.5*anti_act_tmp)
##    controlActivity(control_id = servo_ids["thigh.R"], control_activity = 0.5*act_tmp)

    
##    controlActivity(control_id = servo_ids["wrist.L"], control_activity = outV[0])
##    controlActivity(control_id = servo_ids["wrist.R"], control_activity = outV[1])
##    controlActivity(control_id = servo_ids["forearm.L"], control_activity = outV[2])
##    controlActivity(control_id = servo_ids["forearm.R"], control_activity = outV[3])
##    controlActivity(control_id = servo_ids["upper_arm.L"], control_activity = outV[4])
##    controlActivity(control_id = servo_ids["upper_arm.R"], control_activity = outV[5])
##    controlActivity(control_id = servo_ids["shin_lower.L"], control_activity = outV[6])
##    controlActivity(control_id = servo_ids["shin_lower.R"], control_activity = outV[7])
##    controlActivity(control_id = servo_ids["shin.L"], control_activity = outV[8])
##    controlActivity(control_id = servo_ids["shin.R"], control_activity = outV[9])
##    controlActivity(control_id = servo_ids["thigh.L"], control_activity = outV[10])
##    controlActivity(control_id = servo_ids["thigh.R"], control_activity = outV[11])


##    controlActivity(control_id = servo_ids["wrist.L"], control_activity = outV[0])
##    controlActivity(control_id = servo_ids["wrist.R"], control_activity = outV[1])
##    controlActivity(control_id = servo_ids["forearm.L"], control_activity = 0.8*outV[2])
##    controlActivity(control_id = servo_ids["forearm.R"], control_activity = 0.8*(1-outV[2]))
##    controlActivity(control_id = servo_ids["upper_arm.L"], control_activity = 1.0*outV[3])
##    controlActivity(control_id = servo_ids["upper_arm.R"], control_activity = 1.0*(1-outV[3]))
##    controlActivity(control_id = servo_ids["shin_lower.L"], control_activity = 0.8*(1-outV[4]))
##    controlActivity(control_id = servo_ids["shin_lower.R"], control_activity = 0.8*outV[4])
##    controlActivity(control_id = servo_ids["shin.L"], control_activity = 0.5*(1-outV[5]))
##    controlActivity(control_id = servo_ids["shin.R"], control_activity = 0.5*outV[5])
##    controlActivity(control_id = servo_ids["thigh.L"], control_activity = 0.5*(1-outV[2]))
##    controlActivity(control_id = servo_ids["thigh.R"], control_activity = 0.5*outV[2])

##    controlActivity(control_id = control_ids["Leg1_FLEX"], control_activity = act_tmp1)
##    controlActivity(control_id = control_ids["Leg1_EXT"] , control_activity = anti_act_tmp1)
##    controlActivity(control_id = control_ids["Leg2_FLEX"], control_activity = act_tmp2)
##    controlActivity(control_id = control_ids["Leg2_EXT"] , control_activity = anti_act_tmp2)
##    controlActivity(control_id = control_ids["Leg3_FLEX"], control_activity = act_tmp3)
##    controlActivity(control_id = control_ids["Leg3_EXT"] , control_activity = anti_act_tmp3)
##    controlActivity(control_id = control_ids["Leg4_FLEX"], control_activity = act_tmp4)
##    controlActivity(control_id = control_ids["Leg4_EXT"] , control_activity = anti_act_tmp4)


##    speed_ = 6.0
##    act_tmp         = 0.5 + 0.5*np.sin(speed_*t_bl)
##    anti_act_tmp    = 1.0 - act_tmp
##    act_tmp_p1      = 0.5 + 0.5*np.sin(speed_*t_bl - np.pi*0.5)
##    anti_act_tmp_p1 = 1.0 - act_tmp_p1
##    act_tmp_p2      = 0.5 + 0.5*np.sin(speed_*t_bl + np.pi*0.5)
##    anti_act_tmp_p2 = 1.0 - act_tmp_p2
    
    # Torque-based muscles
    #~ controlActivity(control_id = muscle_ids["forearm.L_FLEX"], control_activity = act_tmp)
    #~ controlActivity(control_id = muscle_ids["forearm.L_EXT"] , control_activity = anti_act_tmp)
    # Servos
    #~ controlActivity(control_id = servo_ids["wrist.L"], control_activity = 0.8*act_tmp_p1)
    #~ controlActivity(control_id = servo_ids["wrist.R"], control_activity = 0.8*anti_act_tmp_p1)

##    controlActivity(control_id = servo_ids["wrist.L"], control_activity = outV[0])
##    controlActivity(control_id = servo_ids["wrist.R"], control_activity = outV[1])
##    controlActivity(control_id = servo_ids["forearm.L"], control_activity = 0.8*outV[2])
##    controlActivity(control_id = servo_ids["forearm.R"], control_activity = 0.8*(1-outV[2]))
##    controlActivity(control_id = servo_ids["upper_arm.L"], control_activity = 1.0*outV[3])
##    controlActivity(control_id = servo_ids["upper_arm.R"], control_activity = 1.0*(1-outV[3]))
##    controlActivity(control_id = servo_ids["shin_lower.L"], control_activity = 0.8*(1-outV[4]))
##    controlActivity(control_id = servo_ids["shin_lower.R"], control_activity = 0.8*outV[4])
##    controlActivity(control_id = servo_ids["shin.L"], control_activity = 0.5*(1-outV[5]))
##    controlActivity(control_id = servo_ids["shin.R"], control_activity = 0.5*outV[5])
##    controlActivity(control_id = servo_ids["thigh.L"], control_activity = 0.5*(1-outV[2]))
##    controlActivity(control_id = servo_ids["thigh.R"], control_activity = 0.5*outV[2])

    
##    controlActivity(control_id = servo_ids["wrist.L"], control_activity = 0.4)
##    controlActivity(control_id = servo_ids["wrist.R"], control_activity = 0.4)
##    controlActivity(control_id = servo_ids["forearm.L"], control_activity = 0.8*act_tmp)
##    controlActivity(control_id = servo_ids["forearm.R"], control_activity = 0.8*anti_act_tmp)
##    controlActivity(control_id = servo_ids["upper_arm.L"], control_activity = 1.0*act_tmp_p1)
##    controlActivity(control_id = servo_ids["upper_arm.R"], control_activity = 1.0*anti_act_tmp_p1)
##    controlActivity(control_id = servo_ids["shin_lower.L"], control_activity = 0.8*anti_act_tmp)
##    controlActivity(control_id = servo_ids["shin_lower.R"], control_activity = 0.8*act_tmp)
##    controlActivity(control_id = servo_ids["shin.L"], control_activity = 0.5*anti_act_tmp_p1)
##    controlActivity(control_id = servo_ids["shin.R"], control_activity = 0.5*act_tmp_p1)
##    controlActivity(control_id = servo_ids["thigh.L"], control_activity = 0.5*anti_act_tmp)
##    controlActivity(control_id = servo_ids["thigh.R"], control_activity = 0.5*act_tmp)








#bge.logic.endGame()
#bge.logic.restartGame()
#scn.reset()




##nest.Simulate(200.)
##T=nest.GetStatus(outSpikes,'events')[0]
##T_times=nest.GetStatus(outSpikes,'events')[0]['times']
##T_senders=nest.GetStatus(outSpikes,'events')[0]['senders']
##time=nest.GetKernelStatus()['time']
##T_times-=time
##T_act=ActFunc(T_times)
##
##outV=[[T_act[j] for j in range(T_senders.size) if OutNeurons[i]==T_senders[j]]for i in range(numOut)]
##
##outV=[sum(outV[i]) for i in range(numOut)]
##outV=outV/numOut
##print(outV)



##TT=nest.GetStatus(outSpikes2,'events')




##numCtxRS=40     # Number of Sensory Cortex Regular Spiking Excitatory Neurons
##numCtxFS=10     # Number of Sensory Cortex Fast Spiking Inhibitory Neurons
##
##
##numVolCell=20   # Number of Volume Cells to transmit Spikes for STDP
##DopDelay=200.   # Dopamine Delay in ms
##MaxWeight=20.   # Max Weight of STDP Modulated Synapses
##MinWeight=8.    # Min Weight of STDP Modulated Synapses
##MinPoissonRate=5.
##MaxPoissonRate=15.
##
##WeiR_R=15.
##WeiR_F=10.
##WeiF_R=-10.
##WeiF_F=-10.
##
##CnR_R=int(numCtxRS*0.1) # Number of Connections from Cortex Regular Spiking to Cortex RS neurons
##CnR_F=int(numCtxRS*0.1) # Number of Connections from Cortex Regular Spiking to Cortex Fast Spiking neurons
##CnF_F=int(numCtxFS*0.1) # Number of Connections from Cortex Fast Spiking to Cortex FS neurons
##CnF_R=int(numCtxFS*0.1) # Number of Connections from Cortex FS to Cortex RS neurons
##
##Inp_Conn_Type='stdp_dopamine_synapse'
##
##
##Ctx1RS=nest.Create('izhikevich',numCtxRS)
##nest.SetStatus(Ctx1RS,{"c":-65.0,"d":8.0})
##Ctx1FS=nest.Create('izhikevich',numCtxFS)
##nest.SetStatus(Ctx1FS[:],"a",0.1)
##
##poisson1 = nest.Create( 'poisson_generator' , 1 , { 'rate' : MinPoissonRate }) # for Inp1
##poisson2 = nest.Create( 'poisson_generator' , 1 , { 'rate' : MinPoissonRate }) # for Inp2
##poisson3 = nest.Create( 'poisson_generator' , 1 , { 'rate' : MinPoissonRate }) # for Inp3
##
### Dopamine
##vt=nest.Create("volume_transmitter")	# Dopamine releasing neurons
##volume_cells=nest.Create('izhikevich',numVolCell)
##nest.SetDefaults("stdp_dopamine_synapse",{"vt":vt[0],"Wmax":MaxWeight,"Wmin":MinWeight,"b":0.005,"tau_n":800.,"tau_c":500.})
##step_current=nest.Create('spike_generator',1)
##
##Ctx1_spikes=nest.Create("spike_detector", 1, {"to_file": True})
##Ctx2_spikes=nest.Create("spike_detector", 1, {"to_file": True})
##Ctx3_spikes=nest.Create("spike_detector", 1, {"to_file": True})
##
##RandomConnect(Ctx1RS,Ctx1RS, CnR_R, WeiR_R+np.random.rand()/10.-0.05, 1., Inp_Conn_Type)
##RandomConnect(Ctx1RS,Ctx1FS, CnR_F, WeiR_F+np.random.rand()/10.-0.05, 1., Inp_Conn_Type)
##RandomConnect(Ctx1FS,Ctx1RS, CnF_R, WeiF_R+np.random.rand()/10.-0.05, 1., Inp_Conn_Type)
##RandomConnect(Ctx1FS,Ctx1FS, CnF_F, WeiF_F+np.random.rand()/10.-0.05, 1., Inp_Conn_Type)
##
##nest.Connect(poisson1,Ctx1RS,{'rule':'all_to_all'},{'weight':10.,'model':'static_synapse','delay':1.})
##nest.Connect(poisson2,Ctx1RS,{'rule':'all_to_all'},{'weight':10.,'model':'static_synapse','delay':1.})
##nest.Connect(poisson3,Ctx1RS,{'rule':'all_to_all'},{'weight':10.,'model':'static_synapse','delay':1.})
##nest.Connect(poisson1,Ctx1FS,{'rule':'all_to_all'},{'weight':10.,'model':'static_synapse','delay':1.})
##nest.Connect(poisson2,Ctx1FS,{'rule':'all_to_all'},{'weight':10.,'model':'static_synapse','delay':1.})
##nest.Connect(poisson3,Ctx1FS,{'rule':'all_to_all'},{'weight':10.,'model':'static_synapse','delay':1.})
##
##nest.ConvergentConnect(Ctx1RS, Ctx1_spikes, model="static_synapse")


#pos=[(np.mod(neuron,dx),np.mod(int(neuron/dx),dy),np.mod(int(neuron/(dx*dy)),dz)) for neuron in range(numSC)]
#aa=nest.GetConnections(SpinCord,SpinCord)
#print(len(aa))

#RandomConnect(SpinCord,SpinCord,conn,1.,1.,synType)
#nest.GetStatus(nest.GetConnections(SpinCord,SpinCord))


##
##    if Record_or_Test==0: # Record Case
##        ax_avg=(ax_avg*(i_bl)+ax)/(i_bl+1)
##        ay_avg=(ay_avg*(i_bl)+ay)/(i_bl+1)
##        az_avg=(az_avg*(i_bl)+az)/(i_bl+1)
##        #ax_avg=scn.objects["obj_head"].localAngularVelocity[0]
##        #ay_avg=scn.objects["obj_head"].localAngularVelocity[1]
##        #az_avg=scn.objects["obj_head"].localAngularVelocity[2]
##        z4=z3
##        z3=z2
##        z2=np.vstack((x,u))
##
##        out4=out3
##        out3=out2
##        out2=out
##        if ay_avg<-1. and ax_avg<2. and ax_avg>-2.: # Record the outputs if acceleration is positive
##            z=np.vstack((x,u))
##            Record=np.vstack((Record,np.vstack((z,out)).T))
##            #Record=np.vstack((Record,np.vstack((z2,out2)).T))
##            #Record=np.vstack((Record,np.vstack((z3,out3)).T))
##            #Record=np.vstack((Record,np.vstack((z4,out4)).T))
##            #print(Record.shape,np.vstack((z,out)).T.shape)    
##        if np.mod(i_bl,100)==0: # Save once in 1000 steps
##            PickleIt(Record,'paramWalking')
##    else:
##        ax_avg=(ax_avg*(i_bl)+ax)/(i_bl+1)
##        ay_avg=(ay_avg*(i_bl)+ay)/(i_bl+1)
##        az_avg=(az_avg*(i_bl)+az)/(i_bl+1)


##    x=(1-a)*x+a*NeuronFunc(wInp.dot(np.vstack((u,out)))+wRes.dot(x)) # States get output as input
##    #x=(1-(a))*x+a*NeuronFunc(wInp.dot(u)+wRes.dot(x)) # States dont have outputs as inputs
##    
##
##    #input_array=[ax,ay,az,ax2,ay2,az2,ax3,ay3,az3]
##    input_array1=[ax,ay,az]
##    #input_array2=[sF1[0],sF1[1],sF2[0],sF2[1],sF3[0],sF3[1],sF4[0],sF4[1]] # Inputs are only muscle positions
##
##    input_array1=InputFunc(input_array1,len(input_array1),nAct,100.,-100.)
##    #input_array2=InputFunc(input_array2,len(input_array2),nAct,2.,-2.) # Convert muscle positions to inputs of the ESN
##    
##    u=input_array1
##
##    u=u.reshape(nInp,1) # Reshape inputs (n,1)
##    x=x.reshape(nRes,1)
##    z=np.vstack((x,u))
##
##    if Record_or_Test==1: # Test case with using Regressed wOut
##        out=wOut.dot(z) # Calculate outputs with using wOut
##        #out=out/np.amax(out) # This scale needed not to diverge ???
##        out.reshape(nOut,1)
##    elif Record_or_Test==0: # Record case with random outputs
##        out=(np.random.rand(nOut,1)-.5)*2


##FF=1.5
##FF2=.05
##muscle_ids = {}
##[ muscle_ids["wrist.L_FLEX"], muscle_ids["wrist.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_wrist.L",  attached_object_name = "obj_forearm.L",  maxF = FF2)
##[ muscle_ids["wrist.R_FLEX"], muscle_ids["wrist.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_wrist.R",  attached_object_name = "obj_forearm.R",  maxF = FF2)
##[ muscle_ids["forearm.L_FLEX"], muscle_ids["forearm.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_forearm.L",  attached_object_name = "obj_upper_arm.L",  maxF = FF)
##[ muscle_ids["forearm.R_FLEX"], muscle_ids["forearm.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_forearm.R",  attached_object_name = "obj_upper_arm.R",  maxF = FF)
##
##[ muscle_ids["upper_arm.L_FLEX"], muscle_ids["upper_arm.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_upper_arm.L",  attached_object_name = "obj_shoulder.L",  maxF = FF)
##[ muscle_ids["upper_arm.R_FLEX"], muscle_ids["upper_arm.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_upper_arm.R",  attached_object_name = "obj_shoulder.R",  maxF = FF)
##[ muscle_ids["shin_lower.L_FLEX"], muscle_ids["shin_lower.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin_lower.L",  attached_object_name = "obj_shin.L",  maxF = FF2)
##[ muscle_ids["shin_lower.R_FLEX"], muscle_ids["shin_lower.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin_lower.R",  attached_object_name = "obj_shin.R",  maxF = FF2)
##
##[ muscle_ids["shin.L_FLEX"], muscle_ids["shin.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin.L",  attached_object_name = "obj_thigh.L",  maxF = FF)
##[ muscle_ids["shin.R_FLEX"], muscle_ids["shin.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin.R",  attached_object_name = "obj_thigh.R",  maxF = FF)
##[ muscle_ids["thigh.L_FLEX"], muscle_ids["thigh.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_thigh.L",  attached_object_name = "obj_hips",  maxF = FF2)
##[ muscle_ids["thigh.R_FLEX"], muscle_ids["thigh.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_thigh.R",  attached_object_name = "obj_hips",  maxF = FF2)

