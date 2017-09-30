#Simulate the particle reacher task
import numpy as np
import cv2
#import matplotlib.pyplot as plt
GREEN=(0,255,0) #openCV BGR Format
RED=(0,0,255)
import math
import matplotlib.pyplot as plt

class particle_reacher(object):
    
    def __init__(self,num_distractors=2,window_size=128):
        #self.num_distractors=num_distractors
        self.window_height=window_size
        self.window_width=window_size
        self.radius=int(self.window_height/12)
        self.target_centre=(0,0)
        self.particle_centre=(0,0)
        #First let us create the background array, then this will be updated with our particle
        
    def create_instance(self,k=2):
        
        #generate the targets, and centre, making sure there is a minimum distance
        self.start_frame=np.ones([self.window_height,self.window_height,3],np.uint8)
        while np.linalg.norm(np.subtract(self.target_centre,self.particle_centre),1)<3*self.radius:
            self.target_centre=(np.random.randint(self.window_height-self.radius),np.random.randint(self.window_width-self.radius))
            self.particle_centre=np.array([np.random.randint(self.window_height-self.radius),np.random.randint(self.window_width-self.radius)])
            #print('hi')
        cv2.circle(self.start_frame,tuple(self.target_centre),self.radius,GREEN,-1)
        self.target_centre=np.array(self.target_centre)
        self.step(np.array([0,0]))#check move, dummy

    def step(self,action):
        self.moving_frame=self.start_frame.copy()
        self.particle_centre=np.add(action,self.particle_centre)
        cv2.circle(self.moving_frame,(self.particle_centre[0],self.particle_centre[1]),self.radius,RED,-1)
        
    def get_training_data(self,num_samples=100):
        trajectory_batches=[] # The collecton of trajectories
        self.num_samples=num_samples
        trajectory_batches=[] # A single trajectory
        for m in range(self.num_samples):
            self.create_instance()
            trajectory_batches.append(self.get_expert_trajectory())
           #plt.imshow(self.moving_frame)
        return trajectory_batches
        
    def get_expert_trajectory(self):
        trajectory=[]
        while(np.linalg.norm(np.subtract(self.target_centre,self.particle_centre),1)>2*self.radius):
               #print((np.linalg.norm(np.subtract(self.target_centre,self.particle_centre),1)))
               #print('hi')
               vec=self.target_centre-self.particle_centre
               #print('hello')
               #print(vec)
               #angle from object to target
               angle=math.degrees(math.atan2(vec[1],vec[0])) #measure clockwised from +x
               if angle<0:
                   angle+=360
               self.step((vec*0.1).astype('int32'))
               trajectory.append([self.moving_frame,self.particle_centre,self.target_centre,(angle-180)/180])
               #print('hello')
        return trajectory
        
#a=particle_reacher()
#a.create_instance()
#a.step(np.array([0,0]))
#vec=a.target_centre-a.particle_centre
#angle=math.degrees(math.atan2(vec[1],vec[0]))
#if angle<0:
#    angle+=360
#plt.imshow(a.moving_frame)
#print(angle)
##plt.imshow(a.start_frame)
            
        
