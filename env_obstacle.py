# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:45:05 2017

@author: sritee
"""

#Simulate the particle reacher task
import numpy as np
import cv2
#import matplotlib.pyplot as plt
GREEN=(0,255,0) #openCV BGR Format
RED=(0,0,255)
BLACK=(255,255,255)
import matplotlib.pyplot as plt
import time
from scipy.ndimage.filters import gaussian_filter1d
import pickle
class obstacle_reacher(object):
    
    def __init__(self,num_distractors=2,window_size=128):
        #self.num_distractors=num_distractors
        self.window_height=window_size
        self.window_width=window_size
        self.radius=int(self.window_height/12)
        self.target_centre=(0,0)
        self.particle_centre=(0,0)
        self.drawing=False
        #First let us create the background array, then this will be updated with our particle
        
    def create_instance(self,k=2):
        
        #generate the targets, and centre, making sure there is a minimum distance
        self.start_frame=np.ones([self.window_height,self.window_height,3],np.uint8)
        self.target_centre=np.array([self.window_width/2,int(1.5*self.radius)]).astype('int32')
        self.particle_centre=(np.array([self.window_height/2,self.window_width-10])+np.random.randint(0,3,[2])).astype('int32')
        self.target_centre=np.array(self.target_centre).astype('int64')
        self.wall_centre=np.array([self.window_width/2,np.random.randint(40,90)]).astype('int32')
        cv2.circle(self.start_frame,tuple(self.target_centre),self.radius,GREEN,-1)
        cv2.rectangle(self.start_frame,(self.wall_centre[0]-10,self.wall_centre[1]-1),(self.wall_centre[0]+10,self.wall_centre[1]+1),BLACK,-1)
        self.test_frame=self.start_frame.copy()
        self.moving_frame=self.start_frame.copy()
        cv2.circle(self.moving_frame,(self.particle_centre[0],self.particle_centre[1]),self.radius,RED,-1)
        
        #self.step(np.array([0,0]))#check move, dummy

    def step(self,action):
        self.test_frame=self.start_frame.copy()
        angle_theta=(action*180+180)*np.pi/180
        move=np.array([5*np.cos(angle_theta),-5*np.sin(angle_theta)]).astype('int32')
        #print(move)
        self.particle_centre+=move
        cv2.circle(self.test_frame,(self.particle_centre[0],self.particle_centre[1]),self.radius,RED,-1)
        
    def get_training_data(self,num_samples=1):

        self.num_samples=num_samples
        self.batches_trajectory=[] # A collection of trajectories
        for m in range(self.num_samples):
            self.create_instance()
            self.trajectory=[]
            self.num=0
            #trajectory_batches.append(self.get_expert_trajectory())
            cv2.namedWindow('expert') #origin selector       
            cv2.setMouseCallback('expert',self.get_expert_trajectory)

            while True:
                cv2.imshow('expert',self.moving_frame) 
                k = cv2.waitKey(1) & 0xFF 
                if k == 27:
                    print('over to next')
                    break
                time.sleep(0.05)
                cv2.imwrite('training_data'+str(3)+'.jpg',self.moving_frame)
            #print(a.trajectory)
            r=np.array(self.trajectory)[:,0:2]
            r=(r-np.roll(r,1,axis=0))[1:,:]
            ang=np.degrees(np.arctan2(r[:,1],r[:,0])) #get the angle
            ang[ang<0]+=360
            ang=gaussian_filter1d(ang,3)
            self.batches_trajectory.append([np.array(self.trajectory)[1:],(ang-180)/180])
            #print(r.shape)
            print(m)
           #plt.imshow(self.moving_frame)
        return self.batches_trajectory
        
    def get_expert_trajectory(self,event,former_x,former_y,flags,param):
        global current_former_x,current_former_y,drawing,mode,num
        
        if event==cv2.EVENT_LBUTTONDOWN and self.num==0:
            self.drawing=True
            current_former_x,current_former_y=former_x,former_y
            self.num=1

        elif event==cv2.EVENT_MOUSEMOVE:
            if self.drawing==True:
             
                cv2.line(self.moving_frame,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),2)
                self.trajectory.append([current_former_x,self.window_height-current_former_y,self.particle_centre[0],self.particle_centre[1],self.target_centre[0],self.target_centre[1]])
                current_former_x = former_x
                current_former_y = former_y
                    
                    #print former_x,former_y
        elif event==cv2.EVENT_LBUTTONDOWN and self.num==1:
            #print('hello')
            self.drawing=False
#a=obstacle_reacher()
#a.create_instance()
#a.get_training_data()

            
        
