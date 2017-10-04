# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 21:52:20 2017

@author: sritee
"""

#Imitation learning for reacher task
# First, get the expert demonstration
# Get the embedding of the demonstration by passing it through an LSTM
#Concatenate the embedding with the current state, and train it to output correct actions
#Train using cross entropy loss

import tensorflow as tf
import numpy as np
from env_obstacle import obstacle_reacher
#reset the default graph
tf.reset_default_graph()
num_epochs=100
batch_size=12
num_hidden=12
num_hidden1=32
num_hidden2=12
env=obstacle_reacher()
max_seqlen=42
state_size=6
num_samp=95
import cv2
    
x=tf.placeholder('float32',[None,None,state_size],name='X')
y=tf.placeholder('float32',[None,1],name='Y')
embedding_vec=tf.placeholder('float32',[1,num_hidden])
position=tf.placeholder('float32',[1,state_size])
#max_seqlen=tf.placeholder('int32',[1])
W1=tf.Variable(tf.random_normal([num_hidden+state_size,num_hidden1]))
b1=tf.Variable(tf.random_normal([num_hidden1]))
W2=tf.Variable(tf.random_normal([num_hidden1,num_hidden2]))
b2=tf.Variable(tf.random_normal([num_hidden2]))
W3=tf.Variable(tf.random_normal([num_hidden2,1]))
b3=tf.Variable(tf.random_normal([1]))

seqlen=tf.placeholder('int32',[None],name='Seq_len')
cell=tf.nn.rnn_cell.BasicLSTMCell(num_hidden)


#init_tuple=tf.nn.rnn_cell.LSTMStateTuple(c_state,h_state)



def get_embedding(cell,x,seqlen):
    #tf.variable
    #with tf.variable_scope('model',reuse=None):
    #i=tf.zeros([3, cell.state_size[0]*2])
    b_size=tf.shape(x)[0]
    #output,state=tf.nn.dynamic_rnn(cell,x,dtype='float32',sequence_length=seqlen,initial_state=init_tuple)
    output,state=tf.nn.dynamic_rnn(cell,x,dtype='float32',sequence_length=seqlen)

    seq_r=tf.shape(x)[1] #maximum sequence length, checking via x size
    index = tf.range(0,b_size) * seq_r + (seqlen - 1)
    #Indexing
    last= tf.gather(tf.reshape(output, [-1, num_hidden]), index)
    return last,state

#last=val[5,:,:]
# Indexing

def predict_action(x,embedding,seqlen):
    op_r=tf.tile(tf.reshape(embedding,(-1,num_hidden,1)),[1,1,tf.shape(x)[1]])
    op_r=tf.transpose(op_r,[0,2,1])
    context=tf.concat((x,op_r),axis=2)
    a=tf.sequence_mask(seqlen,tf.shape(x)[1])
    context=tf.boolean_mask(context,a)
    context=tf.reshape(context,(-1,num_hidden+state_size))
    act1=tf.nn.relu(tf.add(tf.matmul(context,W1),b1))
    act2=tf.nn.relu(tf.add(tf.matmul(act1,W2),b2))
    action=tf.nn.tanh(tf.add(tf.matmul(act2,W3),b3))

    return action
    
def test_action(state,t_embed):
    context=tf.concat((state,t_embed),axis=1)
    act1=tf.nn.relu(tf.add(tf.matmul(context,W1),b1))
    act2=tf.nn.relu(tf.add(tf.matmul(act1,W2),b2))
    test_action=tf.nn.tanh(tf.add(tf.matmul(act2,W3),b3))
    return test_action

op,hidden_state=get_embedding(cell,x,seqlen)
action=predict_action(x,op,seqlen)
loss=tf.losses.mean_squared_error(y,action)
optimizer=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

test_act=test_action(position,embedding_vec)

#train=env.get_training_data(num_samples=num_samp)
train=list(np.load('data_1.npy'))
#q,x_data,y_data,seq_len=prepare_data(train,0)
def prepare_data(train,num):
    x_data=[]
    y_data=[]
    seq_len=[]
    for p in train:
        eg_x=[]
        eg_y=[]
        seq_len.append(len(p[0]))
        for idx in range(50): #maxlen
            if idx<len(p[0]):
               s=(p[0][idx]-64)/64
               #print(state)
               eg_x.append(s)
               eg_y.append(p[1][idx])
            else:
                if num==0:
                    eg_x.append(np.zeros(state_size))
        x_data.append(eg_x)
        y_data.append(eg_y)
    
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    seq_len=np.array(seq_len)
    #y_data=(np.concatenate(y_data.ravel())).reshape(-1,1)
    
    return x_data,y_data,seq_len
t=train[0:-2]
tests=train[-2:]
x_data,y_data,seq_len=prepare_data(t,0)
xt_data,yt_data,seqt_len=prepare_data(tests,0)

for m in range(100):
    for k in range(int(num_samp/batch_size)):
        #print(k)
        ind=np.random.randint(0,x_data.shape[0],batch_size)
        x_d,y_d,seq_l=x_data[ind],y_data[ind],seq_len[ind]
        y_d=np.concatenate(y_d.ravel()).reshape(-1,1)
        #sess.run(optimizer,feed_dict={x:x_d,y:y_d,seqlen:seq_l,c_state:np.zeros([batch_size,num_hidden]),h_state:np.zeros([batch_size,num_hidden])})
        sess.run(optimizer,feed_dict={x:x_d,y:y_d,seqlen:seq_l})
    print('epoch number' +str(m))
    y_t=np.concatenate(yt_data.ravel()).reshape(-1,1)
    print(sess.run(loss,feed_dict={x:xt_data,y:y_t,seqlen:seqt_len}))
#   
print('running test')
max_timesteps=30
test=env.get_training_data(num_samples=1)
x_data,y_data,seq_len=prepare_data(test,1)
test_embed=sess.run(op,feed_dict={x:x_data,y:y_data.reshape(-1,1),seqlen:seq_len})
init_centre=env.particle_centre.copy()
print(env.particle_centre)

for m in range(max_timesteps):  
    state=(np.array([env.particle_centre[0],128-env.particle_centre[1],init_centre[0],init_centre[1],env.target_centre[0],env.target_centre[1]])-64)/64
    act_test=float(sess.run(test_act,feed_dict={position:state.reshape(1,-1),embedding_vec:test_embed}))
    env.step(act_test)
    print(act_test)
    
    #print(act_test)
    #print(act_test*180+180)
    cv2.imwrite('./traj/test_trajectory'+str(m)+'.jpg',env.test_frame)
    print(env.particle_centre)
#    
#    
#
