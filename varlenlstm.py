#Imitation learning for reacher task
# First, get the expert demonstration
# Get the embedding of the demonstration by passing it through an LSTM
#Concatenate the embedding with the current state, and train it to output correct actions
#Train using cross entropy loss

import tensorflow as tf
import numpy as np
from env_reacher import particle_reacher
import math
#reset the default graph
tf.reset_default_graph()
num_epochs=100
batch_size=32
num_hidden=32
num_hidden1=24
env=particle_reacher()
max_seqlen=30
state_size=4
num_samp=2000
    
x=tf.placeholder('float32',[None,max_seqlen,4],name='X')
y=tf.placeholder('float32',[None,1],name='Y')

W1=tf.Variable(tf.random_normal([num_hidden+state_size,num_hidden1]))
b1=tf.Variable(tf.random_normal([num_hidden1]))
W2=tf.Variable(tf.random_normal([num_hidden1,1]))
b2=tf.Variable(tf.random_normal([1]))

seqlen=tf.placeholder('int32',[None],name='Seq_len')
size=tf.placeholder('int32',[1])
cell=tf.nn.rnn_cell.BasicLSTMCell(num_hidden)

output,state=tf.nn.dynamic_rnn(cell,x,dtype='float32',sequence_length=seqlen)

def get_embedding(cell,x,seqlen):
    #tf.variable
    #with tf.variable_scope('model',reuse=None):
    output,state=tf.nn.dynamic_rnn(cell,x,dtype='float32',sequence_length=seqlen)
    b_size=tf.shape(output)[0]
    index = tf.range(0,b_size) * max_seqlen + (seqlen - 1)
    #Indexing
    last= tf.gather(tf.reshape(output, [-1, num_hidden]), index)
    return last,b_size

#last=val[5,:,:]
# Indexing
op,b_size=get_embedding(cell,x,seqlen)

def predict_action(x,embedding,seqlen):
    op_r=tf.tile(tf.reshape(op,(-1,num_hidden,1)),[1,1,max_seqlen])
    op_r=tf.transpose(op_r,[0,2,1])
    context=tf.concat((x,op_r),axis=2)
    a=tf.sequence_mask(seqlen,max_seqlen)
    context=tf.boolean_mask(context,a)
    context=tf.reshape(context,(-1,num_hidden+state_size))
    act1=tf.nn.relu(tf.add(tf.matmul(context,W1),b1))
    action=tf.nn.tanh(tf.add(tf.matmul(act1,W2),b2))

    return action

action=predict_action(x,op,seqlen)
loss=tf.losses.mean_squared_error(y,action)
optimizer=tf.train.AdamOptimizer().minimize(loss)

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


def prepare_data(t,test):

    x_data=[]
    y_data=[]
    seq_len=[]
    for p in t:
        eg_x=[]
        eg_y=[]
        seq_len.append(len(p))
        image=p[0][0]
        for idx in range(max_seqlen):
            if idx<len(p):
               xx=(p[idx][1]-64)/64
               yy=(p[idx][2]-64)/64
               eg_x.append(np.hstack([xx,yy]))
               eg_y.append(p[idx][3])
              
            else:
                eg_x.append(np.zeros(state_size))
        
        x_data.append(eg_x)
        y_data.append(eg_y)

    x_data=np.array(x_data)
    y_data=np.array(y_data)
    seq_len=np.array(seq_len)
    #y_data=np.concatenate(y_data.ravel()).reshape(-1,1)
    return x_data,y_data,seq_len,image
      #return [seq_len,x_data,y_data]
    

train=env.get_training_data(num_samples=num_samp)
x_data,y_data,seq_len,_=prepare_data(train,0)


for m in range(10):
    for k in range(int(num_samp/batch_size)):
        #print(k)
        ind=np.random.randint(0,x_data.shape[0],batch_size)
        x_d,y_d,seq_l=x_data[ind],y_data[ind],seq_len[ind]
        y_r=np.concatenate(y_d.ravel()).reshape(-1,1)
        sess.run(optimizer,feed_dict={x:x_d,y:y_r,seqlen:seq_l})
    print('epoch number' +str(m))
    print(sess.run(loss,feed_dict={x:x_d,y:y_r,seqlen:seq_l}))
        
    #print(x_test[0])
        
#After training, let us test on new samples
for i in range(10):
    test=env.get_training_data(num_samples=1)
    x_test,y_test,seq_test,image=prepare_data(test,1)
    y_test=y_test.ravel().reshape(-1,1)
    vec=np.array(x_test[0][0][2:]-x_test[0][0][0:2])
    actions=(sess.run(action,feed_dict={x:x_test,y:y_test,seqlen:seq_test})*180+180)[0]
    angle=(math.degrees(math.atan2(vec[1],vec[0])))
    if angle<0:
        angle+=360
    print(angle,actions)
    


    

        

