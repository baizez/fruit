import tensorflow as tf

import scipy.io as sio  
lr=0.001
training_iters=128*10*20

batch_size=105

n_inputs=1     #28行
n_steps=41     #28列
n_hidden_units=128
n_classes=4  #10类

data = sio.loadmat('data/fruit0329.mat')
x_in=data['CSH_all_baseline']
y_in=data['TH_all2']

with tf.name_scope('inputs'):
    x=tf.placeholder(tf.float32,[None,n_steps,n_inputs],name='x_in')
    y=tf.placeholder(tf.float32,[None,n_classes],name='y_in')

weights={
         #28,128
         'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
         'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
         }

biases={
         #128,10
         'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
         'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
         }

def RNN(X,weights,biases):

    #（128 batch,28 step, 28 inputs) => (128*28，28 inputs)
    X=tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,weights['in'])+biases['in']
    #X_in ==>(128batch,28step,128hidden)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])

    #cell
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden_units,forget_bias=0.1,state_is_tuple=True)
    _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)

    #lstm cell is divided into two parts(c_state,m_state)
    outputs,states=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)

    #hidden layer for output as the final results
    results=tf.matmul(states[1],weights['out'])+biases['out']
    return results


pred =RNN(x,weights,biases)
with tf.name_scope('cost'):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

with tf.name_scope('train'):
    train_op=tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred=tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    batch_xs=x_in
    batch_ys=y_in
    batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
    sess.run(init)
    step=0
    while step*batch_size<training_iters:
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys,
            })
        if step % 20==0:
            print(sess.run(accuracy,feed_dict={
            x:batch_xs,
            y:batch_ys,
            }))
        step=step+1
