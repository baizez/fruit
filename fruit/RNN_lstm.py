import tensorflow as tf
from fruit import Fruit

lr=0.001        #学习率
training_iters=5000    #迭代次数

batch_size=24

n_inputs=8     #8个传感器
n_steps=60     #120个点
n_hidden_units=128
n_classes=4  #4类

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

    #（5 batch,120step, 8 inputs) => (5*120，8 inputs)
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
    #writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    step=0
    data=Fruit(dataname='C1',task='train')

    while step*batch_size<training_iters:
        batch_xs,batch_ys=data.next_batch(batch_size)
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

    print('train over\n\n')

    dataTest=Fruit(dataname='C1',task='test')
    x_test=dataTest.images[:]
    y_test=dataTest.labels[:]
    print(sess.run(accuracy,feed_dict={
        x:x_test,
        y:y_test,
        }))
    print(sess.run(pred,feed_dict={
            x:x_test,
            y:y_test,
            }))
    print(sess.run(accuracy,feed_dict={
        x:x_test,
        y:y_test,
        }))
    print(sess.run(accuracy,feed_dict={
        x:x_test,
        y:y_test,
        }))