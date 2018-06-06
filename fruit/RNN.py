

num_epochs = 200    # 训练循环次数
batch_size = 256    # batch大小
rnn_size = 512      # lstm层中包含的unit个数
seq_length = 30     # 训练步长
learning_rate = 0.003# 学习率
show_every_n_batches = 30# 每多少步打印一次训练信息

save_dir = './save'# 保存session状态的位置

def get_inputs():

    # inputs和targets的类型都是整数的
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return inputs, targets, learning_rate

def get_init_cell(batch_size, rnn_size):
    num_layers = 2      # lstm层数 
    keep_prob = 0.8     # dropout时的保留概率
   
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)       # 创建包含rnn_size个神经元的lstm cell 
    drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)# 使用dropout机制防止overfitting等   
    cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(num_layers)])# 创建2层lstm层   
    init_state = cell.zero_state(batch_size, tf.float32)    # 初始化状态为0.0
    init_state = tf.identity(init_state, name='init_state') # 使用tf.identify给init_state取个名字，后面生成文字的时候，要使用这个名字来找到缓存的state

    return cell, init_state

def build_nn(cell, rnn_size, input_data, vocab_size):

    '''
    cell就是上面get_init_cell创建的cell
    '''
    outputs, final_state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    final_state = tf.identity(final_state, name="final_state")  # 同样给final_state一个名字，后面要重新获取缓存

    return outputs, final_state


    # remember to initialize weights and biases, or the loss will stuck at a very high point
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                               weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                                               biases_initializer=tf.zeros_initializer())

    return logits, final_state

# 导入seq2seq，下面会用他计算loss
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    # 文字总量
    vocab_size = len(int_to_vocab)

    # 获取模型的输入，目标以及学习率节点，这些都是tf的placeholder
    input_text, targets, lr = get_inputs()

    # 输入数据的shape
    input_data_shape = tf.shape(input_text)

    # 创建rnn的cell和初始状态节点，rnn的cell已经包含了lstm，dropout
    # 这里的rnn_size表示每个lstm cell中包含了多少的神经元
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)

    # 创建计算loss和finalstate的节点
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # 使用softmax计算最后的预测概率
    probs = tf.nn.softmax(logits, name='probs')

    # 计算loss
    cost = seq2seq.sequence_loss(logits,targets,tf.ones([input_data_shape[0], input_data_shape[1]]))

    # 使用Adam提督下降
    optimizer = tf.train.AdamOptimizer(lr)

    # 裁剪一下Gradient输出，最后的gradient都在[-1, 1]的范围内
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

    # 获得训练用的所有batch
batches = get_batches(int_text, batch_size, seq_length)

# 打开session开始训练，将上面创建的graph对象传递给session
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # 打印训练信息
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

    helper.save_params((seq_length, save_dir))