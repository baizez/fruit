

num_epochs = 200    # ѵ��ѭ������
batch_size = 256    # batch��С
rnn_size = 512      # lstm���а�����unit����
seq_length = 30     # ѵ������
learning_rate = 0.003# ѧϰ��
show_every_n_batches = 30# ÿ���ٲ���ӡһ��ѵ����Ϣ

save_dir = './save'# ����session״̬��λ��

def get_inputs():

    # inputs��targets�����Ͷ���������
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return inputs, targets, learning_rate

def get_init_cell(batch_size, rnn_size):
    num_layers = 2      # lstm���� 
    keep_prob = 0.8     # dropoutʱ�ı�������
   
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)       # ��������rnn_size����Ԫ��lstm cell 
    drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)# ʹ��dropout���Ʒ�ֹoverfitting��   
    cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(num_layers)])# ����2��lstm��   
    init_state = cell.zero_state(batch_size, tf.float32)    # ��ʼ��״̬Ϊ0.0
    init_state = tf.identity(init_state, name='init_state') # ʹ��tf.identify��init_stateȡ�����֣������������ֵ�ʱ��Ҫʹ������������ҵ������state

    return cell, init_state

def build_nn(cell, rnn_size, input_data, vocab_size):

    '''
    cell��������get_init_cell������cell
    '''
    outputs, final_state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    final_state = tf.identity(final_state, name="final_state")  # ͬ����final_stateһ�����֣�����Ҫ���»�ȡ����

    return outputs, final_state


    # remember to initialize weights and biases, or the loss will stuck at a very high point
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                               weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                                               biases_initializer=tf.zeros_initializer())

    return logits, final_state

# ����seq2seq���������������loss
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    # ��������
    vocab_size = len(int_to_vocab)

    # ��ȡģ�͵����룬Ŀ���Լ�ѧϰ�ʽڵ㣬��Щ����tf��placeholder
    input_text, targets, lr = get_inputs()

    # �������ݵ�shape
    input_data_shape = tf.shape(input_text)

    # ����rnn��cell�ͳ�ʼ״̬�ڵ㣬rnn��cell�Ѿ�������lstm��dropout
    # �����rnn_size��ʾÿ��lstm cell�а����˶��ٵ���Ԫ
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)

    # ��������loss��finalstate�Ľڵ�
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # ʹ��softmax��������Ԥ�����
    probs = tf.nn.softmax(logits, name='probs')

    # ����loss
    cost = seq2seq.sequence_loss(logits,targets,tf.ones([input_data_shape[0], input_data_shape[1]]))

    # ʹ��Adam�ᶽ�½�
    optimizer = tf.train.AdamOptimizer(lr)

    # �ü�һ��Gradient���������gradient����[-1, 1]�ķ�Χ��
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

    # ���ѵ���õ�����batch
batches = get_batches(int_text, batch_size, seq_length)

# ��session��ʼѵ���������洴����graph���󴫵ݸ�session
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

            # ��ӡѵ����Ϣ
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # ����ģ��
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

    helper.save_params((seq_length, save_dir))