import scipy.io as sio  
import numpy

class Fruit:

    def __init__(self,filename='data/fruit0606.mat', dataname='D1',task='train'):

        data = sio.loadmat(filename)
        self._epochs_completed = 0
        self._index_in_epoch = 0
           
        if task=="train":
            self.images = data[dataname][:80]
            self.labels = data['TH_all'][:80]
            self._num_examples=80 #是指所有训练数据的样本个数

        if task=="test":
            self.images = data[dataname][80:]
            self.labels = data['TH_all'][80:]
            self._num_examples=24 #是指所有训练数据的样本个数


    def next_batch(self, batch_size, fake_data=False, shuffle=True):
    #.....中间省略过一些fake
        start = self._index_in_epoch  #self._index_in_epoch  所有的调用，总共用了多少个样本，相当于一个全局变量 #start第一个batch为0，剩下的就和self._index_in_epoch一样，如果超过了一个epoch，在下面还会重新赋值。
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
          perm0 = numpy.arange(self._num_examples)  #生成的一个所有样本长度的np.array
          numpy.random.shuffle(perm0)
          self._images = self.images[perm0]
          self._labels = self.labels[perm0]
        # Go to the next epoch
        #从这里到下一个else，所做的是一个epoch快运行完了，但是不够一个batch，将这个epoch的结尾和下一个epoch的开头拼接起来，共同组成一个batch——size的数据。

        if start + batch_size > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest examples in this epoch
          rest_num_examples = self._num_examples - start  # 一个epoch 最后不够一个batch还剩下几个
          images_rest_part = self._images[start:self._num_examples]
          labels_rest_part = self._labels[start:self._num_examples]
          # Shuffle the data
          if shuffle:
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_num_examples
          end = self._index_in_epoch
          images_new_part = self._images[start:end] 
          labels_new_part = self._labels[start:end]
          return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
          #新的epoch，和上一个epoch的结尾凑成一个batch
        else:
          self._index_in_epoch += batch_size  #每调用这个函数一次，_index_in_epoch就加上一个batch——size的，它相当于一个全局变量，上不封顶
          end = self._index_in_epoch
          return self._images[start:end], self._labels[start:end]

if __name__ == '__main__':
    fruit=Fruit()
    x,y=fruit.next_batch(5)
    print(x)
    print(y)
        
