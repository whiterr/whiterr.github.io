迁移学习用来解决当某一任务A数据量不足时，通过另一相似任务B提供经验（也就是从任务B迁移到任务A）的问题。

此处的情形是，在MNIST数据集中，通过对前5个数字(0-4)的学习迁移到后5个数字(5-9)的任务，在一些paper中似乎也有teacher和student任务的叫法。


```python
'''Transfer learning toy example.
迁移学习实例
1 - Train a simple convnet on the MNIST dataset the first 5 digits [0..4].
1 - 基于MINIST数据集，训练简单卷积网络，前5个数字[0..4].
2 - Freeze convolutional layers and fine-tune dense layers
   for the classification of digits [5..9].
2 - 为[5..9]数字分类，冻结卷积层并微调全连接层
Get to 99.8% test accuracy after 5 epochs
for the first five digits classifier
and 99.2% for the last five digits after transfer + fine-tuning.
5个周期后，前5个数字分类测试准确率99.8% ，同时通过迁移+微调，后5个数字测试准确率99.2%
'''

from __future__ import print_function
 
import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

```


```python
# 获取当前时间
now = datetime.datetime.now
 
batch_size = 128
num_classes = 5 #分类类别个数
epochs = 5 #迭代次数
 
img_rows, img_cols = 28, 28  
filters = 32 #卷积器数量
pool_size = 2
kernel_size = 3
 
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)
```


```python
def train_model(model, train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
 
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
 
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
 
    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
```


```python
# the data, shuffled and split between train and test sets
# 筛选（数据顺序打乱）、划分训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# create two datasets one with digits below 5 and one with 5 and above
# 创建2个数据集，一个数字小于5，另一个数学大于等与5
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]
 
x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5

```


```python
# define two groups of layers: feature (convolutions) and classification (dense)
# 定义2组层：特征（卷积）和分类（全连接）
# 特征 = Conv + relu + Conv + relu + pooling + dropout
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

```


```python
# 分类 = 128全连接 + relu + dropout + 5全连接 + softmax
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]
 
# create complete model
# 创建完整模型
model = Sequential(feature_layers + classification_layers)
 
# train model for 5-digit classification [0..4]
# 为5数字分类[0..4]训练模型
train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)
```

    x_train shape: (30596, 28, 28, 1)
    30596 train samples
    5139 test samples
    Train on 30596 samples, validate on 5139 samples
    Epoch 1/5
    30596/30596 [==============================] - 4s 123us/step - loss: 0.1627 - acc: 0.9478 - val_loss: 0.0329 - val_acc: 0.9903
    Epoch 2/5
    30596/30596 [==============================] - 2s 68us/step - loss: 0.0503 - acc: 0.9854 - val_loss: 0.0125 - val_acc: 0.9959
    Epoch 3/5
    30596/30596 [==============================] - 2s 68us/step - loss: 0.0335 - acc: 0.9896 - val_loss: 0.0119 - val_acc: 0.9947
    Epoch 4/5
    30596/30596 [==============================] - 2s 63us/step - loss: 0.0267 - acc: 0.9924 - val_loss: 0.0086 - val_acc: 0.9967
    Epoch 5/5
    30596/30596 [==============================] - 2s 66us/step - loss: 0.0200 - acc: 0.9939 - val_loss: 0.0088 - val_acc: 0.9969
    Training time: 0:00:12.201241
    Test score: 0.008841257099600918
    Test accuracy: 0.9968865538042421



```python
# 冻结上层并重建模型
for l in feature_layers:
    l.trainable = False
 
# 迁移：训练下层为[5..9]分类任务
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)

```

    x_train shape: (29404, 28, 28, 1)
    29404 train samples
    4861 test samples
    Train on 29404 samples, validate on 4861 samples
    Epoch 1/5
    29404/29404 [==============================] - 1s 43us/step - loss: 0.2380 - acc: 0.9315 - val_loss: 0.0535 - val_acc: 0.9821
    Epoch 2/5
    29404/29404 [==============================] - 2s 62us/step - loss: 0.0830 - acc: 0.9736 - val_loss: 0.0361 - val_acc: 0.9881
    Epoch 3/5
    29404/29404 [==============================] - 2s 76us/step - loss: 0.0611 - acc: 0.9815 - val_loss: 0.0307 - val_acc: 0.9889
    Epoch 4/5
    29404/29404 [==============================] - 2s 76us/step - loss: 0.0539 - acc: 0.9835 - val_loss: 0.0285 - val_acc: 0.9891
    Epoch 5/5
    29404/29404 [==============================] - 2s 68us/step - loss: 0.0450 - acc: 0.9864 - val_loss: 0.0259 - val_acc: 0.9918
    Training time: 0:00:09.743428
    Test score: 0.025949684178022327
    Test accuracy: 0.9917712404854968



```python
# model.fit()
# #fit参数详情
# keras.models.fit(
# self,
# x=None, #训练数据
# y=None, #训练数据label标签
# batch_size=None, #每经过多少个sample更新一次权重，defult 32
# epochs=1, #训练的轮数epochs
# verbose=1, #0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# callbacks=None,#list，list中的元素为keras.callbacks.Callback对象，在训练过程中会调用list中的回调函数
# validation_split=0., #浮点数0-1，将训练集中的一部分比例作为验证集，然后下面的验证集validation_data将不会起到作用
# validation_data=None, #验证集
# shuffle=True, #布尔值和字符串，如果为布尔值，表示是否在每一次epoch训练前随机打乱输入样本的顺序，如果为"batch"，为处理HDF5数据
# class_weight=None, #dict,分类问题的时候，有的类别可能需要额外关注，分错的时候给的惩罚会比较大，所以权重会调高，体现在损失函数上面
# sample_weight=None, #array,和输入样本对等长度,对输入的每个特征+个权值，如果是时序的数据，则采用(samples，sequence_length)的矩阵
# initial_epoch=0, #如果之前做了训练，则可以从指定的epoch开始训练
# steps_per_epoch=None, #将一个epoch分为多少个steps，也就是划分一个batch_size多大，比如steps_per_epoch=10，则就是将训练集分为10份，不能和batch_size共同使用
# validation_steps=None, #当steps_per_epoch被启用的时候才有用，验证集的batch_size
# **kwargs #用于和后端交互
# )
# 
# 返回的是一个History对象，可以通过History.history来查看训练过程，loss值等等
```
