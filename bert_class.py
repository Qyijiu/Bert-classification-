import math

import numpy as np
import os
from keras.layers import *
from keras.models import Model,load_model
from keras.losses import categorical_crossentropy,sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras_bert.datasets import get_pretrained, PretrainedList
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, get_checkpoint_paths, Tokenizer, \
    get_custom_objects

from keras import backend as K
from preprocess import tag_dict,generator,read_corpus
import logging as log

# 加载中文预训练模型(缓存在当前用户.keras/datasets目录中)
model_path = get_pretrained(PretrainedList.chinese_base)
# 模型所在目录的path
paths = get_checkpoint_paths(model_path)
# 加载预训练模型
bert_model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, trainable=False, seq_len=None)



# 训练批次大小
batch_size = 32
# 训练的类别数量
class_num =5
# 模型训练的轮数
epochs = 12


def create_model():

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(5, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(1e-5),
                  metrics=['accuracy'])
    model.summary()
    return model

def train_model(model,train_gen,train_step,valid_gen,valid_step,callbacks=None):
    model.fit(train_gen,
              steps_per_epoch=train_step,
              epochs=epochs,
              validation_data=valid_gen,
              validation_steps=valid_step,
              callbacks=callbacks)
    model.save('bert_class.h5')

def index_tag_dict(tag):
  index_tag={j:i for i,j in tag.items()}
  return index_tag

def spls(x):
    x = x.split('\n')
    x = [i for i in x if i]
    dels = []
    for n, i in enumerate(x):
        x[n] = i.replace('_x000D_', '')
        x[n] = x[n].replace(' ', '')
        if i[0] == '【':
            dels.append(n)

    if len(dels) == 0:
        return ','.join(x)
    left = 0
    for i in dels[::-1]:
        for j, n in enumerate(x[i]):
            if n == '】':
                left = j
                break
        st = x[i][left + 1:].strip('：').strip('）')
        if len(st) > 6:
            x[i] = st
        else:
            x.pop(i)
    x = [i for i in x if i]
    result = ','.join(x)
    result = result.replace(',,', ',')
    result = result.replace('，，', ',')
    result = result.replace('∮', '')

    result = result.replace(' ', '')
    result = result.replace('	', '')

    result = result.replace(',。', '')
    result = result.replace(' 。,', '')
    result = result.replace(',.', '。')
    result = result.replace(' .,', '。')
    result = result.replace('!,', '!')
    result = result.replace('！,', '！')
    result = result.replace(',！', '!')
    result = result.replace(',!', '!')
    result = result.replace(',：', ':')
    result = result.replace('：,', ':')
    result = result.replace('，：', ':')
    result = result.replace('：，', ':')

    return result

def predict(model,x):
  index_tag=index_tag_dict(tag_dict)
  senten=spls(x)
  sentences=[z for z in senten]
  X1=[tokenizer.encode(z)[0][1] for z in sentences]
  X1=[101]+X1+[102]
  X2=[0]*len(X1)
  probe=model.predict([np.array([X1]),np.array([X2])])

  index=np.argmax(probe)

  return index_tag[index]

if __name__=='__main__':

    #模型搭建
    model_bert_path=os.path.join(os.path.dirname(__file__),'bert_class.h5')
    model_file=os.path.join(os.path.dirname(__file__),'model.h5')
    corpus_file=os.path.join(os.path.dirname(__file__),'file','train.txt')
    token_dict = load_vocabulary(paths.vocab)
    tokenizer = Tokenizer(token_dict)

    # ------------------------------------------------------------------------------            1.
    # data = read_corpus(corpus_file)
    #
    # random_order = list(range(len(data[0])))
    # np.random.shuffle(random_order)
    # train_X = [data[1][j] for i, j in enumerate(random_order) if i % 10 != 0]
    # train_Y = [data[0][j] for i, j in enumerate(random_order) if i % 10 != 0]
    # train_da = (train_X, train_Y)
    #
    # valid_X = [data[1][j] for i, j in enumerate(random_order) if i % 10 == 0]
    # valid_Y = [data[0][j] for i, j in enumerate(random_order) if i % 10 == 0]
    # valid_da = (valid_X, valid_Y)
    #
    # train_gen = generator(train_da, tokenizer, tag_dict, batch_size)
    # valid_gen = generator(valid_da, tokenizer, tag_dict, batch_size)
    # train_step = math.ceil(len(train_da[0]) / batch_size)
    # valid_step = math.ceil(len(valid_da[0]) / batch_size)
    # ----------------------------------------------------------------------------                读取数据部分可用于初次、重复训练
    # callback_list = [K.callbacks.ModelCheckpoint(model_file, save_best_only=True)]
    # model=create_model()
    # train_model(model,train_gen,train_step,valid_gen,valid_step,callback_list)
    # --------------------------------------------------------------------------以上为初次训练
    # custom_objects = get_custom_objects()
    # my_objects = {'lambda': Lambda(lambda x: x[:, 0])}
    # custom_objects.update(my_objects)
    # model = load_model(model_bert_path, custom_objects=custom_objects)
    # callback_list = [K.callbacks.ModelCheckpoint(model_file, save_best_only=True)]
    # train_model(model, train_gen, train_step, valid_gen, valid_step, callback_list)
    # ------------------------------------------------------------------------------之间为重复训练
    custom_objects=get_custom_objects()
    my_objects = {'lambda': Lambda(lambda x:x[:,0])}
    custom_objects.update(my_objects)
    model=load_model(model_bert_path,custom_objects=custom_objects)
    flag=1
    while flag!=0:
      sentences=input('请输入评论(0退出):')
      result=predict(model,sentences)
      print(result)
