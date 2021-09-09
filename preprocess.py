
from keras_bert.datasets.pretrained import PretrainedList
from keras_bert.tokenizer import Tokenizer
import numpy as np
import math
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from keras_bert import get_checkpoint_paths,load_vocabulary
from keras_bert.datasets import get_pretrained,PretrainedInfo
import os
import logging as log

log.basicConfig(level=log.INFO)

tag_dict={"比较不满":0,
"非常不满-渠道敏感":1,
"一般不满":2,
"非常不满-服务敏感":3,
"非常不满-费用敏感":4}

def read_corpus(path):
    with open(path,'r',encoding='utf8') as f:
        fd=f.read().split('\n')
        f.close()
    sens=[]
    tags=[]
    fd=[i for i in fd if i]

    for i in fd:
  
        tag,sen=i.split(' ')
        sens.append(sen)
        tags.append(tag)
    return (tags,sens)

def pad_sen(x,num):
    return x+(num-len(x))[0]

def generator(data,tokenizer,tag_dict,batch_size=4):
    while True:

        sens,tags=data
        token_batch,segment_token,Y=[],[],[]
        idxs=list(range(len(sens)))
        np.random.shuffle(idxs)

        token_batch,segment_token=[],[]
        for n,i in enumerate(idxs):
            if len(sens[i])>510:
                continue
                
            sentences=[z for z in sens[i]]
            X1=[tokenizer.encode(z)[0][1] for z in sentences]
            X1=[101]+X1+[102]
            X2=[0]*len(X1)
            token_batch.append(X1)
            segment_token.append(X2)
            Y.append(tag_dict[tags[i]])


            if batch_size==len(token_batch) or n==idxs[-1]:
                l=max([len(z) for z in token_batch])
                token_batch=pad_sequences(token_batch,maxlen=l,padding='post')
                segment_token=pad_sequences(segment_token,maxlen=l,padding='post')

                yield [np.array(token_batch),np.array(segment_token)],\
                      to_categorical(Y,num_classes=5)
                token_batch,segment_token,Y=[],[],[]

if __name__=='__main__':
    model_path=get_pretrained(PretrainedList.chinese_base)

    paths=get_checkpoint_paths(model_path)

    token_dict=load_vocabulary(paths.vocab)

    tokenizer=Tokenizer(token_dict)

    corpus_path=os.path.join(os.path.dirname(__file__),'file','train.txt')
    
    data=read_corpus(corpus_path)

    train_gen=generator(data,tokenizer,tag_dict)

    [a,b],c=next(train_gen)

    log.info(a.shape)
    log.info(b.shape)
    log.info(c.shape)