import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
from model import Seq2Seq
from utils import *
import logging
import os
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

batch_size = 32
x_seq_len = 300
y_seq_len = 100
emb_size = 100
hid_size = 128
train_data_path = "../data/train"
valid_data_path = "../data/valid"
test_data_path = "../data/test"
model_dir = "./models"
def map2word(rev_vocab, idxs):
    temp = []
    for idx in idxs:
        word = rev_vocab[idx]
        temp.append(word)#.decode('utf-8'))
    return temp
def evaluate(sess,model,iter_,index,vocab):
    bleus = []
    for x,y in iter_:
        outputs = model.eval(sess,x,np.ones((x.shape[0]))*index)
        
        for i in range(x.shape[0]):
            
            predict = map2word(vocab,outputs[i])
            bleus.append(sentence_bleu([y[i]],predict,weights=[1,0,0,0]))
    return np.mean(bleus)
logging.info("1. build vocab")
vocab,vocab_dict = build_vocab(train_data_path)
vocab_size = len(vocab)
logging.info("2. deal test data")
test_iter = Itertool(test_data_path,batch_size=batch_size,
                    max_len=x_seq_len,max_y_len = y_seq_len, vocab = vocab_dict,evaluate=True)

logging.info("3. build model")
model = Seq2Seq(emb_size,hid_size,vocab_size,y_seq_len)
model.build_model()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


logging.info("4. begin evaluate")
with tf.Session(config=config) as sess:

    #checkpoint = os.path.join(model_dir, "model.ckpt")
    checkpoint = "model.ckpt"
    model.saver.restore(sess,checkpoint)

    bleu = evaluate(sess,model,test_iter,1,vocab)
    logging.info("bleu is {}".format(bleu))
                


