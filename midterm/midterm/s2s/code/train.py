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
x_seq_len = 500
y_seq_len = 200
emb_size = 100
hid_size = 128
train_data_path = "../data/train"
valid_data_path = "../data/valid"
test_data_path = "../data/test"
model_dir = "./models"
epoch = 30
def map2word(rev_vocab, idxs):
    temp = []
    for idx in idxs:
        word = rev_vocab[idx]
        if word == "_PAD" or word == "_EOS":
            break
        temp.append(word)#.decode('utf-8'))
    return temp
def evaluate(sess,model,iter_,index,vocab):
    bleus = []
    for x,y in iter_:
        outputs = model.eval(sess,x,np.ones((x.shape[0]))*index)
        for i in range(x.shape[0]):

            predict = map2word(vocab,outputs[i])
            if i == 0:
                print (predict)
                print ("\n")
            bleus.append(sentence_bleu([y[i]],predict,weights=[1,0,0,0]))
    return np.mean(bleus)
logging.info("1. build vocab")
vocab,vocab_dict = build_vocab(train_data_path)
vocab_size = len(vocab)
logging.info("2. deal train data")
train_iter = Itertool(train_data_path,batch_size=batch_size,
                    max_len=x_seq_len,max_y_len = y_seq_len, vocab = vocab_dict,evaluate=False)
logging.info("3. deal valid data")
test_iter = Itertool(valid_data_path,batch_size=batch_size,
                    max_len=x_seq_len,max_y_len = y_seq_len, vocab = vocab_dict,evaluate=True)

logging.info("4. build model")
model = Seq2Seq(emb_size,hid_size,vocab_size,y_seq_len)
model.build_model()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


logging.info("5. begin train")
with tf.Session(config=config) as sess:
    all_loss = 0.
    step = 0.
    best_bleu = 0. 
    init_op = tf.group(tf.global_variables_initializer())
    sess.run(init_op)

    for i in range(epoch):
        
        for x, y in train_iter:
            loss = model.train(sess, x,y)
            all_loss += loss
            step += 1
            
            if step % 50 == 0:
                logging.info("The loss of step {} is {}".format(step,all_loss/1000))
                all_loss = 0.
                bleu = evaluate(sess,model,test_iter,1,vocab)
                logging.info("bleu is {}".format(bleu))
                if bleu > best_bleu:
                    logging.info("Best bleu && save model")
                    #model.saver.save(sess, os.path.join(model_dir, "model.ckpt"))
                    model.saver.save(sess, "model.ckpt")
                    best_bleu = bleu
                


