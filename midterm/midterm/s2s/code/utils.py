
from collections import Counter
import numpy as np 
import nltk
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [ _PAD,_GO,_UNK,_EOS]
#[_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 3 
UNK_ID = 2 

def build_vocab(datapath):
    counter = Counter([])
    summary_path = datapath+".sum"
    
    with open(datapath, 'r',errors='ignore') as f:
        for line in f:
            words = nltk.word_tokenize(line.strip())
            counter.update(Counter(words))
    with open(summary_path, 'r',errors='ignore') as f:
        for line in f:
            words = nltk.word_tokenize(line.strip())
            counter.update(Counter(words))

    words = counter.most_common(50000)
    vocab = _START_VOCAB + [word[0] for word in words]
    vocab_dict = dict(zip(vocab,range(len(vocab))))
    return vocab,vocab_dict
def seq_pad(samples, max_len=20,max_y_len=100,dtype='int32',evaluate=False):
    sample_num = len(samples)
    x = (np.ones((sample_num, max_len)) * 0.).astype(dtype)
    if not evaluate:
        y = (np.ones((sample_num, max_y_len+1)) * 0.).astype(dtype)
    else:
        y = []
    for idx, (n,s) in enumerate(samples):
        len_x = len(n)
        x[idx, :len_x] = n
        if not evaluate:
            len_y = len(s)
            y[idx, :len_y] = s
        else:
            y.append(s)
    return x, y

class Itertool(object):

    def __init__(self, data_path, batch_size=128, max_len=200,max_y_len=100,vocab=None,evaluate=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_y_len = max_y_len
        self.vocab = vocab
        self.summary_path = data_path + ".sum"

        self.evaluate = evaluate
    def __iter__(self):
        with open(self.data_path, 'r',errors='ignore') as f,open(self.summary_path,"r",errors='ignore') as s:
            samples = []
            for line in f:
                summary = s.readline()
                line = nltk.word_tokenize(line)
                summary = nltk.word_tokenize(summary)
    
                line_id = [self.vocab.get(w,UNK_ID) for w in line][:self.max_len]
                summary_id = [self.vocab.get(w,UNK_ID) for w in summary]
                summary_id = [GO_ID] + summary_id[:self.max_y_len-1] +[EOS_ID]    
                
                if self.evaluate:
                    samples.append((line_id,summary))
                else:
                    samples.append((line_id,summary_id))
                
                if len(samples) == self.batch_size:
                    yield seq_pad(samples,self.max_len,self.max_y_len,evaluate=self.evaluate)
                    samples = []
            if len(samples) > 0:
                yield seq_pad(samples,self.max_len,self.max_y_len,evaluate=self.evaluate)
