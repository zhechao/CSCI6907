
from utils import *
from HMM import HMM

hmm = HMM()
train_X, train_Y = load_data("Assignment Files/brown.train.tagged.txt")

hmm.train(train_X, train_Y)

test_X,test_Y = load_data("Assignment Files/brown.test.tagged.txt")

fw = open("result/brown.test.hmm.txt","w")
for x in test_X:
    
    tags = hmm.viterbi_decoder(x)
    line = ""
    for word,tag in zip(x,tags):
        line += word + "/" + tag + " "
    fw.write(line.strip()+"\n")   

fw.close() 

_,predict_Y = load_data("result/brown.test.hmm.txt")
print ("The accuracy of the HMM model is {}".format(evaluate(predict_Y,test_Y)))