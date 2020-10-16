from utils import *

test_X,test_Y = load_data("Assignment Files/brown.test.tagged.txt")

print ("Question a")
_,predict_Y = load_data("result/brown.test.baseline.txt")
print ("The accuracy of the baseline model is {}".format(evaluate(predict_Y,test_Y)))

print ("Question b")
_,predict_Y = load_data("result/brown.test.rule.txt")
print ("The accuracy of the baseline model with rule is {}".format(evaluate(predict_Y,test_Y)))

print ("Question c")
_,predict_Y = load_data("result/brown.test.hmm.txt")
print ("The accuracy of the HMM model is {}".format(evaluate(predict_Y,test_Y)))

print ("Question d")
_,predict_Y = load_data("result/brown.test.beam_search.txt")
print ("The accuracy of the HMM model with beam search is {}".format(evaluate(predict_Y,test_Y)))