from utils import *
from baseline import baseline

rule = {}
rule["in","to"] = {("following","at"):"in"} #3751
rule["in","to"][("following","nn")] = "in"  #1024
rule["in","to"][("following","np")] = "in"  #787
rule["in","to"][("following","pp$")] = "in" #698
rule["in","to"][("following","ppo")] = "in"#644

rule["nn","vb"] = {}
rule["nn","vb"] = {("previous","to"):"vb"} #5493
rule["nn","vb"][("previous","md")] = "vb"  #2572 
rule["nn","vb"][("following","at")] = "vb" #2189


if __name__ == '__main__':

    # construct the model
    model = baseline()

    # load train data
    train_X, train_Y = load_data("Assignment Files/brown.train.tagged.txt")

    # train the model
    model.train(train_X, train_Y)

#    tags = []
#    for x in train_X:
#        tags += [model.predict(x)]
#    confusion_matrix(model.tag_count,tags,train_Y)
#    exit()
  
        

    # load test data
    test_X,test_Y = load_data("Assignment Files/brown.test.tagged.txt")

    # predict the label for test data
    fw = open("result/brown.test.rule.txt","w")
    for x in test_X:
        tags = model.predict_with_rule(x,rule)
        line = ""
        for word,tag in zip(x,tags):
            line += word + "/" + tag + " "
        fw.write(line.strip()+"\n")   

    fw.close() 

    _,predict_Y = load_data("result/brown.test.rule.txt")
    print ("The accuracy of the baseline model with rule is {}".format(evaluate(predict_Y,test_Y)))