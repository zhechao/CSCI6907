from utils import *
from baseline import baseline

if __name__ == '__main__':

    # construct the model
    model = baseline()

    # load train data
    train_X, train_Y = load_data("Assignment Files/brown.train.tagged.txt")

    # train the model
    model.train(train_X, train_Y)

    # load test data
    test_X,test_Y = load_data("Assignment Files/brown.test.tagged.txt")

    # predict the label for test data
    fw = open("result/brown.test.baseline.txt","w")
    for x in test_X:
        tags = model.predict(x)
        line = ""
        for word,tag in zip(x,tags):
            line += word + "/" + tag + " "
        fw.write(line.strip()+"\n")   

    fw.close() 

    _,predict_Y = load_data("result/brown.test.baseline.txt")
    print ("The accuracy of the baseline model is {}".format(evaluate(predict_Y,test_Y)))