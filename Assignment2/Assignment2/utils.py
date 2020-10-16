import numpy as np

from collections import Counter
from collections import defaultdict

def load_data(tag_file):
    
    X = []
    Y = []
    for line in open(tag_file):
        line = [token.split("/") for token in line.strip().split()]
        x,y = zip(*line)
        X.append(x)
        Y.append(y)

    return X, Y


def evaluate(predicts, targets):
    right = 0.
    all_ = 0.
    for predict,target in zip(predicts,targets):
        for t1, t2 in zip(predict,target) :
            if t1 == t2:
                right += 1
            all_ += 1
    return right/float(all_)

def confusion_matrix(tags_count,predicts, targets):
    #matrix = np.zeros((len(tags_count),len(tags_count)))
    confusion_dict = defaultdict(Counter)
    #vocab = dict(zip(tags_count.keys(),range(len(tags_count))))

    for predict,target in zip(predicts, targets):
        for t1,t2 in zip(predict,target):
            confusion_dict[t1][t2] += 1

    for key in confusion_dict.keys():
        if confusion_dict[key][key] == tags_count[key]:
            continue
        else:
            for key2 in confusion_dict[key].keys():
                if confusion_dict[key][key2] != 0 and key != key2 and confusion_dict[key][key2]>500:
                    print (key,"-",key2,":",confusion_dict[key][key2])

def select(tag1,tag2,train_X,train_Y,words):
    tag1_previous = Counter()
    tag2_previous = Counter()
    tag1_following  = Counter()
    tag2_following  = Counter()
    for i,tags in enumerate(train_Y):
        for j,tag in enumerate(tags):
            if tag == tag1 and train_X[i][j] in words:
                pre = tags[j-1] if j>0 else "<s>"
                fol = tags[j+1] if j<len(tags)-1 else "<e>"
                tag1_previous[pre] += 1
                tag1_following[fol] += 1
            if tag == tag2 and train_X[i][j] in words:
                pre = tags[j-1] if j>0 else "<s>"
                fol = tags[j+1] if j<len(tags)-1 else "<e>"
                tag2_previous[pre] += 1
                tag2_following[fol] += 1
    for key in tag2_previous.most_common(5):
        key = key[0]
        print("{}:{}-{} = {} ".format(key,tag2_previous[key],tag1_previous[key],tag2_previous[key]-tag1_previous[key]))
    print ("*"*10)
    for key in tag2_following.most_common(5):
        key = key[0]
        print("{}:{}-{} = {} ".format(key,tag2_following[key],tag1_following[key],tag2_following[key]-tag1_following[key]))