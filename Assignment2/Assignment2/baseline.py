
from collections import Counter
from collections import defaultdict


class baseline():
    
    def __init__(self):

        pass

    def train(self,train_X, train_Y):
      
        word_tag_dict = defaultdict(Counter)# To calculate the emission probabilities
        tag_count= Counter()
        tag_word_dict = defaultdict(set)

        for i in range(len(train_X)):
            words = train_X[i]
            tags = train_Y[i]
            for j,(word,tag) in enumerate(zip(words,tags)):        
                word_tag_dict[word][tag] += 1  
                tag_count[tag] += 1  
                tag_word_dict[tag].add(word)         
        self.word_tag_dict = word_tag_dict   
        self.tag_count = tag_count
        self.tag_word_dict = tag_word_dict
    
    def predict(self,words):
        tags = []
        for word in words:
            if word not in self.word_tag_dict:
                tags += ["NN"]

            else:
                tag = max(self.word_tag_dict[word].keys(),key=self.word_tag_dict[word].get)
                tags.append(tag)
        return tags


    def get_all_tags(self,words):
        tags = []
        num = []
        for word in words:
            if word not in self.word_tag_dict:
                tags += [{"NN":1}]
                num.append(1)
            else:
                tag = self.word_tag_dict[word]
                tags.append(tag)
                num.append(len(tag))
        return tags,num

    def predict_with_rule(self,words,rule):
        tags_dict, num = self.get_all_tags(words)
        tags = []
        for i,word in enumerate(words):
            if num[i] == 1:
                tags.append(list(tags_dict[i].keys())[0])
            else:
                ts = tags_dict[i].keys()
                for t1 in ts:
                    for t2 in ts:
                        if (t1,t2) in rule:
                            pre = tags[-1] if len(tags)>=1 else '<s>'

                            fol = max(tags_dict[i+1],key=tags_dict[i+1].get) if i<len(tags_dict)-1 else "<e>"
                            
                            if ("following",fol) in rule[(t1,t2)]:

                                tags.append(rule[(t1,t2)][("following",fol)])
                                break
                            elif ("previous",pre) in rule[(t1,t2)]:
                                tags.append(rule[(t1,t2)][("previous",pre)])
                                break
                    else:
                        continue
                    break
                else:
                    tags.append(max(tags_dict[i],key=tags_dict[i].get))
        return tags




    








