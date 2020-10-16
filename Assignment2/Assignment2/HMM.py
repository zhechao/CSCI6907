
from collections import Counter
from collections import defaultdict

class HMM():
    
    def __init__(self):

        pass

    def train(self,train_X, train_Y):

        
        #word_tag_pair_count = Counter() # To calculate the emission probabilities
        ngram_tag_count = Counter() # To calculate the  transition probabilities
        previous_tag_count = Counter() # To calculate the prior probability 
        word_vocab = Counter()
        tag_vocab = Counter()
        word_tag_dict = defaultdict(Counter)# To calculate the emission probabilities


        for i in range(len(train_X)):

            words = train_X[i]
            tags = train_Y[i]

            for j,(word,tag) in enumerate(zip(words,tags)):

                
                #word_tag_pair_count[(word,tag)] += 1
                word_vocab[word] += 1
                tag_vocab[tag] += 1
                word_tag_dict[word][tag] += 1

                
                tag_previous = tags[j-1] if j>=1 else "<s>"
                ngram_tag_count[(tag_previous,tag)] += 1
                previous_tag_count[tag_previous] += 1


                
        word_tag_dict["UNK"] = 'NN'
        self.word_tag_dict = word_tag_dict
        self.ngram_tag_count = ngram_tag_count
        self.previous_tag_count = previous_tag_count

        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab

    def get_emission_prob(self,word,tag):
        
        return self.word_tag_dict[word][tag]/float(self.tag_vocab[tag])  if self.word_tag_dict[word][tag] > 0 else 1e-12


    def get_transition_prob(self,tag,previous_tag):
        try:
            return self.ngram_tag_count[tag] / float(self.previous_tag_count[previous_tag])
        except:
            return 1e-12
    def viterbi_decoder(self,words):
        paths = []
        V = []

        # 1. Initialization
        V.append({"<s>":1.})

        # 2. Recursion
        for t,word in enumerate(words):

            #word = word if word in self.word_vocab else "<UNK>"

            #possible_tags = self.word_tag_dict[word].keys()
            possible_tags = self.tag_vocab.keys()

            paths += [{}]
            v = {}
            for tag in possible_tags:  
                v[tag],pre_tag = max([(V[t][previous_tag]* self.get_transition_prob((previous_tag,tag),previous_tag)*self.get_emission_prob(word,tag), previous_tag) for previous_tag in V[t].keys()])
                paths[-1][tag] = pre_tag
            V.append(v)
            


        # 3. Termination
        last_tag = max(V[-1],key=V[-1].get)
        tags = [last_tag]

        for i in range(len(paths)):
            tag = paths[len(paths)-1-i][last_tag]
            if tag == "<s>":
              break
            tags.insert(0,tag)
            last_tag = tag    

        return tags

    def beam_search(self,words,beam_size = 10):

        paths = []
        V = []

        # 1. Initialization
        
        V.append({"<s>":1.})

        # 2. Recursion
        for t,word in enumerate(words):

            possible_tags = self.tag_vocab.keys()

            path = {}
           
            v = {}
            
            for tag in possible_tags:
                
                v[tag],pre_tag = max([(V[t][previous_tag]* self.get_transition_prob((previous_tag,tag),previous_tag)*self.get_emission_prob(word,tag), previous_tag) for previous_tag in V[t].keys()])
                
                path[tag] = pre_tag

            sort_v = sorted(v.items(), key=lambda x:x[1],reverse=True)[:beam_size]
            v = dict(sort_v)
            path_ = {}
            for tag in v.keys():
                path_[tag] = path[tag]

            V.append(v)
            paths.append(path_)

        # 3. Termination
        last_tag = max(V[-1],key=V[-1].get)
        tags = [last_tag]

        #if self.ngram == 2:
        for i in range(len(paths)):
            tag = paths[len(paths)-1-i][last_tag]
            if tag == "<s>":
                break
            tags.insert(0,tag)
            last_tag = tag    

        return tags        











