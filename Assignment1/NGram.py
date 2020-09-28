import re
import nltk
import copy


# use class to hold data for analysis
class NGram:
    def __init__(self, name, n, ngrams, unigrams, bigrams=None):
        self.name = name
        self.n = n
        self.ngrams = ngrams
        self.unigrams = unigrams
        self.tokensCount = sum(unigrams.values())
        self.bigrams = bigrams
        N = 0
        for n in ngrams:
            N += ngrams[n]
        self.N = N
        ngram_num_dict = {}
        for n in ngrams:
            ngram_num_dict[ngrams[n]] = ngram_num_dict.get(ngrams[n], 0) + 1
        self.Ngram_num_dict = ngram_num_dict


# use nltk to do letter tokenize
def letter_tokenize(cont):
    return list(cont.strip().lower())


def word_tokenize(cont):
    return nltk.word_tokenize(cont.strip().lower())


# get the dict of bigram probabilities
def bigram_gene(tokens):
    bigrams_dict = {}

    i = 0
    for t in tokens:
        if i < len(tokens) - 1:
            if (t, tokens[i + 1]) in bigrams_dict.keys():
                bigrams_dict[(t, tokens[i + 1])] += 1
            else:
                bigrams_dict[(t, tokens[i + 1])] = 1
        i += 1

    for ngram in bigrams_dict.keys():
        bigrams_dict[ngram] = bigrams_dict[ngram] / tokens.count(ngram[0])

    return bigrams_dict


# get the dict of ngram dict based on size n with its frequency
def ngram_dict_gene(n, lines):
    ngrams_dict = {}

    for tokens in lines:

        for i in range(max(n - 1, 1)):
            tokens.insert(0, "[[START]]")

        index = 0
        for t in range(len(tokens)):
            ngram = ()
            for i in range(n):
                if t + i >= len(tokens):
                    ngram += ("[[END]]",)
                else:
                    ngram += (tokens[t + i],)
            if ngram in ngrams_dict:
                ngrams_dict[ngram] += 1
            else:
                ngrams_dict[ngram] = 1
            index += 1

    return ngrams_dict


def ngram_model_gene(n, tokens, name):
    unigrams = ngram_dict_gene(1, tokens)
    # print (unigrams)
    ngrams = ngram_dict_gene(n, tokens)
    # print (ngrams)
    if n == 3:
        bigrams = ngram_dict_gene(2, tokens)
    else:
        bigrams = None
    model = NGram(name, n, ngrams, unigrams, bigrams)
    return model


def add_one_sm(line_model, lang_models):
    results = {}

    for lang in lang_models:
        probs = 1.
        # unigrams = copy.deepcopy(lang.unigrams)
        for ngram in line_model:
            N = lang.ngrams.get(ngram, 0) + 1
            if ngram[:1] in lang.unigrams:
                #  ngram count / (ngram count + unigrams #)
                probs *= N / (len(lang.unigrams) + lang.unigrams[ngram[:1]])
            else:
                probs *= 1e-10
        results[lang.name] = probs
    # print (results)
    return results


def without_smoothing(line_model, lang_models):
    results = {}

    for lang in lang_models:
        probs = 1.
        for ngram in line_model:
            N = lang.ngrams.get(ngram, 0)
            if ngram[:1] in lang.unigrams:
                probs *= N / lang.unigrams[ngram[:1]]
            else:
                # print (ngram[:1])
                probs *= 1e-10
        results[lang.name] = probs
    # print (results)
    return results


def good_turing_smoothing(test_grams, lang_models, k=5):
    results = {}
    for lang in lang_models:
        probs = 1.

        for n in test_grams:
            if n not in lang.ngrams:
                prob = lang.Ngram_num_dict[1] / lang.N
            elif lang.ngrams[n] <= k:
                prob = (lang.ngrams[n] + 1) * lang.Ngram_num_dict[lang.ngrams[n] + 1] / lang.Ngram_num_dict[
                    lang.ngrams[n]] / lang.N
            else:
                prob = lang.ngrams[n] / lang.N

            if n[:1] in lang.unigrams:
                probs *= prob / (lang.unigrams[n[:1]] / lang.tokensCount)
            else:
                probs *= 1e-10
        results[lang.name] = probs
    return results


# use katz back off to calculate LM
def katz_back_off(test_grams, lang_models):
    results = {}

    for lang in lang_models:
        probs = 1.
        for ngram in test_grams:
            if ngram not in lang.ngrams:
                bigram_1 = (ngram[0], ngram[1])
                bigram_2 = (ngram[1], ngram[2])

                if bigram_1 not in lang.bigrams and bigram_2 not in lang.bigrams:
                    if ngram[0] not in lang.unigrams and ngram[1] not in lang.unigrams and ngram[2] not in lang.unigrams:
                        probs *= 1e-10
                    else:
                        if ngram[0] in lang.unigrams:
                            probs *= lang.unigrams[ngram[0]] / lang.tokensCount
                        if ngram[1] in lang.unigrams:
                            probs *= lang.unigrams[ngram[1]] / lang.tokensCount
                        if ngram[2] in lang.unigrams:
                            probs *= lang.unigrams[ngram[2]] / lang.tokensCount
                else:
                    if bigram_1 in lang.bigrams:
                        probs *= lang.bigrams[bigram_1] if bigram_1[:1] in lang.unigrams else 1e-10
                    if bigram_2 in lang.bigrams:
                        probs *= lang.bigrams[bigram_2] if bigram_2[:1] in lang.unigrams else 1e-10
            else:
                if ngram[:2] in lang.bigrams:
                    probs *= lang.ngrams[ngram] / lang.bigrams[ngram[:2]]
                else:
                    probs *= 1e-10
        results[lang.name] = probs

    return results


def get_unigram_freq(tokens):
    unigrams_dict = {}

    for t in tokens:
        if t in unigrams_dict.keys():
            unigrams_dict[t] += 1
        else:
            unigrams_dict[t] = 1
    return unigrams_dict


def calc_prob(lang_models, test_grams, smoothing_type, smoothing):
    if smoothing:
        if smoothing_type == 1:
            probabilities = add_one_sm(test_grams, lang_models)
        elif smoothing_type == 2:
            probabilities = good_turing_smoothing(test_grams, lang_models)
        elif smoothing_type == 3:
            probabilities = katz_back_off(test_grams, lang_models)
    else:
        probabilities = without_smoothing(test_grams, lang_models)

    return probabilities


# for q11 to get the accuracy
def check_accuracy(res):
    file = open('LangID.gold.txt')
    file_cont = file.readlines()
    correct = 0
    for i in range(1, len(file_cont) - 1):
        lg = re.findall('[A-Z]+', file_cont[i])[0]
        if lg == res[i - 1]:
            correct += 1
    return correct / 150


# function to calculate different language probabilities based on corresponding training data
def lang_probability(line, en_unigrams, en_bigrams, fr_unigrams, fr_bigrams, gr_unigrams, gr_bigrams):
    en = 1
    fr = 1
    gr = 1
    i = 0
    for token in line:
        if i == 0:
            if token in en_unigrams:
                en *= en_unigrams[token] / sum(en_unigrams.values())
            else:
                en = 0
            if token in fr_unigrams:
                fr *= fr_unigrams[token] / sum(fr_unigrams.values())
            else:
                fr = 0
            if token in gr_unigrams:
                gr *= gr_unigrams[token] / sum(gr_unigrams.values())
            else:
                gr = 0
        if i < len(line) - 1:
            bi = (token, line[i + 1])
            if bi in en_bigrams:
                en *= en_bigrams[bi]
            if bi in fr_bigrams:
                fr *= fr_bigrams[bi]
            if bi in gr_bigrams:
                gr *= gr_bigrams[bi]
        i += 1
    return [en, fr, gr]
