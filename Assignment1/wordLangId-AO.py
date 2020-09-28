from NGram import *


# English
file = open('EN.txt', "r", encoding="MacRoman")
file_cont = file.readlines()
tokens = [word_tokenize(line) for line in file_cont]
en_model = ngram_model_gene(2, tokens, "EN")

# French
file = open('FR.txt', "r", encoding="MacRoman")
file_cont = file.readlines()
tokens = [word_tokenize(line) for line in file_cont]
fr_model = ngram_model_gene(2, tokens, "FR")

# German
file = open('GR.txt', "r", encoding="MacRoman")
file_cont = file.readlines()
tokens = [word_tokenize(line) for line in file_cont]
gr_model = ngram_model_gene(2, tokens, "GR")

lang_models = [en_model, fr_model, gr_model]
test_file = open('LangID.test.txt', "r", encoding="MacRoman")
sentences = test_file.readlines()
cleaned = []
tokenized = []
probs = []
count = 0

for line in sentences:
    line = re.findall(r"^\d+\. (.*)$",line)[0]
    words = [word_tokenize(line)]
    lang_probability = calc_prob(lang_models, ngram_dict_gene(2, words), 1, True)
    probs.append(lang_probability)

results = []
for i in range(len(probs)):
    results.append(max(probs[i], key=probs[i].get))
acc = check_accuracy(results)
print("Percent Correct: ", acc * 100)

output_file = open('WordLangId-AO.out', 'w')
output_file.write("ID LANG\n")
for i in range(len(results)):
    output_file.write("%s. %s\n" % (i + 1, results[i]))
output_file.write('\n Percent Correct: %s' % int(acc * 100))
print("results have been wrote in WordLangId-AO.out file")
output_file.close()