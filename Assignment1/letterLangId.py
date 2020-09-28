from NGram import *


file = open('EN.txt', "r", encoding="MacRoman")
file_cont = file.readlines()
tokens = [letter_tokenize(line) for line in file_cont]
en_model = ngram_model_gene(2, tokens, "EN")

file = open('FR.txt', "r", encoding="MacRoman")
file_cont = file.readlines()
tokens = [letter_tokenize(line) for line in file_cont]
fr_model = ngram_model_gene(2, tokens, "FR")

file = open('GR.txt', "r", encoding="MacRoman")
file_cont = file.readlines()
tokens = [letter_tokenize(line) for line in file_cont]
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
    letters = [letter_tokenize(line)]
    lang_probability = calc_prob(lang_models, ngram_dict_gene(2, letters), 0, False)
    probs.append(lang_probability)

results = []
for i in range(len(probs)):
    results.append(max(probs[i], key=probs[i].get))
acc = check_accuracy(results)
print("Percent Correct: ", acc * 100)

output_file = open('LetterLangId.out', 'w')
output_file.write("ID LANG\n")
for i in range(len(results)):
    output_file.write("%s. %s\n" % (i + 1, results[i]))
output_file.write('\n Percent Correct: %s' % int(acc * 100))
print("results have been wrote in LetterLangId.out file")
output_file.close()
