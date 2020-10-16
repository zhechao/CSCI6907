Q2. Hidden Markov Models for parts-of-speech tagging: 

The introduction of python file:
Filename	Introduction
a.py	The source code for Question a)
b.py	The source code for Question b)
c.py	The source code for Question c)
d.py	The source code for Question d)
baseline.py	The implement of a majority-class baseline
HMM.py	The implement of the HMM model
utils.py	The source code for processing the data
evaluate.py	Evaluate the results of Question a)-d)

The results.
Question	cmd	Results filename	Accuracy
a	python3 a.py	result/brown.test.baseline.txt	90.99%
b	python3 b.py	result/brown.test.rule.txt	91.55%
c	python3 c.py	result/brown.test.hmm.txt	95.31%
d	python3 d.py	result/brown.test.beam_search.txt	95.32%
Note: Running c.py needs more than one hour.

If you want to directly evaluate all models through the results file:
python3 evaluate.py 
Output:
Question a
The accuracy of the baseline model is 0.9099062208408574
Question b
The accuracy of the baseline model with rule is 0.9155450536024757
Question c
The accuracy of the HMM model is 0.9530902817402512
Question d
The accuracy of the HMM model with beam search is 0.9532245396631469
