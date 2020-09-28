Use nltk for word/letter tokenize, so may need to install nltk before running the files.\
If use terminal/cmd, use py + "run_file" to get the output, if use pycharm, then just compile the run_file.
The run file and the accuracy (#of correctly identified pairs/Total) for each language model above using the gold labels in LangID.gold.txt:

#Q7
	run_file: letterLangId.py
	acc: 62.0% 
	result_file: LetterLangId.out            
#Q8
    run_file: wordLangId-AO.py
	acc: 98.0% 
	result_file: wordLangId-AO.out
#Q9
    run_file: wordLangId-GT.py
	acc: 96.7% 
	result_file: wordLangId-GT.out
#Q10
    run_file: trigramWordLangId.py
	acc: 95.3%	
	result_file: trigramWordLangId.out
