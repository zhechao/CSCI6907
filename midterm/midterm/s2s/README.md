This porject used tensorflow=1.15.0, please do not use tensorflow=2.x
Highly suggest train the model on GPU insteand of CPU.
If using the linux system, please uncomment line 82 in train and line 55 in test, and comment line 83 in train and line 56 in test
run train.py first, it may cost a few mintunus if you train on GPU, or hours if you train on CPU, then run the test.py 