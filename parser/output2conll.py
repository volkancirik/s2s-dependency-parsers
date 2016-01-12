#!/usr/bin/env python
import sys
import gzip
import cPickle as pickle
import numpy as np
usage='''
Converts embedding file to pickle file
pyton <vector file> <pickle file for output> <skip first line 1|0 > <unk word>

<vector file>     : plaint txt file <word> <vector of numbers>
<output file>     : output file name with .pkl extension
<skip first line> : vector file's first line should be skipped or not
<unk word>        : vector for unknown words. if exists in vector file it is used. othervise average of word vectors will be set as a vector

'''

def open_file(f_name):
	try:
		f = open(f_name)
	except:
		print >> sys.stderr, "%s cannot be opened" % (f_name)
		quit(0)
	return f

def write_conll(conll_f, sentence, predictions):
	for i,l in enumerate(sentence):

		if int(predictions[i]) == i:
			head = '0'
		else:
			head = str(int(predictions[i])+1)
		print >> conll_f, '\t'.join(l[0:6]+[head]+['-']*3)
	print >> conll_f
test_f = open_file(sys.argv[1])
output_f = open_file(sys.argv[2])
conll_f = open(sys.argv[3],'w')

s = []
for line in test_f:
	l = line.strip().split('\t')
	if len(l) <= 1:
		write_conll(conll_f,s,output_f.readline().strip().split())
		s = []
		continue
	s.append(l)
conll_f.close()
