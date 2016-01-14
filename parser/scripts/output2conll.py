#!/usr/bin/env python
import sys
import gzip
import cPickle as pickle
import numpy as np
usage='''
Converts output file to conll format

pyton output2conll.py <reference file> <.output file>

<reference file>    : reference conll file
<.output file>      : file to be converted to conll format

<output file>.conll file to be written

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
		head = str(int(predictions[i]))
		print >> conll_f, '\t'.join(l[0:6]+[head]+['-']*3)
	print >> conll_f

if len(sys.argv) != 3:
	print >> sys.stderr, usage
	quit(0)


ref_f = open_file(sys.argv[1])
output_f = open_file(sys.argv[2])
conll_f = open(sys.argv[2] +'.conll','w')

s = []
for line in ref_f:
	l = line.strip().split('\t')
	if len(l) < 10:
		write_conll(conll_f,s,output_f.readline().strip().split())
		s = []
		continue
	s.append(l)
conll_f.close()
