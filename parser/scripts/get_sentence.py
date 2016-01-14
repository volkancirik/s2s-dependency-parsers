#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
usage='''
get a sentence from from a conll file and print it.

python get_sentence.py <reference> < sentence id #>
'''

if len(sys.argv) != 3:
	print(usage)
	quit(0)

s = []
ref = []
for line in open(sys.argv[1]):
	l = line.strip().split('\t')
	if len(l) <= 1:
		ref.append(s)
		s = []
		continue
	s.append(l)
if len(s) > 2:
	ref.append(s)

tokens = [ line[1] for line in ref[int(sys.argv[2])]]
arcs = [ line[6] for line in ref[int(sys.argv[2])]]
print "Sentence %d " % int(sys.argv[2])
print " ".join(tokens)
print " ".join(arcs)

