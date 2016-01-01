#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
usage='''
Evaluate output of Neural Parsers

python eval.py <reference> <prediction> <model_type {0 : enc2dec,attention, 1 : pointer}>
'''

if len(sys.argv) != 4:
	print(usage)
	quit(0)

model_type = int(sys.argv[3])
ref = []
s = []
for line in open(sys.argv[1]):
	l = line.strip().split('\t')
	if len(l) <= 1:
		ref.append(s)
		s = []
		continue
	s.append(int(l[6]))
if len(s) > 2:
	ref.append(s)
pred = [ line.strip().split() for line in open(sys.argv[2])]

assert(len(pred) == len(ref))

n = 0
accurate = 0
for r,p in zip(ref,pred):
	for i in range(len(r)):
		if model_type == 1:
			if (int(p[i]) + 1 == r[i]) or (r[i] == 0 and int(p[i]) == i):
				accurate +=1
		else:
			if (int(p[i]) == r[i]) or (r[i] == 0 and int(p[i]) == -1):
				accurate +=1
	n += len(r)
print("in {} total {} arcs and {} of them are correct. ula {}".format(sys.argv[2].split('/')[-1].split('.')[0],n,accurate,accurate * 1.0 / n))
