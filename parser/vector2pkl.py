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
if len(sys.argv) != 5:
	print usage
	quit(0)

v_file = sys.argv[1]
pickle_file = sys.argv[2]
skip_first = bool(sys.argv[3])
unk_word = sys.argv[4]

v = {}
n = 0
for i,line in enumerate(open(v_file)):
	if skip_first and i == 0:
		continue

	l = line.strip().split()
	token = l[0].lower()
	vec = [float(val) for val in l[1:]]
	v[token] = np.array(vec)

	n += 1

dim = len(vec)

if unk_word not in v:
	unk = np.array([0.0]*dim)
	for w in v:
		unk += v[w]
	unk = unk *1.0 / n
	v[unk_word] = unk
else:
	print("found unk!")

out = gzip.open(pickle_file,'wb')
pickle.dump(v,out)
out.close()
