#!/usr/bin/env python
"""
Convert a tab-separated sequence labeling data for training

@author Volkan Cirik
"""
import sys
import theano
import numpy as np

class CharacterTable(object):
	"""
	Given a set of characters:
	+ Encode them to a one hot integer representation
	+ Decode the one hot integer representation to their character output
	+ Decode a vector of probabilties to their character output
	"""
	def __init__(self, chars, maxlen):
		self.chars = chars
		self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
		self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
		self.maxlen = maxlen

	def encode(self, C, maxlen=None):
		maxlen = maxlen if maxlen else self.maxlen
		X = np.zeros((maxlen, len(self.chars)))
		for i, c in enumerate(C):
			X[i, self.char_indices[c]] = 1
		return X

	def decode(self, X, calc_argmax=True):
		if calc_argmax:
			X = X.argmax(axis=-1)
		return ' '.join(str(self.indices_char[x]) for x in X)


def get_max_len(file_name):

	try:
		in_file = open(file_name)
	except:
		print('File {} does not exist'.format(file_name))
		quit(0)
	max_length = 0
	s = []
	for line in in_file:
		l = line.strip().split()
		if len(l) < 1:
			if len(s) > max_length:
				max_length = len(s)
			s = []
			continue
		s.append(l)
	return max_length

def process_file(file_name, max_length, root = -1):

	try:
		in_file = open(file_name)
	except:
		print('File {} does not exist'.format(file_name))
		quit(0)

	S = []
	s = []

	for line in in_file:
		l = line.strip().split('\t')
		if len(l) <= 1:
			if len(s) > max_length or len(s) < 1:
				s = []
				continue
			S.append(s)
			s = []
			continue
		s.append(l)
	if len(s) <= max_length and len(s) >= 1:
		S.append(s)

	num_seq = len(S)
	dim = len(S[0][0][10].split(' '))

	X = np.zeros((num_seq,max_length, dim), dtype=theano.config.floatX)
	if root == -1:
		Y = np.zeros((num_seq,max_length,max_length + 1), dtype=np.bool)
		Y[:,:,0] = True
	else:
		Y = np.zeros((num_seq,max_length,max_length), dtype=np.bool)

	for s_id,s in enumerate(S):
		for j,line, in enumerate(s):
			if root == -1:
				head = int(line[6])
				if head == 0:
					head = -1
			else:
				head = int(line[6]) -1
				if head == -1:
					head = j
			v = [float(val) for val in line[10].split()]

			X[s_id,j,:] = v
			Y[s_id,j,head] = 1
			if root == -1:
				Y[s_id,j,0] = 0
	return X,Y

def prepare_conll(file_list, root = -1):
	[tr_,val_,te_] = file_list
	X = []
	Y = []
	max_length = -1
	for f in file_list:
		ml = get_max_len(f)
		if max_length < ml:
			max_length = ml

	for f in file_list:
		x,y = process_file(f,max_length, root = root)
		X.append(x)
		Y.append(y)
	return X,Y
