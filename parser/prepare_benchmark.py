#!/usr/bin/env python
"""
Convert a tab-separated sequence labeling data with categorical features and embed word vectors

@author Volkan Cirik
"""
import optparse
import sys
import theano
import numpy as np
import pickle
import cPickle
import gzip

UNK="*UNKNOWN*"

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


def process_file(file_name, v_map, feature_map, feature_length, max_length, dim, label_idx, features):

	print file_name
	try:
		in_file = open(file_name)
	except:
		print >> sys.stderr, "sequence file",file_name,"cannot be read in process_file()"
		quit(1)

	S = []
	s = []

	for line in in_file:
		l = line.strip().split()
		if len(l) < 1:
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

	X = np.zeros((num_seq,max_length, dim), dtype=theano.config.floatX)
	Y = np.zeros((num_seq,max_length,max_length), dtype=np.bool)

	unk = 0.0
	ntok = 0.0
	for s_id,s in enumerate(S):
		for j,line, in enumerate(s):
			ntok += 1
			try:
				v = v_map[line[1]]
			except:
				unk +=1
				v = v_map[UNK]
				pass
			categorical_f = []

			for i,feature_idx in enumerate(features):
				f = line[feature_idx]
				f_vec = [0]*feature_length[i]
				f_vec[ feature_map[i][f] ] = 1
				categorical_f += f_vec

			assert len(v) + len(categorical_f) == dim
			##########
			head = int(line[label_idx]) -1
			if head == -1:
				head = j
			##########

			X[s_id,j,:] = list(v) + categorical_f
			Y[s_id,j,head] = 1

	batch = (X,Y)
	return batch, unk/ntok

def pre_process(tr_,val_,te_,vector_file, max_length, features = []):

	try:
		tr_file = open(tr_)
	except:
		print >> sys.stderr, "seqence file",tr_,"cannot be read"
		quit(1)
	try:
		val_file = open(val_)
	except:
		print >> sys.stderr, "seqence file",val_,"cannot be read"
		quit(1)
	try:
		te_file = open(te_)
	except:
		print >> sys.stderr, "seqence file",te_,"cannot be read"
		quit(1)

	if vector_file != '':
		try:
			print >> sys.stderr, "loading word vectors..."
			v_map = cPickle.load(gzip.open(vector_file, "rb"))
			print >> sys.stderr, "word vectors are loaded.."
		except:
			print >> sys.stderr, "word embedding file",vector_file,"cannot be read."
			quit(1)

	voc = {'*UNKNOWN*' : 0}
	voc_idx = 1

	S = []

	data_max_len = 0
	for in_file in [tr_file, val_file, te_file]:
		s = []
		for line in in_file:
			l = line.split()

			if len(l) < 2:
				if (len(s) > max_length or len(s) < 1) and max_length != -1:
					s = []
					continue
				S.append(s)
				if data_max_len < len(s):
					data_max_len = len(s)
				s = []
				continue
			s.append(l)
		if (len(s) <= max_length and len(s) >= 1) or max_length == -1:
			S.append(s)
			if data_max_len < len(s):
				data_max_len = len(s)

	feature_map = {}
	for i in xrange(len(features)):
		feature_map[i] = {}
	feature_length = [0]*len(features)

	for s in S:
		for line in s:
			tok = line[1]
			if tok not in voc:
				voc[tok] = voc_idx
				voc_idx += 1
			for i,feature_idx in enumerate(features):
				f = line[feature_idx]
				if f not in feature_map[i]:
					feature_map[i][ f ] = feature_length[i]
					feature_length[i] += 1

	if vector_file == '':
		v_map = {}
		for w in voc:
			vec = [0]*voc_idx
			vec[ voc[w]] = 1
			v_map[w] = vec

	if max_length == -1:
		max_length = data_max_len
	return v_map, feature_map, feature_length, len(v_map[UNK]) + sum(feature_length), max_length

def convertData( tr_file, val_file, te_file, vector_file, max_length = -1, features = [3], label_idx = 6):
	v_map, feature_map, feature_length, dim, max_length = pre_process(tr_file,val_file,te_file,vector_file, max_length = max_length, features = features)

	train, unk_tr = process_file(tr_file, v_map, feature_map, feature_length, max_length, dim, label_idx, features)
	val, unk_val = process_file(val_file, v_map, feature_map, feature_length, max_length, dim, label_idx, features)
	test, unk_te = process_file(te_file, v_map, feature_map, feature_length, max_length, dim, label_idx, features)

	dataset = (train,val,test)
	meta_data = feature_map, feature_length, max_length, dim

	tr_len = train[0].shape[0]
	val_len =val[0].shape[0]
	test_len = test[0].shape[0]

	print >> sys.stderr, "dimension of feature vector is %d, max sequence length %d" % (dim,max_length)
	print >> sys.stderr, "Training file has %d sequences and unk rate for %.2f" %(tr_len,unk_tr)
	print >> sys.stderr, "Validation file has %d sequences and unk rate for %.2f" %(val_len,unk_val)
	print >> sys.stderr, "Training file has %d sequences and unk rate for %.2f" %(test_len,unk_te)
	return meta_data, dataset

if __name__ == "__main__":
	optparser = optparse.OptionParser()
	optparser.add_option("--train", dest="tr_file", help="training file. space separated, last column label")
	optparser.add_option("--val", dest="val_file", help="validation file. space separated, last column label")
	optparser.add_option("--test", dest="te_file", help="test file. space separated, last column label")
	optparser.add_option("--vector", dest="vector_file", help="word embeddings pkl, defaul '' means use one-hot vectors", default = '')

	optparser.add_option("--features", dest="features", help="list of columns of caregorical features, ex 3-4-5,", default = '3')

	optparser.add_option("--max-length", dest="max_length", type=int,default=-1,help="maximum sequence length. default = -1 meaning use datasets max sequence length")

	(opts, _) = optparser.parse_args()

	meta_data,dataset =	convertData(opts.tr_file,opts.val_file,opts.te_file,opts.vector_file,opts.max_length, features = [int(f) for f in opts.features.split('-')])

