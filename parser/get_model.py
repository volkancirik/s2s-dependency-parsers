# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Graph
from keras.layers.core import Activation,TimeDistributedDense, RepeatVector
from keras.layers import recurrent
from keras.optimizers import RMSprop
from keras.layers.attention import TimeDistributedAttention, PointerPrediction
'''
Get parser model options are enc2dec, attention-based and pointer net
'''

UNIT = {'rnn' : recurrent.SimpleRNN, 'gru' : recurrent.GRU, 'lstm' : recurrent.LSTM}
def get_enc2dec(RNN, HIDDEN_SIZE = 128, LAYERS = 1, DIM = 100, MAXLEN = 100):
	"""
	Enc-Dec Model
	see Vinyals et. al. 2014 http://arxiv.org/pdf/1412.7449v1.pdf
	"""
	model = Graph()

	model.add_input(name='input', input_shape=(None,DIM))
	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='e_r0', input='input')

	prev_node = 'e_r0'
	for layer in xrange(LAYERS-1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='e_r'+str(layer+1), input=prev_node)
		prev_node = 'e_r'+str(layer+1)

	model.add_node(RNN(HIDDEN_SIZE), name='e_final', input=prev_node)
	model.add_node(RepeatVector(MAXLEN), name='encoder', input='e_final')

	prev_node = 'encoder'
	for layer in xrange(LAYERS-1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='d_r'+str(layer+1), input=prev_node)
		prev_node = 'd_r'+str(layer+1)

	model.add_node(TimeDistributedDense(MAXLEN), name='d_tdd', input=prev_node)
	model.add_node(Activation('softmax'), name = 'softmax',input = 'd_tdd')
	model.add_output(name='output', input='softmax')

	return model

def get_attention(RNN, HIDDEN_SIZE = 128, LAYERS = 1, DIM = 100, MAXLEN = 100):
	"""
	Attention-based Decoder
	see Bahdanau et. al. 2014 http://arxiv.org/pdf/1409.0473.pdf
	"""
	model = Graph()

	model.add_input(name='input', input_shape=(None,DIM))
	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='recurrent_context', input='input')

	prev_node = 'recurrent_context'
	for layer in xrange(LAYERS-1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='e_r'+str(layer+1), input=prev_node)
		prev_node = 'e_r'+str(layer+1)
	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='encoder_context', input=prev_node)
	model.add_node(TimeDistributedAttention(prev_dim = HIDDEN_SIZE, att_dim = HIDDEN_SIZE, return_sequences = True, prev_context = False), name='attention', inputs=['encoder_context','recurrent_context'], merge_mode = 'join_att')

	prev_node = 'attention'
	for layer in xrange(LAYERS-1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='d_r'+str(layer+1), input=prev_node)
		prev_node = 'd_r'+str(layer+1)

	model.add_node(TimeDistributedDense(MAXLEN), name='d_tdd', input=prev_node)
	model.add_node(Activation('softmax'), name = 'softmax',input = 'd_tdd')
	model.add_output(name='output', input='softmax')
	return model

def get_pointer(RNN, HIDDEN_SIZE = 128, LAYERS = 1, DIM = 100, MAXLEN = 100):
	"""
	Pointer Network
	see Vinyals et. al. 2015 http://arxiv.org/abs/1506.03134
	"""
	model = Graph()

	model.add_input(name='input', input_shape=(None,DIM))
	model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='e_r0', input='input')

	prev_node = 'e_r0'
	for layer in xrange(LAYERS-1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='e_r'+str(layer+1), input=prev_node)
		prev_node = 'e_r'+str(layer+1)

#### do we need tdd? prev_context?
	model.add_node(TimeDistributedDense(HIDDEN_SIZE), name='encoder_context', input=prev_node)

	prev_node = 'e_r0'
	for layer in xrange(LAYERS-1):
		model.add_node(RNN(HIDDEN_SIZE, return_sequences = True), name='d_r'+str(layer+1), input=prev_node)
		prev_node = 'd_r'+str(layer+1)
	model.add_node(TimeDistributedDense(HIDDEN_SIZE), name='recurrent_context', input=prev_node)

	model.add_node(PointerPrediction(prev_dim = HIDDEN_SIZE, att_dim = HIDDEN_SIZE, return_sequences = True, prev_context = True), name='pointer', inputs=['encoder_context','recurrent_context'], merge_mode = 'join_att')

	model.add_output(name='output', input='pointer')
	return model

def grab_model(m,rnn,h,l,d,maxlen):
	M = { 'enc2dec' : get_enc2dec, 'attention' : get_attention, 'pointer' : get_pointer}
	return M[m](UNIT[rnn],HIDDEN_SIZE = h, LAYERS = l, DIM = d, MAXLEN = maxlen)
