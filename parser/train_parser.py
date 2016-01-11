# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Graph
from keras.optimizers import RMSprop
import numpy as np
from prepare_benchmark import CharacterTable
from utils import get_parser1
from get_model import grab_model
import json, time, datetime, os
import cPickle as pickle
import os

from convert_benchmark import convertData
"""
Dependency Parser with purely Neural Network-based models
"""

parser = get_parser1()
p = parser.parse_args()
TIMESTAMP = "_".join(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S').split())

HIDDEN_SIZE = p.n_hidden
BATCH_SIZE = p.batch_size
LAYERS = p.n_layers
TR = p.tr
VAL = p.val
TEST = p.test
VECTOR = p.vector
PATIENCE = p.patience
MODEL = p.model
PREFIX = 'exp/'+p.prefix + '/'
os.system('mkdir -p '+PREFIX)
FOOTPRINT = 'M_' + p.model + '_U' + p.unit + '_H' + str(HIDDEN_SIZE) + '_L' + str(LAYERS) + '.' + TIMESTAMP

### get training data
meta_data, dataset = convertData(TR,VAL,TEST,VECTOR)
tr,val,test = dataset

X_train,Y_train = tr
X_val,Y_val = val
X_test,Y_test = test

N = X_train.shape[0]
MAXLEN = X_train.shape[1]
DIM = X_train.shape[2]
ctable = CharacterTable(range(MAXLEN),-1)

print('building model...')

model = grab_model(p.model,p.unit, HIDDEN_SIZE, LAYERS, DIM, MAXLEN)
optimizer = RMSprop(clipnorm = 5)
print('compiling model...')
model.compile(optimizer,{'output': 'categorical_crossentropy'})
print("# of parameters of the model :",model.get_n_params())
print('training model...')

pat = 0
train_history = {'loss' : [], 'val_loss' : [], 'val_acc' : []}
best_val_acc = float('-inf')
val_fname = PREFIX + FOOTPRINT + '.validation'
val_eval = PREFIX + FOOTPRINT + '.val_eval'

for iteration in xrange(p.n_epochs):
	print()
	print('-' * 50)
	print('iteration {}/{}'.format(iteration+1,p.n_epochs))

	epoch_history = model.fit({'input' : X_train, 'output' : Y_train}, batch_size=BATCH_SIZE, nb_epoch=1,validation_data = {'input' : X_val, 'output' : Y_val})

	for key in ['loss','val_loss']:
		train_history[key] += epoch_history.history[key]

	### predict arcs on validation data
	prediction = model.predict({'input' : X_val})

	val_file = open(val_fname,'w')
	for i in xrange(len(prediction['output'])):
		instance = ctable.decode(prediction['output'][i], calc_argmax=True)
		val_file.write(instance.encode("utf-8")+'\n')
	val_file.close()

	### evaluate prediction
	cmd = 'python eval.py %s %s %s' % (p.val, val_fname,val_eval)
	os.system(cmd)
	epoch_val_acc = float(open(val_eval).readlines()[0].split()[0])
	train_history['val_acc'].append(epoch_val_acc)

	print("best val acc {}, epoch val acc : {} there was no improvement in {} epochs".format(best_val_acc,epoch_val_acc,pat))

	if train_history['val_acc'][-1] < best_val_acc:
		pat += 1
	else:
		pat = 0
		best_val_acc = train_history['val_acc'][-1]
		model.save_weights(PREFIX + FOOTPRINT + '.model',overwrite = True)
	if pat == PATIENCE:
		break

	# Select 3 samples from the validation set at random so we can visualize errors
	for i in xrange(3):
		ind = np.random.randint(0, len(X_val))
		rowX, rowy = X_val[np.array([ind])], Y_val[np.array([ind])]
		preds = model.predict({'input' : rowX})

		correct = ctable.decode(rowy[0])
		guess = ctable.decode(preds['output'][0], calc_argmax = True)
		print('sample {} in validation'.format(ind))
		print('T:', correct)
		print('P:', guess)
		print('---')

### test
model.load_weights(PREFIX + FOOTPRINT + '.model')
prediction = model.predict({'input' : X_test})
outfile = open( PREFIX + FOOTPRINT + '.output','w')
print("DEBUG:",prediction['output'].shape)
for i in xrange(len(prediction['output'])):
	p = ctable.decode(prediction['output'][i], calc_argmax=True)
	outfile.write(p.encode("utf-8")+'\n')

outfile.close()
with open( PREFIX + FOOTPRINT + '.arch', 'w') as outfile:
	json.dump(model.to_json(), outfile)
pickle.dump({'ctable' : ctable, 'train_history' : train_history},open(PREFIX + FOOTPRINT + '.meta', 'w'))
