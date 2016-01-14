import argparse

def get_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch-size , default 64',type=int,default = 64)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 500',type=int,default = 500)

	parser.add_argument('--patience', action='store', dest='patience',help='# of epochs for patience, default = 10',type=int,default = 10)

	parser.add_argument('--model', action='store', dest='model',help='model type {enc2dec, attention, pointer},default = pointer', default = 'pointer')

	parser.add_argument('--unit', action='store', dest='unit',help='train with {lstm gru rnn} units,default = lstm', default = 'lstm')

	parser.add_argument('--hidden', action='store', dest='n_hidden',help='hidden size of neural networks, default = 256',type=int,default = 256)

	parser.add_argument('--layers', action='store', dest='n_layers',help='# of hidden layers, default = 1',type=int,default = 1)

	parser.add_argument('--train', action='store', dest='tr',help='training conll file ',default = '../data/PTB_SD_3_3_0/train.conll')

	parser.add_argument('--val', action='store', dest='val',help='validation conll file',default = '../data/PTB_SD_3_3_0/dev.conll')

	parser.add_argument('--test', action='store', dest='test',help='test conll file',default = '../data/PTB_SD_3_3_0/test.conll')

	parser.add_argument('--prefix', action='store', dest='prefix',help='exp log prefix to append exp/{} default = 0',default = '0')

	parser.add_argument('--vector', action='store', dest='vector',help='vector pkl file', default = '../embeddings/sskip100.pkl')

	return parser


def get_tester():
	parser = argparse.ArgumentParser()

	parser.add_argument('--prefix', action='store', dest='prefix',help='prefix to model files')

	parser.add_argument('--test', action='store', dest='test',help='test conll file',default = '../data/PTB_SD_3_3_0/test.conll')

	parser.add_argument('--max-length', action='store', dest='max_length',help='max length of sequence',type=int)

	parser.add_argument('--vector', action='store', dest='vector',help='vector pkl file', default = '../embeddings/sskip100.pkl')

	parser.add_argument('--predict', action='store', dest='predict',help='prediction(output) file')

	return parser

def get_lengths(f_name):
	s = []
	lengths = []
	for line in open(f_name):
		l = line.strip().split('\t')
		if len(l) <= 1:
			lengths.append(len(s))
			s = []
			continue
		s.append(l)
	if s != []:
		lengths.append(len(s))
	return lengths

def convert2conll(prediction):

	out = []
	for i,pred in enumerate(prediction):
		if int(pred) == i:
			out.append('0')
		else:
			out.append(str(int(pred) +1))
	return out
