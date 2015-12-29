import argparse

def get_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch-size , default 64',type=int,default = 64)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 500',type=int,default = 500)

	parser.add_argument('--patience', action='store', dest='patience',help='# of epochs for patience, default = 10',type=int,default = 10)

	parser.add_argument('--model', action='store', dest='model',help='model type {enc2dec, attention, pointer},default = enc2dec', default = 'enc2dec')

	parser.add_argument('--unit', action='store', dest='unit',help='train with {lstm gru rnn} units,default = lstm', default = 'lstm')

	parser.add_argument('--hidden', action='store', dest='n_hidden',help='hidden size of softmax layer, default = 128',type=int,default = 128)

	parser.add_argument('--layers', action='store', dest='n_layers',help='# of hidden layers, default = 1',type=int,default = 1)

	parser.add_argument('--train', action='store', dest='tr',help='tr file default ../data/TRAIN',default = '../data/TRAIN')

	parser.add_argument('--val', action='store', dest='val',help='val file default ../data/VAL',default = '../data/VAL')

	parser.add_argument('--test', action='store', dest='test',help='test file default ../data/TEST',default = '../data/TEST')

	parser.add_argument('--prefix', action='store', dest='prefix',help='exp log prefix to append exp/{} default = 0',default = '0')

	return parser
