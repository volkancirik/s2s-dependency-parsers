## Sequence-to-Sequence Neural Networks based Dependency Parsing

Investigation of the uses of sequence-to-sequence neural networks (s2s) on dependency parsing. On-going experiments show *very* interesting results and a promising direction.

### Background

  If you don't know what dependency parsing is there is an excellent resource : [Dependency Parsing](http://www.morganclaypool.com/doi/pdf/10.2200/S00169ED1V01Y200901HLT002) from Kübler, McDonald, and Nivre.
  
  If you want to learn about s2s models you may want to read these : [Kalchbrenner and Blunsom](http://anthology.aclweb.org/D/D13/D13-1176.pdf), [Cho et. al.](http://arxiv.org/pdf/1409.1259.pdf), [Sutskever et. al.](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), and [Vinyals et. al.](http://arxiv.org/pdf/1506.03134v1.pdf)
  
#### 0 Converting Word Vectors

First you need to convert word embeddings into a pkl file. Under ``parser/`` folder. Let's use the sample word vector file under ``embeddings/``

    scripts/vector2pkl.py ../../embeddings/scode100.embeddings ../../embeddings/scode100.pkl 0 *UNKNOWN*
   
This will create a pickle file for word embeddings. It will not skip the first line (beware ``0``, some word vectors have meta-data in the first line, I dunno why). It will has a special tag for unknown words ``*UNKNOWN*``.

#### 1 Data Format

Please take a look at ``data/``. You will find samples of conll formatted files. Setup your data like that. I use the experimental setup of [this paper](http://arxiv.org/pdf/1505.08075.pdf).

#### 2 S2S Models and Keras

I implemented attention and pointer models on top of old Keras backend (befor TensorFlow changes). You need to checkout [my keras fork](https://github.com/wolet/keras) and put keras folder under ``parser/``.

#### 3 Training a Model

Under ``parser/`` if you type following command you will see a bunch of arguments.

      python train_parser.py
      
       --batch-size BATCH_SIZE batch-size , default 64
       --epochs N_EPOCHS     # of epochs, default = 500
       --patience PATIENCE   # of epochs for patience, default = 10
       --model MODEL         model type {enc2dec, attention, pointer},default = pointer
       --unit UNIT           train with {lstm gru rnn} units,default = lstm
       --hidden N_HIDDEN     hidden size of neural networks, default = 256
       --layers N_LAYERS     # of hidden layers, default = 1
       --train TR            training conll file
       --val VAL             validation conll file
       --test TEST           test conll file
       --prefix PREFIX       exp log prefix to append exp/{} default = 0
       --vector VECTOR       vector pkl file
       
Most of above are trivial. `--model` is to choose among s2s models. `enc2dec` is very similar to [Cho et. al.](http://arxiv.org/pdf/1409.1259.pdf). `attention` is for [Bahdanau et. al.](http://arxiv.org/pdf/1409.0473.pdf), and `pointer` is for [Vinyals et. al.](http://arxiv.org/pdf/1506.03134v1.pdf). You may want to take a look at the architecture through `get_model.py` and their backend implementation of [attention layer](https://github.com/wolet/keras/blob/master/keras/layers/attention.py).

`--prefix` is for creating a subfolder under `exp/`.

Let's say you type this command:

    python train_parser.py --model pointer --hidden 128 --layers 2 --train ../data/ptb.train.conll --val ../data/ptb.val.conll --test ../data/ptb.test.conll --vector ../embeddings/word2vec300.pkl --prefix PTB
    
 This will train a 2 layer 128 LSTM unit pointer parser model under `exp/PTB`. Under this folder you will expect to see `Mpointer_Vword2vec_Ulstm_H128_L2_<TIME>` (M is for model V is for word vector U is for rnn unit H is for width of model L is for depth of model <TIME> is the starting time of the experiment, lots of meta-data! ) with extensions:


`.arch`       : for architecture of the model

`.meta`       : for meta data about the setup and training

`.model`      : trained model

`.output`     : prediction of test file without the decoder (pure NN output)

`.decoded`    : prediction of test file using the model and a dependency decoder.

`.val_eval`   : the last training epoch's validation score

`.validation` : the last training epoch's validation prediction (pure NN)


To evaluate using conll's original script first convert the output to conll format using `script/output2conll.py` and use `scripts/conll07.pl` . Note that people generally use `-p` option to ignore punctuation.

#### 4 Problems and Future Work

  If you have any problems or ideas to improve this setup contact me.

#### TODO:
- write a better readme
- experiment -> latex table script
- argument option : ignore pos tags or not
