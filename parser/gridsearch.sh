device=$1
model=$2

tr=../data/bansal100.train
val=../data/bansal100.val
test=../data/bansal100.test

for layer in 3 2 1
do
    for hidden in 256 128 64
    do
	echo "THEANO_FLAGS=device=$device python train_parser.py --train $tr --val $val --test $test --layers $layer --hidden $hidden --model $model --prefix grid_layer_hidden"
	THEANO_FLAGS=device=$device python train_parser.py --train $tr --val $val --test $test --layers $layer --hidden $hidden --model $model --prefix grid_layer_hidden
    done
done
