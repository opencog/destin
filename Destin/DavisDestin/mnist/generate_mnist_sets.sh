#bin/bash

url="http://yann.lecun.com/exdb/mnist"
files="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"
dir="data"

cd "$(dirname "$0")"

if [ ! -d $dir ]
then
    mkdir -p $dir
fi

for f in $files; do

    if [ -e $dir/$f ]
    then
        break
    fi

    wget -v -P $dir $url/$f.gz
    gunzip $dir/$f.gz

done

if [ ! -e $dir/train.bin ]
then
    echo "Generating 16x16 data ..."
    octave -q generate_mnist_set_16.m
fi

if [ ! -e $dir/destin_train_32.bin ]
then
    echo "Generating 32x32 data ..."
    octave -q generate_mnist_set_32.m
fi
