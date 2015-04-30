#!/bin/bash
PROJECT_ROOT=/home/kevin/CME323/project
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/caffe/python
cd $PROJECT_ROOT 
python -m barista $PROJECT_ROOT/models/deepq/train_val.prototxt $PROJECT_ROOT/models/deepq/deepq.caffemodel --dataset $PROJECT_ROOT/replay.hdf5 &> $PROJECT_ROOT/log.txt &
cd -

while read LINE; do
    echo OK
done
