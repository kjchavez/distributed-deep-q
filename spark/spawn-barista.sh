#!/bin/bash
mkdir -p logs
export PYTHONPATH=$PYTHONPATH:/home/kevin/Code/caffe/python
pwd > logs/spawn.log
echo "PYTHONPATH" >> logs/spawn.log
echo $PYTHONPATH >> logs/spawn.log

mkdir -p flags
rm -f flags/__BARISTA_READY__
python main.py models/deepq/train_val.prototxt models/deepq/deepq16.caffemodel --dataset rdset.hdf5 --solver models/deepq/solver.prototxt &> logs/barista.log &
echo "Called successfully" >> logs/spawn.log

# Wait until server is ready
while [ ! -f flags/__BARISTA_READY__ ]
do
      sleep 1
done

while read LINE; do
    echo OK
done
