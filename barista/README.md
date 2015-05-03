# Barista
A server built on top of a long-running Caffe process with a fixed architecture.

## Getting started
Make sure caffe/python is on your PYTHONPATH, otherwise barista will be confused. 
    
    export PYTHONPATH=$PYTHONPATH:<path-to-caffe/python>

If you already have an initialized model, you can start the server with

    python -m barista <architecture.prototxt> <model.caffemodel>

Otherwise, you'll need to specify a solver which can create a new model instance. For example:

    python -m barista models/deepq/train_val.prototxt models/deepq/deepq.caffemodel --solver models/deepq/solver.prototxt

This will create a new instance of the model architecture defined in *train_val.prototxt*, save it to *deepq.caffemodel* and fire up the barista server with this new model instance.
