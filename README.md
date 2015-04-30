# Distributed Deep Q Learning

## Getting Started
Caffe is included as a submodule in this project, with a few modifications to the pycaffe interface. The easiest way to get up and running with Distributed Deep Q is to use this submodule. If you already have Caffe installed, there's a quicker way, which we'll describe next.

### Using the submodule
There is a *caffe* sub-directory in the project root folder. If you cloned the project with the --recursive flag, all of the files should be there. Otherwise, it'll be empty and you should do the following:

    cd caffe; git submodule init; git submodule update;

Follow the instructions at [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html) to build Caffe and pycaffe.

## Barista
Make sure caffe/python is on your PYTHONPATH, otherwise barista will be confused. 
    
    export PYTHONPATH=$PYTHONPATH:<path-to-caffe/python>

If you already have an initialized model, you can start the server with

    python -m barista <architecture.prototxt> <model.caffemodel>

Otherwise, you'll need to specify a solver which can create a new model instance. For example:

    python -m barista models/deepq/train_val.prototxt models/deepq/deepq.caffemodel --solver models/deepq/solver.prototxt

This will create a new instance of the model architecture defined in *train_val.prototxt*, save it to *deepq.caffemodel* and fire up the barista server with this new model instance.

## Open Questions
- Think about how we might use broadcast/accumulate Spark functions to simplify our parameter server
- Might the existence of [dataframes](https://databricks.com/blog/2015/02/17/introducing-dataframes-in-spark-for-large-scale-data-science.html) be helpful?
  
