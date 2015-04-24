# Distributed Deep Q Learning

## Tips
Make sure caffe/python is on your PYTHONPATH, otherwise barista will be confused. 
    
    export PYTHONPATH=$PYTHONPATH:<path-to-caffe/python>


## Barista
To start the server, run

    python -m barista.server <architecture.prototxt> <model.caffemodel> <solver.prototxt>
