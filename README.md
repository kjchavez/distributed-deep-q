# Distributed Deep Q Learning

## Getting Started
A slighted out-of-date version of Caffe is included as a submodule in this project, with a few modifications to the pycaffe interface. The easiest way to get up and running with Distributed Deep Q is to use this submodule.

### Using the submodule
If you cloned the project with the --recursive flag, all of the Caffe source files should be in the **caffe** sub-directory. Otherwise, it'll be empty and you should do the following:

    git submodule init; git submodule update;

Then follow the instructions at [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html) to build Caffe and pycaffe.

## Local Testing
### Without Spark
Start a Redis server instance.

    redis-server

Fire up the parameter server.

    python param-server/server.py models/deepq/solver.protoxt --reset

In a separate terminal, start the Barista application with

    python main.py models/deepq/train_val.prototxt models/deepq/deepq16.caffemodel [--debug]

The debug flag is optional; it will print some information about the parameter and gradient norms. Finally, you may simulate a request from a Spark executor by running the dummy client from the Barista package:

    python -m barista.dummy_client

### With Spark
#### Environment
Each worker machine must have Caffe installed (preferably in the same location) for Distributed Deep Q to work properly. Alternatively, if your workers are sufficiently homogenous, you can build the distribution version of Caffe on the driver, and send this to the workers when submitting the job.

In any case, be sure that each worker's PYTHONPATH includes the *caffe/python* directory. The driver should also have a proper Caffe installation.

Set a couple of environment variables:
    
    export DDQ_ROOT=<path-to-project>
    export PYTHONPATH=$PYTHONPATH:$DDQ_ROOT:$DDQ_ROOT/caffe/python

Fire up the redis and parameter servers,

    redis-server
    python param-server/server.py models/deepq/solver.protoxt --reset

#### Submitting the job
We can now run the application using spark-submit. We will need to include the following python files/packages with the job submission:
- main.py
- replay.py
- ExpGain.py
- gamesim.zip
- barista.zip

And the following non-python files:
- models/deepq/train_val.prototxt

You can create the zipped python packages using

    zip barista barista/*.py
    zip gamesim gamesim/*.py

Then submit the ddq.py script using spark-submit:

    spark-submit --master local --py-files <python files, comma-separated>, --files <regular files, comma-separated> ddq.py

Both stdout and stderr from the Barista server are redirected to a file in the logs directory.

#### Common Errors
- **socket.error: [Errno 99] Cannot assign requested address.** If there is a complaint about "localhost" in the message, check your /etc/hosts file and make sure the line "127.0.0.1 localhost" is present.
- **Output of spark-submit hangs.** Check logs/barista.log. If its the error: "socket.error: [Errno 98] Address already in use" then use:

        netstat -ap | grep 50001

    And see if any processes (pids will be listed as well) are listening on that port. If the status is LISTENING, try killing the process with

        kill -9 <pid>

    Then try spark-submitting again. If the status is TIME_WAIT, just wait a bit and call netstat again. 

## Open Questions
- Think about how we might use broadcast/accumulate Spark functions to simplify our parameter server
- Might the existence of [dataframes](https://databricks.com/blog/2015/02/17/introducing-dataframes-in-spark-for-large-scale-data-science.html) be helpful?
- [https://spark.apache.org/docs/latest/submitting-applications.html](https://spark.apache.org/docs/latest/submitting-applications.html)  
