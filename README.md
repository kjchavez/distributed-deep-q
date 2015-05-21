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

Fire up the redis and parameter servers,

    redis-server
    python param-server/server.py models/deepq/solver.prototxt --reset

#### Submitting the job
We can now run the application using spark-submit. We will need to include the following python files/packages with the job submission:
- main.py
- replay.py
- expgain.py
- gamesim.zip
- barista.zip

And the following non-python files:
- models/deepq/train_val.prototxt
- models/deepq/deepq16.caffemodel
- models/deepq/solver.prototxt

You can create the zipped python packages using

    zip barista barista/*.py
    zip gamesim gamesim/*.py

Then submit the ddq.py script using spark-submit:

    ./spark-submit --master local[*] --files models/deepq/train_val.prototxt,models/deepq/solver.prototxt,models/deepq/deepq16.caffemodel --py-files barista.zip,caffe.zip,gamesim.zip,expgain.py,replay.py,main.py ddq.py 

#### Common Errors
- **socket.error: [Errno 99] Cannot assign requested address.** If there is a complaint about "localhost" in the message, check your /etc/hosts file and make sure the line "127.0.0.1 localhost" is present.
- **Output of spark-submit hangs.** Check logs/barista.log. If its the error: "socket.error: [Errno 98] Address already in use" then use:

        netstat -ap | grep 50001

    And see if any processes (pids will be listed as well) are listening on that port. If the status is LISTENING, try killing the process with

        kill -9 <pid>

    Then try spark-submitting again. If the status is TIME_WAIT, just wait a bit and call netstat again. 

## TODOs
### High Priority
- [BARISTA] Remove Barista's dependence on a .caffemodel argument in the initializer. It should be able to start directly from a solver.prototxt.
- [DDQ] Test a *single-process* version of the ddq application by spawning a Barista object inside the **train_partition** function, rather than using Popen.
- [PARAM-SERVER] Add functionality to periodically save a snapshot of the model.
- [PARAM-SERVER] Implement RMSProp or AdaGrad updates.
- [PARAM-SERVER] Decide when to send a new "target" model (known as P in the .prototxt)
- [EXP-GAIN] Add visualization of game frames and action selection.
- [UTILS] Implement script to evaluate the policy implied by a saved model. (i.e. Use model to play game many times and compute average score)

### Lower Priority
- [SPARK] Implement wrapper class(es) for easy use from within Spark.
- [AWS] Figure out how to run our pipeline on AWS. 

## Open Questions
- AWS? [https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7](https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7)
- Think about how we might use broadcast/accumulate Spark functions to simplify our parameter server
- Might the existence of [dataframes](https://databricks.com/blog/2015/02/17/introducing-dataframes-in-spark-for-large-scale-data-science.html) be helpful?
- [https://spark.apache.org/docs/latest/submitting-applications.html](https://spark.apache.org/docs/latest/submitting-applications.html)  
