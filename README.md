# Distributed Deep Q Learning

## Getting Started
A slighted out-of-date version of Caffe is included as a submodule in this project, with a few modifications to the pycaffe interface. The easiest way to get up and running with Distributed Deep Q is to use this submodule.

### Using the submodule
If you cloned the project with the --recursive flag, all of the Caffe source files should be in the **caffe** sub-directory. Otherwise, it'll be empty and you should do the following:

    git submodule init; git submodule update;

Then follow the instructions at [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html) to build Caffe and pycaffe.

### Installing Redis, and Celery
Redis is a nosql db and Celery is a task scheduling queue for executing jobs in different threads.

    pip install -U celery[redis]

### redis_collections
This module is used for storing dictionary objects in redis for Python

    pip install redis_collections

## Local Testing
### Without Spark
[In a separate termial]Start a Redis server instance.

    redis-server

[In a separate termial]Spawn Celery worker

    cd param_server
    celery -A tasks worker --loglevel=info

[In a separate termial]Fire up the parameter server.

    python param-server/server.py models/deepq/solver.prototxt --reset

[In a separate termial]Start the Barista application with

    python main.py models/deepq/train_val.prototxt models/deepq/deepq16.caffemodel [--debug]

The debug flag is optional; it will print some information about the parameter and gradient norms. Finally, you may simulate a request from a Spark executor by running the dummy client from the Barista package:

    python -m barista.dummy_client

### With Spark
#### Environment
Each worker machine must have Caffe installed (preferably in the same location) for Distributed Deep Q to work properly. Alternatively, if your workers are sufficiently homogenous, you can build the distribution version of Caffe on the driver, and send this to the workers when submitting the job.

In any case, be sure that each worker's PYTHONPATH includes the *caffe/python* directory. The driver should also have a proper Caffe installation.

Start Celery worker in a seperate terminal, 

    cd param_server
    celery -A tasks worker --loglevel=info

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

    ./spark-submit --master local[*] --files models/deepq/train_val.prototxt,models/deepq/solver.prototxt,models/deepq/deepq16.caffemodel --py-files barista.zip,gamesim.zip,expgain.py,replay.py,main.py ddq.py 
    
We could also add caffe.zip for --py-files in aws.  But in a local setting it's not needed.

#### Common Errors
- **socket.error: [Errno 99] Cannot assign requested address.** If there is a complaint about "localhost" in the message, check your /etc/hosts file and make sure the line "127.0.0.1 localhost" is present.
- **Output of spark-submit hangs.** Check logs/barista.log. If its the error: "socket.error: [Errno 98] Address already in use" then use:

        netstat -ap | grep 50001

    And see if any processes (pids will be listed as well) are listening on that port. If the status is LISTENING, try killing the process with

        kill -9 <pid>

    Then try spark-submitting again. If the status is TIME_WAIT, just wait a bit and call netstat again. 

## TODOs
### High Priority
- **[PARAM-SERVER] Optimize. Spending 240 milliseconds pickling data per gradient update. In that much time, we might as well compute a gradient on a single machine!**
- [AWS] Figure out how to run our pipeline on AWS.
- [PARAM-SERVER] Reject gradient updates if too stale. (Or, I have a better idea! It's like adagrad for asynchronous updates. Need to work out details rigorously, but basic idea is that you have an adaptive scaling that is inversely proportional to the deviation in the parameter value from when you used it to what it is now. Takes up extra O(K*D) space on the driver where K is the number of machines and D is the size of the model. Takes an extra O(D) time to compute an update. Proof of effectiveness seems like it would involve a Lipschitz constant.)
- ~~[CORRECTNESS] Deal with END-OF-GAME loss function. It's slightly different. See paper.~~
- ~~[PARAM-SERVER] Add functionality to periodically save a snapshot of the model.~~
- ~~[PARAM-SERVER] Decide when to send a new "target" model (known as P in the .prototxt)~~
- ~~[UTILS] Implement script to evaluate the policy implied by a saved model. (i.e. Use model to play game many times and compute average score)~~
- [SERVER] Fill in the TODO functions in server.py that compute variance of gradients and ratio of weights to updates for each parameter
- [MONITORING] Add display of the metrics from the previous point to the babysitting dashboard.
- [PARAM-SERVER] Profile, profile, profile.
- [BARISTA] Profile, profile, profile. The bottleneck should be the caffe computation.

### Lower Priority
- [SPARK] Implement wrapper class(es) for easy use from within Spark.
- [BARISTA] Switch to shared memory implementation.
- [BARISTA] Remove Barista's dependence on a .caffemodel argument in the initializer. It should be able to start directly from a solver.prototxt.
- [MONITORING] Visualize weights for first layer filters.
- [DDQ] Test a *single-process* version of the ddq application by spawning a Barista object inside the **train_partition** function, rather than using Popen. (Tried this, it broke)

## Open Questions
- AWS? [https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7](https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7)
- Think about how we might use broadcast/accumulate Spark functions to simplify our parameter server
- Might the existence of [dataframes](https://databricks.com/blog/2015/02/17/introducing-dataframes-in-spark-for-large-scale-data-science.html) be helpful?
- [https://spark.apache.org/docs/latest/submitting-applications.html](https://spark.apache.org/docs/latest/submitting-applications.html)  
