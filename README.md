# Distributed Deep Q Learning

## Getting Started
Caffe is included as a submodule in this project, with a few modifications to the pycaffe interface. The easiest way to get up and running with Distributed Deep Q is to use this submodule. If you already have Caffe installed, there's a quicker way, which we'll describe next.

### Using the submodule
There is a *caffe* sub-directory in the project root folder. If you cloned the project with the --recursive flag, all of the files should be there. Otherwise, it'll be empty and you should do the following:

    cd caffe; git submodule init; git submodule update;

Follow the instructions at [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html) to build Caffe and pycaffe.

## Local Testing

### Parameter server on driver
Set a couple of environment variables:
    
    export DDQ_ROOT=<path-to-project>
    export PYTHONPATH=$PYTHONPATH:$DDQ_ROOT:$DDQ_ROOT/caffe/python

Go to the project's root directory

    cd $DDQ_ROOT

and fire up the parameter server

    python param-server/server.py

### Submitting the job
Move to a separate terminal, also in the project root directory. If you haven't already done so, create a zip file of the barista package:

    zip -r barista barista

Then submit the ddq.py script using spark-submit:

    spark-submit --master local --py-files barista.zip,replay.py ddq.py

Both stdout and stderr from the Barista server are redirected to a file in the logs directory.

### Common Errors
- **socket.error: [Errno 99] Cannot assign requested address.** If there is a complaint about "localhost" in the message, check your /etc/hosts file and make sure the line "127.0.0.1 localhost" is present.
- **Output of spark-submit hangs.** Check logs/barista.log. If its the error: "socket.error: [Errno 98] Address already in use" then use:

        netstat -ap | grep 50001

    And see if any processes (pids will be listed as well) are listening on that port. If the status is LISTENING, try killing the process with

        kill -9 <pid>

    Then try spark-submitting again.

## Open Questions
- Think about how we might use broadcast/accumulate Spark functions to simplify our parameter server
- Might the existence of [dataframes](https://databricks.com/blog/2015/02/17/introducing-dataframes-in-spark-for-large-scale-data-science.html) be helpful?
- [https://spark.apache.org/docs/latest/submitting-applications.html](https://spark.apache.org/docs/latest/submitting-applications.html)  
