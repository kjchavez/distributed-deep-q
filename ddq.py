import os, sys, time
from pyspark import SparkContext, SparkConf
from pyspark import SparkFiles

import caffe
import barista
from barista.dummy_client import DummyClient

import subprocess

def sgd_step(step_num):
    dc = DummyClient("127.0.0.1", 50001)
    dc.send(barista.GRAD_UPDATE)
    response = dc.recv()
    return response

def spawn_barista(partition):
    main = SparkFiles.get("main.py")
    architecture = SparkFiles.get("train_val.prototxt")
    model = SparkFiles.get("deepq16.caffemodel")
    solver = SparkFiles.get("solver.prototxt")
    root = SparkFiles.getRootDirectory()
    flag_file = "flags/__BARISTA_READY__"
    if os.path.isfile(flag_file):
        os.remove("flags/__BARISTA_READY__")

    out = open(os.path.join(root, "barista.log"),'w')
    subprocess.Popen(["python", main, architecture, model,
                      "--dataset", "dset.hdf5",
                      "--solver", solver],
                     stdout=out,
                     stderr=subprocess.STDOUT)

    while not os.path.isfile("flags/__BARISTA_READY__"):
        pass

def train_partition(idx, iterator):
    port = 50000 + idx % 256
    main = SparkFiles.get("main.py")
    architecture = SparkFiles.get("train_val.prototxt")
    model = SparkFiles.get("deepq16.caffemodel")
    solver = SparkFiles.get("solver.prototxt")
    root = SparkFiles.getRootDirectory()

    flag_file = "flags/__BARISTA_READY__.%d" % port
    if os.path.isfile(flag_file):
        os.remove(flag_file)

    out = open(os.path.join(root, "barista.log"),'w')
    subprocess.Popen(["python", main, architecture, model,
                      "--dataset", "dset.hdf5",
                      "--solver", solver,
                      "--port", str(port)])#,
                     #stdout=out,
                     #stderr=subprocess.STDOUT)

    while not os.path.isfile(flag_file):
        pass

    for step in iterator:
        dc = DummyClient("127.0.0.1", port)
        dc.send(barista.GRAD_UPDATE)
        response = dc.recv()
        yield response

conf = SparkConf().setAppName("Spark Test")
sc = SparkContext(conf=conf)

N = 20
steps = sc.parallelize(xrange(N))
# Start up Barista processes
# steps.foreachPartition(spawn_barista)
# res = steps.map(sgd_step)
res = steps.mapPartitionsWithIndex(train_partition)
print res.collect()
