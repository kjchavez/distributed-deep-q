import os, sys
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

def spawn_barista(idx):
    main = SparkFiles.get("main.py")
    architecture = SparkFiles.get("train_val.prototxt")
    model = SparkFiles.get("deepq16.caffemodel")
    solver = SparkFiles.get("solver.prototxt")
    root = SparkFiles.getRootDirectory()
    if os.path.isfile("flags/__BARISTA_READY__"):
        os.remove("flags/__BARISTA_READY__")

    subprocess.Popen(["python", main, architecture, model,
                      "--dataset", "dset.hdf5",
                      "--solver", solver])

    while not os.path.isfile("flags/__BARISTA_READY__"):
        pass

    return "OK"

conf = SparkConf().setAppName("Spark Test")
sc = SparkContext(conf=conf)

# Start up Barista processes
# script_path = SparkFiles.get("spawn-barista.sh")
# print script_path
# print "Running DDQ from:", os.getcwd()
rdd = sc.parallelize([1]).map(spawn_barista)
rdd.collect()

N = 100
steps = sc.parallelize(range(N))
res = steps.map(sgd_step)
print res.collect()
