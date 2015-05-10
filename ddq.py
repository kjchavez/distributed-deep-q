import os, sys
from pyspark import SparkContext, SparkConf

import caffe
import barista
from barista.dummy_client import DummyClient


def sgd_step(step_num):
    dc = DummyClient("127.0.0.1", 50001)
    dc.send(barista.GRAD_UPDATE)
    return dc.recv()

conf = SparkConf().setAppName("Spark Test")
sc = SparkContext(conf=conf)

# Start up Barista processes
script_path = "spark/spawn-barista.sh"
rdd = sc.parallelize([1]).pipe(script_path)
rdd.collect()

N = 20
steps = sc.parallelize(range(N))
res = steps.map(sgd_step)
print res.collect()
