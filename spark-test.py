from pyspark import SparkContext

import barista
from barista.dummy_client import DummyClient


def sgd_step(step_num):
    dc = DummyClient("127.0.0.1", 50001)
    dc.send(barista.GRAD_UPDATE)
    return dc.recv()

sc = SparkContext()
steps = sc.parallelize(range(10))
res = steps.map(sgd_step)
print res.collect()
