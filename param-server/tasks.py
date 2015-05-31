from celery import Celery
import time
from redis import Redis
import redis_collections as redisC
import evaluation
from datetime import datetime
import pickle


redisURL = "redis://localhost:6379/0"
redisInstance = Redis(host='localhost', port=6379, db=0)

app = Celery('tasks', broker=redisURL, backend=redisURL)

app.config_from_object('celeryconfig')



@app.task
def doSomething(arg):
	print "received argument", arg
	time.sleep(5)
	print "something is done"

@app.task
def saveSnapshot(snapshot_name, model):
	print "saving snapshot"
	snapshot = redisC.Dict(redis=redisInstance,key=snapshot_name)
	for key in model:
		snapshot[key] = model[key]
	print "[SNAPSHOT] Model snapshot saved:", snapshot_name

@app.task
def evaluateReward():
	evaluation.start(app.conf.ARCHITECTURE_FILE, app.conf.MODEL_FILE, recompute=False)

@app.task
def saveAverageReward():
	averageReward = redisC.Dict(redis=redisInstance, key='averageReward')
	filename = "averageReward" + datatime.now()
	with open(filename, 'wb') as f:
		pickle.dump(dict(averageReward), f)