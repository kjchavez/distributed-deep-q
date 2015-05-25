from celery import Celery
import time
from redis import Redis
import redis_collections as redisC

redisURL = "redis://localhost:6379/0"
redisInstance = Redis(host='localhost', port=6379, db=0)

app = Celery('tasks', broker=redisURL, backend=redisURL)

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


