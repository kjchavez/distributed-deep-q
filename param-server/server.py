from flask import Flask, Response, jsonify, request
from barista import messaging
from redis import Redis
import redis_collections as redisC
import pdb
import numpy as np

SGD_ALPHA = 0.01
app = Flask(__name__)


@app.route("/")
def hello():
    return "Param Server"

@app.route('/api/v1/latest_model', methods=['GET'])
def get_model_params():
  # assuming the return message would be a string(of bytes)
  model = redisC.Dict(key="centralModel")
  
  m = messaging.create_message(dict(model), compress=False)
  # pdb.set_trace()
  return Response(m, status=200)

@app.route('/api/v1/update_model', methods=['POST'])
def update_params():
  updateParams = messaging.load_gradient_message(request.data, compressed = False)
  SGDUpdate(updateParams)
  return Response("Updated", status=200)

@app.route('/api/v1/clear_model', methods=['POST'])
def clear_params():
  model = redisC.Dict(key="centralModel")
  model.clear()
  return Response("Cleared", status=200)

def SGDUpdate(params):
  # get model stored in redis
  model = redisC.Dict(key="centralModel")
  for k in params:
    if k not in model:
      model[k] = []
    for i in range(len(params[k])):
      if len(model[k]) < (i+1):
        model[k].append(np.zeros(len(params[k][i])))
      model[k][i] -= SGD_ALPHA*params[k][i]
  return 

def initParams():
  redisInstance = Redis(host='localhost', port=6379, db=0)
  d = redisC.Dict(redis=redisInstance, key="centralModel")
  return  

if __name__ == "__main__":
    initParams()
    app.run(debug=True, port=5500)
