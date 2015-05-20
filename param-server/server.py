import argparse

from flask import Flask, Response, jsonify, request, render_template
from barista import messaging
from redis import Redis
import redis_collections as redisC
import pdb
import numpy as np

from caffe import SGDSolver

SGD_ALPHA = 0.01
app = Flask(__name__)


@app.route("/")
def hello():
    return "Param Server"
    
@app.route("/current_status")
def get_current_status():
    return render_template('current_status.html')

@app.route("/api/v1/status_data", methods=['GET'])
def get_current_data():
    d = {"a":"B", "c":"d"}
    return jsonify(d)

@app.route('/api/v1/latest_model', methods=['GET'])
def get_model_params():
    # assuming the return message would be a string(of bytes)
    model = redisC.Dict(key="centralModel")
    print "Qconv1 norm", np.linalg.norm(model['Qconv1'][0])
    m = messaging.create_message(dict(model), compress=False)
    # pdb.set_trace()
    return Response(m, status=200)

@app.route('/api/v1/update_model', methods=['POST'])
def update_params():
    updateParams = messaging.load_gradient_message(request.data, compressed = False)
    SGDUpdate(updateParams)
    return Response("Updated", status=200)

@app.route('/api/v1/clear_model', methods=['GET'])
def clear_params():
    model = redisC.Dict(key="centralModel")
    model.clear()
    return Response("Cleared", status=200)

def SGDUpdate(params):
    # get model stored in redis
    model = redisC.Dict(key="centralModel")
    # print model
    for k in params:
        if k not in model:
            print "Warning: parameter %s not found in model." % k
            model[k] = []
        for i in range(len(params[k])):
            if len(model[k]) < (i+1):
                arr = model[k]
                arr.append(np.zeros(params[k][i].shape,
                                    dtype=params[k][i].dtype))
                model[k] = arr
        # pdb.set_trace()
        model[k][i] -= SGD_ALPHA*params[k][i]

def initParams(solver_filename, reset=True):
    redisInstance = Redis(host='localhost', port=6379, db=0)
    model = redisC.Dict(redis=redisInstance, key="centralModel")

    if reset:
        model.clear()

        # Instantiate model parameters according to initialization
        # scheme specified in .prototxt file
        solver = SGDSolver(solver_filename,)
        for name in solver.net.params:
            parameters = solver.net.params[name]
            init = []
            for i in range(len(parameters)):
                init.append(np.array(parameters[i].data, dtype='float32'))
            model[name] = init

        print
        print "[Redis Collection]: Initialized the following parameters:"
        for key in model.keys():
            print "  - " + key + ' (%d parameters)' % len(model[key])

    else:
        solver = SGDSolver(solver_filename,)
        for name in solver.net.params:
            parameters = solver.net.params[name]
            assert(name in model.keys() and
                   len(model[name]) == len(parameters),
                   "Model in Redis database does not match specified solver.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("solver")
    parser.add_argument("--reset", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    initParams(args.solver, reset=args.reset)
    app.run(debug=True, port=5500)
