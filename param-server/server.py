import argparse

from flask import Flask, Response, jsonify, request
from barista import messaging
from redis import Redis
import redis_collections as redisC
import pdb
import numpy as np

from caffe import SGDSolver

SGD_ALPHA = 0.01
app = Flask(__name__)


def add_to_model(model, updates, weight, scale=None):
    """ Applies an update to the model parameters.

    Args:
        model:  dict-like container of all parameters
        updates: dict-like container of updates (must be subset of model)
        weight:  int or float by which the update is multiplied
        scale:   dict-like container of normalization factors by which to scale
                 individual elements of the updates (a la Adagrad or RMSProp)
    """
    if scale is None:
        for key in updates:
            for i in range(len(updates[key])):
                model[key][i] += weight*updates[key][i]
    else:
        for key in updates:
            for i in range(len(updates[key])):
                model[key][i] += weight*updates[key][i]/scale[key][i]

@app.route("/")
def hello():
    return "Param Server"


@app.route('/api/v1/latest_model', methods=['GET'])
def get_model_params():
    # assuming the return message would be a string(of bytes)
    model = redisC.Dict(key="centralModel")
    print "Qconv1 norm", np.linalg.norm(model['Qconv1'][0])
    m = messaging.create_message(dict(model), compress=False)
    # pdb.set_trace()
    return Response(m, status=200)


def rmsprop(updateParams):
    rmsprop = redisC.Dict(key="rmsprop")
    if not rmsprop:
        rmsprop = updateParams
    else:
        for k in updateParams:
            rmsprop[k] = 0.9 * rmsprop[k] + 0.1 * updateParams[k]**2


def adagrad(updateParams):
    adagrad = redisC.Dict(key="adagrad")
    if not adagrad:
        for k in updateParams:
            params = []
            for i in range(len(updateParams[k])):
                params.append(updateParams[k][i]**2)
            adagrad[k] = params
    else:
        for k in updateParams:
            for i in range(len(updateParams[k])):
                adagrad[k][i] += updateParams[k][i]**2


@app.route('/api/v1/update_model', methods=['POST'])
def update_params():
    updateParams = messaging.load_gradient_message(request.data, compressed = False)

    # rmsprop(updateParams)
    adagrad(updateParams)

    SGDUpdate(updateParams)
    return Response("Updated", status=200)


@app.route('/api/v1/clear_model', methods=['GET'])
def clear_params():
    model = redisC.Dict(key="centralModel")
    model.clear()
    return Response("Cleared", status=200)


def SGDUpdate(params):
    # get model and rms_meansq stored in redis
    model = redisC.Dict(key="centralModel")
    # rmsprop = redisC.Dict(key="rmsprop")
    adagrad = redisC.Dict(key="adagrad")
    print adagrad.keys()
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

            # update model with rmsprop
            # model[k][i] -= SGD_ALPHA * params[k][i] / np.sqrt(rmsprop[k][i])

            # update model with adagrad
            model[k][i] -= SGD_ALPHA * params[k][i] / np.sqrt(adagrad[k][i])


def initParams(solver_filename, reset=True):
    redisInstance = Redis(host='localhost', port=6379, db=0)
    model = redisC.Dict(redis=redisInstance, key="centralModel")
    rmsprop = redisC.Dict(redis=redisInstance, key="rmsprop")
    adagrad = redisC.Dict(redis=redisInstance, key="adagrad")

    if reset:
        model.clear()
        rmsprop.clear()
        adagrad.clear()

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
