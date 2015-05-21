from flask import Flask, Response, jsonify, request, render_template
from barista import messaging
from redis import Redis
import redis_collections as redisC
import numpy as np
import argparse
from caffe import SGDSolver

# Constants
MODEL_NAME = "centralModel"

# Global settings, only set once, when the server is started
redisInstance = None
snapshot_frequency = 2
step_size = 0.0
rmsprop_decay = 0.9
update_fn = None

app = Flask(__name__)


def get_snapshot_name(iteration):
    return MODEL_NAME + "-%06d" % iteration


def apply_descent(model_name, updates, weight=1, scale=None, fn=lambda x: x):
    """ Applies an update to the model parameters.

    Args:
        model_name: key for the model in the redis server
        updates: dict-like container of updates (must be subset of model)
        weight:  int or float by which the update is multiplied
        scale:   dict-like container of normalization factors by which to scale
                 individual elements of the updates (a la Adagrad or RMSProp)
        fn:     function to apply to scale
    """
    redisInstance.incr("iteration")
    iteration = int(redisInstance.get("iteration"))

    prev_model = dict(redisC.Dict(key=model_name))
    model = redisC.Dict(key=model_name)

    if scale is None:
        for key in updates:
            model[key] = [prev_model[key][i] - weight*updates[key][i]
                          for i in range(len(updates[key]))]
    else:
        for key in updates:
            model[key] = [prev_model[key][i] - weight*updates[key][i]/fn(scale[key][i])
                          for i in range(len(updates[key]))]

    if iteration % snapshot_frequency == 0:
        snapshot_name = get_snapshot_name(iteration)
        snapshot = redisC.Dict(key=snapshot_name)
        for key in model:
            snapshot[key] = model[key]
        print "[SNAPSHOT] Model snapshot saved:", snapshot_name


def sgd_update(updateParams):
    print "[SGD UPDATE]"
    apply_descent(MODEL_NAME, updateParams, weight=step_size)


def rmsprop_update(updateParams):
    print "[RMSPROP UPDATE]"
    rmsprop = redisC.Dict(key="rmsprop")
    if not rmsprop:
        for k in updateParams:
            params = []
            for i in range(len(updateParams[k])):
                params.append(updateParams[k][i]**2)
            rmsprop[k] = params
    else:
        for k in updateParams:
            rmsprop[k] = [rmsprop_decay * rmsprop[k][i] +
                          (1-rmsprop_decay) * updateParams[k][i]**2
                          for i in range(len(updateParams[k]))]

    apply_descent("centralModel", updateParams,
                  weight=step_size, scale=rmsprop,
                  fn=lambda x: np.sqrt(x+1e-8))


def adagrad_update(updateParams):
    print "[ADAGRAD UPDATE]"
    adagrad = redisC.Dict(key="adagrad")
    if not adagrad:
        for k in updateParams:
            params = []
            for i in range(len(updateParams[k])):
                params.append(updateParams[k][i]**2)
            adagrad[k] = params
    else:
        for k in updateParams:
            adagrad[k] = [adagrad[k][i] + updateParams[k][i]**2
                          for i in range(len(updateParams[k]))]

    apply_descent(MODEL_NAME, updateParams,
                  weight=step_size, scale=adagrad,
                  fn=lambda x: np.sqrt(x+1e-8))


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
    model = redisC.Dict(key=MODEL_NAME)
    print "Qconv1 norm", np.linalg.norm(model['Qconv1'][0])
    m = messaging.create_message(dict(model), compress=False)
    # pdb.set_trace()
    return Response(m, status=200)


@app.route('/api/v1/update_model', methods=['POST'])
def update_params():
    updateParams = messaging.load_gradient_message(request.data)
    print "Grad. Qconv1 norm", np.linalg.norm(updateParams['Qconv1'][0])
    update_fn(updateParams)
    print "Iteration:", redisInstance.get("iteration")
    return Response("Updated", status=200)


@app.route('/api/v1/clear_model', methods=['GET'])
def clear_params():
    model = redisC.Dict(key=MODEL_NAME)
    model.clear()
    return Response("Cleared", status=200)


def initParams(solver_filename, reset=True):
    global redisInstance
    redisInstance = Redis(host='localhost', port=6379, db=0)
    model = redisC.Dict(redis=redisInstance, key=MODEL_NAME)
    rmsprop = redisC.Dict(redis=redisInstance, key="rmsprop")
    adagrad = redisC.Dict(redis=redisInstance, key="adagrad")

    if reset:
        # Remove all previously saved snapshots from redis
        for name in redisInstance.keys(MODEL_NAME+"-*"):
            snapshot = redisC.Dict(redis=redisInstance, key=name)
            snapshot.clear()

        rmsprop.clear()
        adagrad.clear()
        redisInstance.set("iteration", 0)

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
                   len(model[name]) == len(parameters) and
                   "Model in Redis database does not match specified solver.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("solver")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--update", choices=['adagrad', 'rmsprop', 'sgd'],
                        default='rmsprop')
    parser.add_argument("--stepsize", type=float, default=1e-3)
    parser.add_argument("--rmsprop_decay", type=float, default=0.9)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Initialize global settings
    step_size = args.stepsize
    if args.update == "sgd":
        update_fn = sgd_update
    elif args.update == "rmsprop":
        update_fn = rmsprop_update
        rmsprop_decay = args.rmsprop_decay
    elif args.update == "adagrad":
        update_fn = adagrad_update

    initParams(args.solver, reset=args.reset)
    app.run(debug=True, port=5500)
