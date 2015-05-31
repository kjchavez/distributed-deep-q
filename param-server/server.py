from flask import Flask, Response, jsonify, request, render_template
from barista import messaging
from redis import Redis
import redis_collections as redisC
import numpy as np
import argparse
from caffe import SGDSolver
from werkzeug.contrib.profiler import ProfilerMiddleware
import tasks
from threading import Lock

# Constants
MODEL_NAME = "centralModel"

# Global settings, only set once, when the server is started
redisInstance = None
snapshot_frequency = None
stats_frequency = None
special_update_period = 2
learning_rate = 0.0
rmsprop_decay = 0.9
update_fn = None
# central model
centralModel = {}
modelLock = Lock()
# rmsprop dictionary
rmsprop = {}
rmspropLock = Lock()

# adagrad dictionary
adagrad = {}
adagradLock = Lock()






app = Flask(__name__)

def get_snapshot_name(iteration):
    return MODEL_NAME + "-%06d" % iteration


def is_tracked_param(name):
    return name[0] == 'Q'


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
    iteration = int(redisInstance.get("iteration"))

    prev_model = dict(centralModel)

    with modelLock:
        if scale is None:
            for key in updates:
                centralModel[key] = [prev_model[key][i] - weight*updates[key][i]
                              for i in range(len(updates[key]))]
        else:
            for key in updates:
                centralModel[key] = [prev_model[key][i] - weight*updates[key][i]/fn(scale[key][i])
                              for i in range(len(updates[key]))]

        if iteration % snapshot_frequency == 0:
            print "sending snapshot params to task queue"
            snapshot_name = get_snapshot_name(iteration)
            tasks.saveSnapshot.delay(snapshot_name, centralModel)



def sgd_update(updateParams):
    print "[SGD UPDATE]"
    apply_descent(MODEL_NAME, updateParams, weight=learning_rate)


def rmsprop_update(updateParams):
    print "[RMSPROP UPDATE]"
    with rmspropLock:
        d_rmsprop = dict(rmsprop)
        if not rmsprop:
            for k in updateParams:
                params = []
                for i in range(len(updateParams[k])):
                    params.append(updateParams[k][i]**2)
                rmsprop[k] = params
                d_rmsprop = dict(rmsprop)
        else:
            for k in updateParams:
                rmsprop[k] = [rmsprop_decay * d_rmsprop[k][i] +
                              (1-rmsprop_decay) * updateParams[k][i]**2
                              for i in range(len(updateParams[k]))]

    apply_descent("centralModel", updateParams,
                  weight=learning_rate, scale=d_rmsprop,
                  fn=lambda x: np.sqrt(x+1e-8))


def adagrad_update(updateParams):
    print "[ADAGRAD UPDATE]"
    with adagradLock:
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
                  weight=learning_rate, scale=adagrad,
                  fn=lambda x: np.sqrt(x+1e-8))


def special_update_transform_model(model):
    """ For Distributed Deep Q, this involves copying the Q network to P.
    """
    updates = []
    for key in model:
        if key[0] == 'Q':
            pkey = 'P'+key[1:]
            updates.append((pkey, model[key]))

    model.update(updates)
    return model

# ============================================================================
#                 STATS for "babysitting" the learning process.
# ============================================================================
# ** See http://cs231n.github.io/neural-networks-3/#baby for more information.
# ** Should these statistics be saved to Redis or to file?

# TODO: Compute ratio of parameter weights to *updates* (not gradients,
# but the actual updates)
def compute_parameter_to_update_ratio(model, updateParams):
    """ Strongly suggest passing in a model that is a standard Python
    dict and not a redisC Dict. """
    raise NotImplementedError()


# TODO: Compute variance of gradients for each layer
# of the network
def compute_layer_variances(model, updateParams):
    """ Strongly suggest passing in a model that is a standard Python
    dict and not a redisC Dict. """
    raise NotImplementedError()

# TODO: Consider if it's worth it to compute variance of activations for
# each layer. This information is ONLY available on the worker machines,
# so it's significantly more trouble than these other metrics.


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
    iteration = int(redisInstance.get("iteration"))
    # print "Qconv1 norm", np.linalg.norm(model['Qconv1'][0])
    # TODO: Eliminate data duplication for improved efficiency

    with modelLock:
        if iteration % special_update_period == 0:
            special_update_transform_model(centralModel)

        print "Parameters sent:", ", ".join(centralModel.keys())
        m = messaging.create_message(centralModel, iteration, compress=False)
    return Response(m, status=200)


@app.route('/api/v1/update_model', methods=['POST'])
def update_params():
    updateParams = messaging.load_gradient_message(request.data)
    # print "Grad. Qconv1 norm", np.linalg.norm(updateParams['Qconv1'][0])
    redisInstance.incr("iteration")
    iteration = int(redisInstance.get("iteration"))

    if iteration % stats_frequency == 0:
        print "Monitoring stats not implemented."

    print "Iteration", iteration
    update_fn(updateParams)

    return Response("Updated", status=200)


@app.route('/api/v1/clear_model', methods=['GET'])
def clear_params():
    centralModel.clear()
    return Response("Cleared", status=200)


def initParams(solver_filename, reset=True):
    global redisInstance
    redisInstance = Redis(host='localhost', port=6379, db=0)
    averageReward = redisC.Dict(redis=redisInstance, key="averageReward")

    if reset:
        # Remove all previously saved snapshots from redis
        for name in redisInstance.keys(MODEL_NAME+"-*"):
            snapshot = redisC.Dict(redis=redisInstance, key=name)
            snapshot.clear()

        centralModel.clear()
        rmsprop.clear()
        adagrad.clear()
        averageReward.clear()
        redisInstance.set("iteration", 0)

        # Instantiate model parameters according to initialization
        # scheme specified in .prototxt file
        solver = SGDSolver(solver_filename,)
        for name in solver.net.params:
            if is_tracked_param(name):
                parameters = solver.net.params[name]
                init = []
                for i in range(len(parameters)):
                    init.append(np.array(parameters[i].data, dtype='float32'))
                centralModel[name] = init

        print
        print "[Redis Collection]: Initialized the following parameters:"
        for key in centralModel.keys():
            print "  - " + key + ' (%d parameters)' % len(centralModel[key])

    else:
        solver = SGDSolver(solver_filename,)
        for name in solver.net.params:
            parameters = solver.net.params[name]
            assert(name in centralModel.keys() and
                   len(centralModel[name]) == len(parameters) and
                   "Model in Redis database does not match specified solver.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("solver", help="Prototxt file defining solver")
    parser.add_argument("--reset", action="store_true",
                        help="Start training brand new model")
    parser.add_argument("--update", choices=['adagrad', 'rmsprop', 'sgd'],
                        default='rmsprop',
                        help="Choose type of gradient update")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate for updates")
    parser.add_argument("--rmsprop_decay", type=float, default=0.9,
                        help="Decay rate for moving average in RMSProp")
    parser.add_argument("--snapshot-freq", '-s', type=int, default=500,
                        help="Number of iterations between snapshots")
    parser.add_argument("--special-update", type=int, default=10,
                        help="Number of iterations between special updates")
    parser.add_argument("--stats-freq", type=int, default=500,
                        help="Record stats to monitor learning process every"
                        " this many iterations")
    parser.add_argument("--profile", action="store_true",
                        help="Print profiling stats per request")

    args = parser.parse_args()
    return args


def init_global_settings(settings):
    raise NotImplementedError("This still happens at global scope level")

if __name__ == "__main__":
    args = get_args()

    # Initialize global settings
    learning_rate = args.lr
    snapshot_frequency = args.snapshot_freq
    special_update_period = args.special_update
    stats_frequency = args.stats_freq
    if args.update == "sgd":
        update_fn = sgd_update
    elif args.update == "rmsprop":
        update_fn = rmsprop_update
        rmsprop_decay = args.rmsprop_decay
    elif args.update == "adagrad":
        update_fn = adagrad_update

    initParams(args.solver, reset=args.reset)

    if args.profile:
        app.config['PROFILE'] = True
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])

    app.run(debug=True, port=5500)
