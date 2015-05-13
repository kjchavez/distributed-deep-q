from barista.baristanet import BaristaNet
import numpy as np

def test_ipc_interface(barista_net):
    semaphore, handles = barista_net.get_ipc_interface()
    print semaphore
    for handle in handles:
        print handle

def dummy_load_minibatch(barista_net):
    """ Writes random data into the numpy arrays.
    """
    state = barista_net.shared_arrays['state']
    state[...] = np.random.randint(0, 256, size=state.shape)

    # Actions matrix has a one-hot representation
    action = barista_net.shared_arrays['action']
    random_actions = np.random.randint(0, action.shape[1],
                                       size=(action.shape[0],))

    action[...] = np.zeros(action.shape)
    action[np.arange(action.shape[0]), random_actions] = 1

    reward = barista_net.shared_arrays['reward']
    reward[...] = np.random.randint(-5, 6, size=reward.shape)

    next_state = barista_net.shared_arrays['next_state']
    next_state[...] = np.random.randint(0, 256, size=next_state.shape)


def test_in_memory_config(barista_net):
    print "-" * 50
    print "Checking IN-MEMORY data layer configuration..."
    net = barista_net.net
    dummy_load_minibatch(barista_net)

    state = barista_net.shared_arrays['state']
    action = barista_net.shared_arrays['action']
    reward = barista_net.shared_arrays['reward']
    next_state = barista_net.shared_arrays['next_state']

    # Nothing should be loaded in data blobs before calling net forward
    assert(not np.all(net.blobs['state'].data == state))

    net.forward()
    assert(np.all(net.blobs['state'].data == state))
    assert(np.all(net.blobs['next_state'].data == next_state))
    assert(np.all(net.blobs['action'].data == action))
    assert(np.all(net.blobs['reward'].data == reward))

    # Should read from the IN-MEMORY location directly, no copy
    pointer_local, _ = state.__array_interface__['data']
    pointer_caffe, _ = net.blobs['state'].data.__array_interface__['data']
    print "state data mem address (local):", hex(pointer_local)
    print "state data mem address (caffe):", hex(pointer_caffe)
    assert(pointer_local == pointer_caffe)

    pointer_local, _ = next_state.__array_interface__['data']
    pointer_caffe, _ = net.blobs['next_state'].data.__array_interface__['data']
    print "next_state data mem address (local):", hex(pointer_local)
    print "next_state data mem address (caffe):", hex(pointer_caffe)
    assert(pointer_local == pointer_caffe)

    print "IN-MEMORY data layers correctly configured."


# def test_action_selection():
#     baristanet = BaristaNet('models/deepq/train_val.prototxt',
#                             'models/deepq/deepq.caffemodel',
#                             'Augustus')

#     for _ in xrange(10):
#         state = np.random.rand(4, 128, 128)
#         opt_action = baristanet.select_action(state)
#         print opt_action

def write_ipc_interface(barista_net, filename):
    comp_sem, model_sem, interface = barista_net.get_ipc_interface()
    with open(filename, 'w') as fp:
        print >> fp, comp_sem
        print >> fp, model_sem
        for name, shape, dtype in interface:
            print >> fp, ':'.join([name, str(shape), dtype])


def main(architecture, model):
    barista_net = BaristaNet(architecture, model, None)
    write_ipc_interface(barista_net, 'ipc.out')

    print "Barista running. Waiting on compute semaphore:",
    print barista_net.compute_semaphore
    for i in range(5):
        barista_net.full_pass()
        print "Completed full pass #%d" % i

    #test_in_memory_config(barista_net)
    #test_ipc_interface(barista_net)

if __name__ == "__main__":
    main("models/deepq/train_val.prototxt", "models/deepq/deepq16.caffemodel")
