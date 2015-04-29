import os
import time
import random
import numpy as np

from dataset import ReplayDataset


def test_correctness():
    dset = ReplayDataset("test.hdf5", overwrite=True, dset_size=10)

    print "Initially..."
    print "------------------------"
    print "Head: ", dset.head
    print "Valid: ", dset.valid

    try:
        sample = dset.sample(1)
    except ValueError as e:
        print e.message
        print "Correctly triggered exception"

    # Add one experience to dataset, then try sampling again
    dset.add_experience(0, 10, np.random.randint(0, 256, size=(4, 128, 128)))
    print "Added one experience"
    try:
        sample = dset.sample(1)
        print "Correctly avoided exception when sampling"
    except ValueError as e:
        print e.message

    for _ in xrange(13):
        state = np.random.randint(0, 256, size=(4, 128, 128))
        action = random.choice(range(0, 10))
        reward = random.choice(range(-5, 6))
        dset.add_experience(action, reward, state)

    assert(dset.head == (13 + 1) % 10)
    assert(dset.valid == 10)

    print
    print "Current state of dataset:"
    print "----------------------------"
    print "Action:", dset.action
    print "Reward:", dset.reward
    print "State:", dset.state
    print "State[0,0,0]:", [dset.state[i, 0, 0, 0] for i in range(10)]
    print
    print "Draw full sample:"
    s, a, r, ns = dset.sample(9)
    print "S (0,0,0):", [s[i, 0, 0, 0] for i in range(9)]
    print "S'(0,0,0):", [ns[i, 0, 0, 0] for i in range(9)]


def test_timing(num_write=20000, num_samples=1000, sample_size=32):
    """ Speed test for replay dataset storage scheme. """
    import matplotlib.pyplot as plt
    if os.path.exists("test.hdf5"):
        os.remove("test.hdf5")

    if not os.path.isdir("evaluation"):
        os.makedirs("evaluation")

    results_file = open("evaluation/results.txt", 'w')

    # Create and fill dataset
    dset_size = 1e6
    dset = ReplayDataset("test.hdf5", dset_size=dset_size)

    # Time writing speed
    start = time.time()
    tocs = []
    for _ in xrange(num_write):
        state = np.ones((4, 128, 128), dtype=np.float32)
        dset.add_experience(0, 0, state)
        toc = time.time()
        tocs.append(toc)

    cum_time = np.array(tocs) - start
    plt.figure(1)
    plt.plot(cum_time)
    plt.title("Writing Performance")
    plt.xlabel("number of states written")
    plt.ylabel("cumulative run time (s)")
    plt.savefig("evaluation/write.png")

    print >> results_file, \
        "Time to write %d samples: %0.2f milliseconds" \
        % (num_write, 1000*(tocs[-1] - start))

    # Time sampling speed
    times = []
    for _ in xrange(num_samples):
        tic = time.time()
        sample = dset.sample(sample_size=sample_size)
        toc = time.time()
        times.append(1000 * (toc - tic))

    mean_sample_time = np.mean(times)
    std_sample_time = np.std(times)
    print >> results_file, \
        "Mean sample time: %0.3f milliseconds" % np.mean(times)

    print >> results_file, \
        "Std. dev. sample time: %0.3f milliseconds" % np.std(times)

    plt.figure(2)
    plt.hist(times, bins=np.linspace(0, mean_sample_time + std_sample_time),
             alpha=0.5, color="green")

    plt.title("Distribution of Sampling Times (Batch Size = %d)" %
              sample_size)
    plt.xlabel("sampling time (ms)")
    plt.ylabel("frequency")
    plt.savefig("evaluation/sample.png")

if __name__ == "__main__":
    test_correctness()
    test_timing()
