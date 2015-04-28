""" Implements class for reading / writing experiences to the "replay dataset."

We assume the following:

(1) Actions and rewards for the full history fit comfortably in memory.
(2) The game state representation for the full history does not.
(3) A single sample of states fits comfortable in memory

For example, if the replay dataset stores the last 1 million experiences,
then the history of actions is 1 byte x 1 M = 1 MB. The same holds for the
history of rewards. However, a modest game state representation might be
four frames of a 64 x 64 pixel grayscale image. In which case the full history
of 1 million states would be (64 x 64 x 4 x 1 bytes x 1 M = 17 GB)
"""
import random
import h5py
import numpy as np


class ReplayDataset(object):
    """ A wrapper around a replay dataset residing on disk as HDF5. """
    def __init__(self, filename, dset_size=1000, state_shape=(4, 128, 128),
                 overwrite=False):

        if overwrite:
            self.fp = h5py.File(filename, 'w')
        else:
            self.fp = h5py.File(filename, 'a')

        if all(x in self.fp for x in ("state", "action", "reward")):
            self.state = self.fp['state']
            self.dset_size = self.state.shape[0]

            self.action = np.empty(self.dset_size, dtype=np.uint8)
            self.fp['action'].read_direct(self.action)

            self.reward = np.empty(self.dset_size, dtype=np.int16)
            self.fp['reward'].read_direct(self.reward)

            if self.dset_size != dset_size:
                print ("Warning: dataset loaded from %s is of size %d, "
                       "not %d as requested. Using existing size."
                       % (filename, self.dset_size, dset_size))

        else:
            self.state = self.fp.create_dataset("state",
                                                (dset_size,) + state_shape,
                                                dtype='uint8')
            self.fp.create_dataset("action", (dset_size,), dtype='uint8')
            self.fp.create_dataset("reward", (dset_size,), dtype='int16')

            self.action = np.empty(dset_size, dtype=np.uint8)
            self.reward = np.empty(dset_size, dtype=np.int16)

            self.state.attrs['head'] = 0
            self.state.attrs['valid'] = 0
            self.dset_size = dset_size

        # Index of current 'write' location
        self.head = self.state.attrs["head"]

        # Greatest index of locations with valid (i.e. initialized)
        # experiences. Indices in the range [0, self.valid) are OK.
        self.valid = self.state.attrs["valid"]

    def add_experience(self, action, reward, state):
        """ Add the next step in a game sequence.

        Question: How do we deal with end of game boundary?
        """
        self.action[self.head] = action
        self.reward[self.head] = reward
        self.state[self.head] = state

        # Update head pointer and valid pointer
        self.head = (self.head + 1) % self.dset_size
        self.valid = min(self.dset_size, self.valid + 1)

    def sample(self, sample_size=32):
        """ Uniformly sample (s,a,r,s) experiences from the replay dataset.

        Args:
            sample_size: (self explanatory)

        Returns:
            A tuple of numpy arrays for the |sample_size| experiences.

                (state, action, reward, next_state)

            The zeroth dimension of each array corresponds to the experience
            index. The i_th experience is given by:

                (state[i], action[i], reward[i], next_state[i])
        """
        if sample_size >= self.valid:
            raise ValueError(
                  "Can't draw sample of size %d from replay dataset of size %d"
                  % (sample_size, self.valid))

        idx = random.sample(xrange(0, self.valid), sample_size)

        # We can't include head - 1 in sample because we don't know the next
        # state, so simply resample (very rare if dataset is large)
        while (self.head - 1) in idx:
            idx = random.sample(xrange(0, self.valid), sample_size)

        idx.sort()  # Slicing for hdf5 must be in increasing order
        next_idx = [x + 1 for x in idx]

        # next_state might wrap around end of dataset
        if next_idx[-1] == self.dset_size:
            shape = (sample_size,)+self.state[0].shape
            next_states = np.empty(shape, dtype=np.uint8)
            next_states[0:-1] = self.state[next_idx[0:-1]]
            next_states[-1] = self.state[0]
        else:
            next_states = self.state[next_idx]

        return (self.state[idx], self.action[idx],
                self.reward[idx], next_states)

    def __del__(self):
        self.fp['action'][:] = self.action
        self.fp['reward'][:] = self.reward
        self.state.attrs['head'] = self.head
        self.state.attrs['valid'] = self.valid

        self.fp.close()
