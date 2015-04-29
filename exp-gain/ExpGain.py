import random

class ExpGain(object):
    def __init__(network, actions, preprocessor, game):
        self.network = network
        self.actions = actions
        self.preprocessor = preprocessor
        self.game = game

    def select_action(pstate, epsilon):
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            return network(pstate)

    def generate_experience(state, epsilon):
        pstate = self.preprocessor(state)
        action = select_action(pstate, epsilon)
        new_state, reward = self.game(state, action)
        return (pstate, action, reward, self.preprocessor(new_state))