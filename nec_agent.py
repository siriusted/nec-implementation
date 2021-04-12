import numpy as np

def _argmax(values):
    """
    return index with max value with random tie-breaking
    """
    idxs = np.non_zero(values == np.max(values))[0]
    return np.random.choice(idxs)

class NECAgent:
    """
    NEC agent
    """

    def __init__(self, env, config):
        self.nec_net = config['NEC']
        self.train_eps = 1 # initializing agent to be fully exploratory
        self.eval_eps = config['eval_eps']
        self.num_actions = env.action_space.n
        self.train()


    def train(self):
        self.training = True
        self.nec_net.train()

    def eval(self):
        self.training = False
        self.nec_net.eval()

    def new_episode(self):
        pass

    def set_epsilon(self, eps):
        self.train_eps = eps

    def step(self, state):
        q_values = self.nec_net.lookup(state)

        eps = self.train_eps if self.training else self.eval_eps

        # do epsilon-greedy crap
        if np.random.rand() < eps:
            return np.random.choice(np.arange(self.num_actions))

        return _argmax(q_values) # ensure q_values here is a list or numpy array

    def update(self, transition):
        obs, action, reward, done = transition

        # keep track of things

        if done:
            # batch update of replay memory
            # batch update of episodic memories
            pass

    def optimize(self):
        pass