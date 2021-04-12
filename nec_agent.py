import numpy as np
from torch.optim import RMSprop
from torch.nn import MSELoss
from memory import ReplayBuffer

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
        self.replay_buffer = ReplayBuffer(config['replay_size'])
        self.batch_size = config['batch_size']
        self.train()

        # make sure model is on appropriate device at this point before constructing optimizer
        self.optimizer = RMSprop(self.nec_net.parameters(), lr = config['learning_rate'])
        self.loss_fn = MSELoss()


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

    def step(self, obs):
        q_values = self.nec_net.lookup(obs)

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
        """
        Here, we sample from the replay buffer and train the NEC model end-to-end with backprop
        """
        observations, actions, returns = self.replay_buffer.sample(self.batch_size)
        self.optimizer.zero_grad()
        q_values = self.nec_net(observations)[:, actions] # pick q_values for chosen actions
        loss = self.loss_fn(q_values, returns)
        loss.backward()
        self.optimizer.step()
