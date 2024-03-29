import numpy as np
import os
import torch
from torch.optim import RMSprop
from torch.nn import MSELoss
from memory import ReplayBuffer
from nec import NEC
from logger import ScoreLogger

def _argmax(values):
    """
    return index with max value with random tie-breaking
    """
    idxs = np.nonzero(values == np.max(values))[0]
    return np.random.choice(idxs)

class NECAgent:
    """
    NEC agent
    """

    def __init__(self, config):
        self.nec_net = NEC(config).to(config['device'])
        self.train_eps = config['train_eps']
        self.eval_eps = config['eval_eps']
        self.num_actions = config['num_actions']
        self.replay_buffer = ReplayBuffer(config['observation_shape'], config['replay_buffer_size'])
        self.batch_size = config['batch_size']
        self.discount = config['discount']
        self.n_step_horizon = config['horizon']
        self.episode = 0
        self.logger = ScoreLogger(config['env_name'], config['exp_name'])
        self.env_name = config['env_name']
        self.exp_name = config['exp_name']
        self.device = config['device']
        self.train()

        # make sure model is on appropriate device at this point before constructing optimizer
        self.optimizer = RMSprop(self.nec_net.parameters(), lr = config['learning_rate'], alpha = config['rmsprop_alpha'], eps = config['rmsprop_epsilon'])
        self.loss_fn = MSELoss()


    def train(self):
        self.training = True
        self.nec_net.train()

    def eval(self):
        self.training = False
        self.nec_net.eval()

    def new_episode(self):
        # trackers for computing N-step returns and updating replay and dnd memories at the end of episode
        self.observations, self.keys, self.actions, self.values, self.rewards = [], [], [], [], []
        self.episode += 1

    def set_epsilon(self, eps):
        self.train_eps = eps

    def step(self, obs):
        q_values, key = self.nec_net.lookup(obs)

        eps = self.train_eps if self.training else self.eval_eps

        # do epsilon-greedy crap
        action = np.random.choice(np.arange(self.num_actions)) if np.random.rand() < eps else _argmax(q_values)

        # update trackers
        if self.training:
            self.actions.append(action)
            self.observations.append(obs)
            self.keys.append(key)
            self.values.append(np.max(q_values))

        return action


    def update(self, consequence):
        """
        Called from main training loop to inform agent of consequence of last action including
        reward and if the episode terminated
        """
        reward, done = consequence

        if self.env_name.startswith("CartPole"):
            reward = reward if not done else -reward

        # update reward tracker
        self.rewards.append(reward)

        if done:
            episode_length = len(self.actions)

            # compute N-step returns in reverse order
            returns, n_step_returns = [None] * (episode_length + 1), [None] * episode_length
            returns[episode_length] = 0

            for t in range(episode_length - 1, -1, -1):
                returns[t] = self.rewards[t] + self.discount * returns[t + 1]
                if episode_length - t > self.n_step_horizon:
                    n_step_returns[t] = returns[t] + self.discount ** self.n_step_horizon * (self.values[t + self.n_step_horizon] - returns[t + self.n_step_horizon])
                else: # use on-policy monte carlo returns when below horizon
                    n_step_returns[t] = returns[t]

            self.keys, n_step_returns = torch.stack(self.keys), np.array(n_step_returns, dtype = np.float32) # for fancy indexing

            # batch update of replay memory
            self.replay_buffer.append_batch(np.stack(self.observations), np.asarray(self.actions, dtype = np.int64), n_step_returns)

            # batch update of episodic memories
            unique_actions = np.unique(self.actions)
            for action in unique_actions:
                action_idxs = np.nonzero(self.actions == action)[0]
                self.nec_net.update_memory(action, self.keys[action_idxs], n_step_returns[action_idxs])

            # save/log metrics for plotting or whatever
            solved = self.logger.add_score(sum(self.rewards), self.episode)
            if solved:
                path = f'{os.getcwd()}/cartpole/trained_agents/nec_{self.exp_name}.pth'
                torch.save(self.nec_net.state_dict(), path)
                return True

        return False

    def optimize(self):
        """
        Here, we sample from the replay buffer and train the NEC model end-to-end with backprop
        """
        if self.replay_buffer.size() < self.batch_size:
            return

        observations, actions, returns = self.replay_buffer.sample(self.batch_size)
        self.optimizer.zero_grad()
        q_values = self.nec_net(observations.to(self.device))[range(self.batch_size), actions] # pick q_values for chosen actions
        loss = self.loss_fn(q_values, returns.to(self.device))
        loss.backward()
        self.optimizer.step()

    def get_q_values(self, observations, actions):
        """
        Computes q_values for observation, action pairs passed in.

        Used for testing
        """
        with torch.no_grad():
            self.eval()
            observations = torch.from_numpy(observations)
            q_values = self.nec_net(observations)[range(len(actions)), actions]

            return q_values.numpy()
