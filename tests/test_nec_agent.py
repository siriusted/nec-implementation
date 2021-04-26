import gym
import numpy as np
import torch
from torch import nn
from main import run_training

class ENV:
    """
    2 step decision mdp for test purposes.

    s0  -a0-> s1 (r=0.0) -a0-> s3 (r=+2)
                         -a1-> s4 (r=+5)
        -a1-> s2 (r=0.0) -a0-> s5 (r=+3)
                         -a1-> s6 (r=+1)

    """

    def __init__(self):

        self._obs = np.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self._sigma = 0.01

        self._q_ground_truth_dnd_a = np.array([5.0, 2.0, 3.0], dtype=np.float32)

        self._q_ground_truth_dnd_b = np.array([3.0, 5.0, 1.0], dtype=np.float32)

        self._rewards = np.array([0.0, 0.0, 0.0, 2.0, 5.0, 3.0, 1.0], dtype=np.float32)


        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs.shape[1], ))
        self.cur_state = None

    def get_obs(self, state=None):
        if state is None:
            state = self.cur_state
        return self._obs[state] + np.random.randn(self._obs.shape[1]) * self._sigma

    def reset(self):
        self.cur_state = 0
        return self.get_obs()

    def step(self, action):
        assert action in [0, 1]

        if self.cur_state == 0:
            done = False
            next_state = 1 if action == 0 else 2
        elif self.cur_state == 1:
            done = True
            next_state = 3 if action == 0 else 4
        elif self.cur_state == 2:
            done = True
            next_state = 5 if action == 0 else 6

        self.cur_state = next_state
        reward = self._rewards[next_state]

        return self.get_obs(), reward, done, {}


def test_nec():
    print('inside test')
    env = ENV()
    key_size = 4

    seed = 1

    np.random.seed(seed)
    torch.manual_seed(seed)

    net = nn.Linear(env.observation_space.shape[0], key_size)


    config = {
        "env": env,
        "env_name": "test_env",
        "exp_name": "test",
        "seed": 1,
        "max_steps": 40,
        "initial_epsilon": 1,
        "final_epsilon": 0.1,
        "epsilon_anneal_start": 1,
        "epsilon_anneal_end": 2,
        "start_learning_step": 1,
        "replay_frequency": 1,
        "eval_frequency": 1000000, # no eval for now
        ###### NEC AGENT CONFIG #################
        "train_eps": 1, # initializing agent to be fully exploratory
        "eval_eps": 0,
        "num_actions": env.action_space.n,
        "observation_shape": env.observation_space.shape[0],
        "replay_buffer_size": 20,
        "batch_size": 3,
        "discount": 1,
        "horizon": 2,
        "learning_rate": 0.1,
        ###### NEC CONFIG #######################
        "embedding_net": net,
        ###### DND CONFIG #######################
        "dnd_capacity": 15,
        "num_neighbours": 1,
        "key_size": key_size,
        "alpha": 0.99,
    }


    # perform experiment
    agent = run_training(config, True)

    # check learned q_values for all states
    q_values = agent.get_q_values(env._obs[[0, 1, 2, 0, 1, 2]], [0, 0, 0, 1, 1, 1])
    expected_values = np.array([5.0, 2.0, 3.0, 3.0, 5.0, 1.0])

    assert np.allclose(q_values,  expected_values)
    print(q_values)
    # print(list(agent.nec_net.parameters()))


if __name__ == '__main__':
    test_nec()