import numpy as np
import gym
import torch
from torch import nn
from tqdm import tqdm
from logging import ERROR
from nec_agent import NECAgent
from embedding_models import DQN, MLP
import matplotlib.pyplot as plt
import random

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger
# torch.autograd.set_detect_anomaly(True) # detect anomaly

def run_evaluation(config, path, episodes=50):
    env = config["env"]
    agent = NECAgent(config)
    agent.nec_net.load_state_dict(torch.load(path))
    agent.eval()

    rewards = []

    for ep in range(1, episodes + 1):
        obs, reward_sum = env.reset(), 0

        while True:
            env.render(mode='rgb-array')
            obs = torch.from_numpy(np.float32(obs))
            action = agent.step(obs)
            next_obs, reward, done, info = env.step(action)
            reward_sum += reward
            obs = next_obs

            if done:
                if config['env_name'].startswith('CartPole'):
                    reward_sum -= reward

                rewards.append(reward_sum)
                break

    plt.plot(range(1, episodes + 1), rewards)
    plt.savefig(f"eval_{config['exp_name']}.png")


def run_training(config, return_agent=False):
    env = config["env"]

    agent = NECAgent(config)

    done = True
    epsilon = 1
    for t in tqdm(range(1, config["max_steps"] + 1)):
        if done:
            obs, done = env.reset(), False
            agent.new_episode()

        if config["epsilon_anneal_start"] < t <= config["epsilon_anneal_end"]:
            epsilon -= (config["initial_epsilon"] - config["final_epsilon"]) / (config["epsilon_anneal_end"] - config["epsilon_anneal_start"])
            agent.set_epsilon(epsilon)

        # env.render(mode='rgb-array')
        if type(obs) is np.ndarray:
            obs = torch.from_numpy(np.float32(obs))
        action = agent.step(obs.to(config['device']))
        next_obs, reward, done, info = env.step(action)
        solved = agent.update((reward, done))

        if solved:
            return

        obs = next_obs

        if t >= config["start_learning_step"]:
            if t % config["replay_frequency"] == 0:
                agent.optimize()

            if t % config["eval_frequency"] == 0:
                agent.eval()
                # evaluate agent here #
                agent.train()

    if return_agent:
        return agent

if __name__ == "__main__":
    env_name = "CartPole-v1"#"Acrobot-v1"#"MountainCar-v0"
    env = gym.make(env_name)
    key_size = 4
    horizon = 50
    in_size = env.observation_space.shape[0]
    hidden = 16
    seed = 245
    exp_name = f"{env_name}_mlp{in_size}_{hidden}_{key_size}_capacity_10000"

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")


    config = {
        "env": env,
        "max_steps": 100_000,
        "initial_epsilon": 1,
        "final_epsilon": 0.001,
        "epsilon_anneal_start": 1,
        "epsilon_anneal_end": 2000,
        "start_learning_step": 1,
        "replay_frequency": 4,
        "eval_frequency": 100_000, # no eval for now
        "device": device,
        ###### NEC AGENT CONFIG #################
        "env_name": env_name,
        "exp_name": exp_name,
        "train_eps": 1, # initializing agent to be fully exploratory
        "eval_eps": 0,
        "num_actions": env.action_space.n,
        "observation_shape": env.observation_space.shape[0],
        "replay_buffer_size": 100000,
        "batch_size": 32,
        "discount": 1,
        "horizon": horizon,
        "learning_rate": 0.01,
        ###### NEC CONFIG #######################
        "embedding_net": MLP(in_size, hidden, key_size),
        ###### DND CONFIG #######################
        "dnd_capacity": 10000,
        "num_neighbours": 50,
        "key_size": key_size,
        "alpha": 0.9,
    }



    run_training(config)
    # run_evaluation(config, f'cartpole/trained_agents/nec_{exp_name}.pth')

    #TODO: run an experiment with pong and plot the data