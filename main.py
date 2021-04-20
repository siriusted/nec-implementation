import numpy as np
import gym
import torch
from tqdm import tqdm
from nec_agent import NECAgent
from embedding_models import DQN, MLP


#TODO: store metrics somewhere (done in nec agent)
def run_training(config):
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"]) #torch.manual_seed(np.random.randint(1, 10000))
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

        env.render(mode='rgb-array')
        action = agent.step(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update((reward, done))
        obs = next_obs

        if t >= config["start_learning_step"]:
            if t % config["replay_frequency"] == 0:
                agent.optimize()

            if t % config["eval_frequency"] == 0:
                agent.eval()
                # evaluate agent here #
                agent.train()


if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    config = {
        "env": env,
        "seed": 245,
        "max_steps": 1000000,
        "intial_epsilon": 1,
        "final_epsilon": 0.001,
        "epsilon_anneal_start": 1,
        "epsilon_anneal_end": 1000,
        "start_learning_step": 50,
        "replay_frequency": 16,
        "eval_frequency": 1000000, # no eval for now
        ###### NEC AGENT CONFIG #################
        "env_name": env_name,
        "exp_name": "key_8",
        "train_eps": 1, # initializing agent to be fully exploratory
        "eval_eps": 0,
        "num_actions": env.action_space.n,
        "replay_buffer_size": 100000,
        "batch_size": 32,
        "discount": 0.99,
        "horizon": 100,
        "learning_rate": 5e-5,
        ###### NEC CONFIG #######################
        "embedding_net": MLP(8),
        ###### DND CONFIG #######################
        "dnd_capacity": 500000,
        "num_neighbours": 50,
        "key_size": 8,
        "alpha": 0.1
    }



    run_training(config)

    #TODO: run an experiment with pong and plot the data