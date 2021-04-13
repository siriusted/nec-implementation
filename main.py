import numpy as np
import torch
import tqdm
from nec_agent import NECAgent
from embedding_models import DQN


#TODO: store metrics somewhere
def run_training(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = config.env

    agent = NECAgent(env, config.agent_config)

    done = True
    epsilon = 1
    for t in tqdm(range(1, config.max_steps + 1)):
        if done:
            obs, done = env.reset(), False
            agent.new_episode()

        if 1 < t <= config.epsilon_anneal_end:
            epsilon -= (config.initial_epsilon - config.final_epsilon) / (config.epsilon_anneal_end - 1) # maybe change later to take full exploratory steps for a while at the beginning
            agent.set_epsilon(epsilon)

        action = agent.step(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update((obs, action, reward, done))
        obs = next_obs

        if t >= config.start_learning_step:
            if t % config.replay_frequency == 0:
                agent.optimize()

            if t % config.eval_frequency == 0:
                agent.eval()
                # evaluate agent here #
                agent.train()


if __name__ == "__main__":
    run_training({})
    #TODO: run an actual experiment with pong and plot the data