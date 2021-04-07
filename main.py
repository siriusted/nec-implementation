import numpy as np
import torch
from nec_agent import NECAgent
from embedding_models import DQN


def run_training(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = config.env

    agent = NECAgent(env, config.agent_config)

    done = True
    for t in range(1, config.max_steps + 1):
        if done:
            obs, done = env.reset(), False
            agent.new_episode()

        action = agent.step(obs)
        next_obs, reward, done = env.step(action)
        agent.update((obs, action, reward, done))
        obs = next_obs

        if t >= config.start_learning_step:
            if t % config.replay_frequency == 0:
                agent.optimize()

            if t % config.eval_frequency == 0:
                agent.eval()
                # do the evaluation later #
                agent.train()


if __name__ == "__main__":
    run_training({})