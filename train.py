import logging
import os
import random
from collections import deque
from dataclasses import dataclass

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class ReplayMemory:
    data: deque = deque([], maxlen=int(100000))


# create checkpoints directory
if not os.path.isdir("checkpoints"):
    os.mkdir("checkpoints")

# create logs directory
if not os.path.isdir("logs"):
    os.mkdir("logs")

# setup logger
logger = logging.getLogger("dqn_log")
file_handler = logging.FileHandler("logs/dqn_log.log", "w+")
stream_handler = logging.StreamHandler()
log_formatter = logging.Formatter("%(asctime)s - %(message)s")
file_handler.setFormatter(log_formatter)
stream_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def create_model(action_shape, state_shape):
    """
    create the DQN model

    :param action_shape: number of actions
    :param state_shape:
    :return:
    """
    model = nn.Sequential(
        nn.Linear(state_shape, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_shape)
    )

    return model


def train_policy_model(policy_model: nn.Module, target_model: nn.Module, memory: ReplayMemory,
                       optim: torch.optim.Optimizer, criterion: nn.Module, batch_size, delta):
    """
        Train the policy model

        :param policy_model: the model we attempt to train
        :param target_model: use for the training proccess
        :param memory: memory of experiences
        :param optim: the optimizer for hte model
        :param criterion: the loss of the model
        :param batch_size: the number of samples to use for training
        :param delta: discount factor
    """
    # get random batch of experiences
    if batch_size > len(memory.data):
        return

    batch = random.sample(memory.data, batch_size)

    # separate experiences
    states = torch.vstack([torch.tensor(exp[0]) for exp in batch])
    actions = torch.tensor([exp[1] for exp in batch])
    rewards = torch.tensor([exp[2] for exp in batch])
    next_states = torch.tensor([list(exp[3]) for exp in batch])
    dones = torch.tensor([exp[4] for exp in batch])

    # get policy net prediction
    policy_out = policy_model(states.float())

    # get target_net prediction
    target_out = target_model(next_states.float())

    # prepare labels
    labels = torch.clone(policy_out)
    for idx, (action, reward) in enumerate(zip(actions, rewards)):
        labels[idx][action] = reward
        if not dones[idx]:
            labels[idx][action] += delta * torch.max(target_out[idx])

    # calculate loss and do backpropagation
    optim.zero_grad()
    loss = criterion(policy_out, labels)
    loss.backward()
    optim.step()


def main():
    # parameters
    n_episodes = 1000
    train_per_steps = 4
    update_target_steps = 120
    exploration_prob = 1.0
    exploration_decay = 0.01
    batch_size = 64
    lr = 0.001
    delta = 0.99
    save_path = "checkpoints"
    n_show_episode = 100
    show_episodes = False

    # environment
    env = gym.make("LunarLander-v2")
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape[0]
    logger.info("n_actions: {}".format(n_actions))
    logger.info("n_states: {}".format(state_shape))

    # memory
    memory = ReplayMemory()

    # create models and define loss and optimizer
    policy_model = create_model(n_actions, state_shape)
    target_model = create_model(n_actions, state_shape)
    target_model.load_state_dict(policy_model.state_dict())
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(params=policy_model.parameters(), lr=lr)

    # play episodes
    steps = 0
    rewards = []
    for episode in tqdm(range(n_episodes)):
        observation = env.reset()
        done = False
        total_reward = 0

        while not done:
            if show_episodes and episode % n_show_episode == 0:
                env.render()

            # choose an action
            rand = random.random()
            if rand <= exploration_prob:
                action = env.action_space.sample()
            else:
                policy_model.eval()
                with torch.no_grad():
                    action = torch.argmax(policy_model(torch.from_numpy(observation).float())).item()
                policy_model.train()

            # execute the action
            next_observation, reward, done, info = env.step(action)
            total_reward += reward

            # save experience
            exp = (observation, action, reward, next_observation, done)
            memory.data.append(exp)

            # switch observation
            observation = next_observation

            # increment steps
            steps += 1

            # train policy model and update target model if the conditions are met
            if steps % train_per_steps == 0:
                train_policy_model(policy_model, target_model, memory, optim, criterion, batch_size, delta)
            if steps % update_target_steps == 0:
                target_model.load_state_dict(policy_model.state_dict())

        # decay exploration rate
        exploration_prob -= exploration_decay

        # save total reward for this episode
        rewards.append(total_reward)

        # save model if this episode was the best one yet
        if total_reward >= max(rewards):
            save_name = save_path + "/best.pt"
            logger.info("New Best Episode, Saving Model To '{}', Reward: {}".format(save_name, total_reward))
            torch.save({
                'epoch': episode,
                'model_state_dict': target_model.state_dict(),
            }, save_name)

    # save best episode
    best_episode = np.argmax(rewards)
    logger.info("---------------------------- END ----------------------------")
    logger.info("total episodes: {}".format(n_episodes))
    logger.info("best_episode: {} reward {}".format(best_episode, rewards[best_episode]))

    # plot reward over episodes
    x = np.linspace(0, n_episodes, n_episodes)
    plt.plot(x, rewards, "-r")
    plt.show()


if __name__ == '__main__':
    main()
