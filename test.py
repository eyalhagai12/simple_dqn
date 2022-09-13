import gym
import torch

from train import create_model


def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    model = create_model(env.action_space.n, env.observation_space.shape[0])
    checkpoint = torch.load("checkpoints/best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    n_episodes = 20

    for episode in range(n_episodes):
        done = False
        observation, info = env.reset()
        episode_reward = 0.0

        while not done:
            env.render()
            with torch.no_grad():
                action = torch.argmax(model(torch.from_numpy(observation).float())).item()

            observation, reward, done, trunc, _ = env.step(action)
            episode_reward += reward

        print("Episode - {}, Reward - {}".format(episode, episode_reward))


if __name__ == '__main__':
    main()
