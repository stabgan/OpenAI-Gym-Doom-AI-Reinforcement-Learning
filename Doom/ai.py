# AI for Doom — Deep Convolutional Q-Learning

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Importing the packages for OpenAI Gymnasium and Doom
import gymnasium as gym

# Importing the other Python files
import experience_replay
import image_preprocessing


# Part 1 — Building the AI


class CNN(nn.Module):
    """Convolutional Neural Network brain for the agent."""

    def __init__(self, number_actions):
        super().__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(
            in_features=self._count_neurons((1, 80, 80)), out_features=40
        )
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)

    def _count_neurons(self, image_dim):
        with torch.no_grad():
            x = torch.rand(1, *image_dim)
            x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
            x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
            x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SoftmaxBody(nn.Module):
    """Softmax exploration policy with temperature scaling."""

    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=-1)
        actions = probs.multinomial(num_samples=1)
        return actions


class AI:
    """Combines brain (CNN) and body (SoftmaxBody) into an agent."""

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input_tensor = torch.from_numpy(np.array(inputs, dtype=np.float32))
        output = self.brain(input_tensor)
        actions = self.body(output)
        return actions.data.numpy()


# Part 2 — Training the AI with Deep Convolutional Q-Learning


def eligibility_trace(batch, cnn):
    """Compute n-step eligibility trace targets for a batch of episodes."""
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input_tensor = torch.from_numpy(
            np.array([series[0].state, series[-1].state], dtype=np.float32)
        )
        with torch.no_grad():
            output = cnn(input_tensor)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data.clone()
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


class MA:
    """Moving average tracker for rewards."""

    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        return np.mean(self.list_of_rewards)


def main():
    # ----------------------------------------------------------------
    # NOTE: The original code used ppaquette_gym_doom which is no longer
    # maintained. To run this, you need a Gymnasium-compatible Doom env
    # such as ViZDoom (vizdoom). The environment setup below is a
    # template — adjust the env ID to match your ViZDoom installation.
    #
    # Example with ViZDoom:
    #   pip install vizdoom
    #   env = gym.make("VizdoomCorridor-v0")
    # ----------------------------------------------------------------

    # Getting the Doom environment
    doom_env = gym.make("VizdoomCorridor-v0", render_mode="rgb_array")
    doom_env = image_preprocessing.PreprocessImage(
        doom_env, width=80, height=80, grayscale=True
    )
    doom_env = gym.wrappers.RecordVideo(doom_env, "videos")
    number_actions = doom_env.action_space.n

    # Building an AI
    cnn = CNN(number_actions)
    softmax_body = SoftmaxBody(T=1.0)
    ai = AI(brain=cnn, body=softmax_body)

    # Setting up Experience Replay
    n_steps = experience_replay.NStepProgress(env=doom_env, ai=ai, n_step=10)
    memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000)

    # Training the AI
    ma = MA(100)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    nb_epochs = 100

    for epoch in range(1, nb_epochs + 1):
        memory.run_steps(200)
        for batch in memory.sample_batch(128):
            inputs, targets = eligibility_trace(batch, cnn)
            predictions = cnn(inputs)
            loss_error = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss_error.backward()
            optimizer.step()
        rewards_steps = n_steps.rewards_steps()
        ma.add(rewards_steps)
        avg_reward = ma.average()
        print(f"Epoch: {epoch}, Average Reward: {avg_reward:.2f}")
        if avg_reward >= 1500:
            print("Congratulations, your AI wins!")
            break

    # Closing the Doom environment
    doom_env.close()


if __name__ == "__main__":
    main()
