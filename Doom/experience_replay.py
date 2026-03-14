# Experience Replay

# Importing the libraries
import numpy as np
from collections import namedtuple, deque

# Defining one Step
Step = namedtuple("Step", ["state", "action", "reward", "done"])


class NStepProgress:
    """Makes the AI progress on several (n_step) steps."""

    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step

    def __iter__(self):
        state, _info = self.env.reset()
        history = deque()
        reward = 0.0
        while True:
            action = self.ai(np.array([state]))[0][0]
            next_state, r, terminated, truncated, _info = self.env.step(action)
            is_done = terminated or truncated
            reward += r
            history.append(
                Step(state=state, action=action, reward=r, done=is_done)
            )
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                state, _info = self.env.reset()
                history.clear()

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps


class ReplayMemory:
    """Implements experience replay buffer."""

    def __init__(self, n_steps, capacity=10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size):
        """Creates an iterator that returns random batches."""
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size : (ofs + 1) * batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter)
            self.buffer.append(entry)
            samples -= 1
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()
