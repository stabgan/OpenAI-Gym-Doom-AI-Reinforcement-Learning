# OpenAI Gym Doom AI Built with pyTorch
I implemented the agent which learns to play the classic Doom , using OpenAi Gym's Environment .


Using OpenAI Gym :

Getting Setup:
Follow the instruction on https://gym.openai.com/docs

```
git clone https://github.com/openai/gym
cd gym
pip install -e . # minimal install
```

Basic Example using CartPole-v0:

Level 1: Getting environment up and running
```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000): # run for 1000 steps
    env.render()
    action = env.action_space.sample() # pick a random action
    env.step(action) # take action
```

Level 2: Running trials(AKA episodes)
```
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset() # reset for each new trial
    for t in range(100): # run for 100 timesteps or until done, whichever is first
        env.render()
        action = env.action_space.sample() # select a random action (see https://github.com/openai/gym/wiki/CartPole-v0)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
```

Level 3: Non-random actions
```
import gym
env = gym.make('CartPole-v0')
highscore = 0
for i_episode in range(20): # run 20 episodes
  observation = env.reset()
  points = 0 # keep track of the reward each episode
  while True: # run until episode is done
    env.render()
    action = 1 if observation[2] > 0 else 0 # if angle if positive, move right. if angle is negative, move left
    observation, reward, done, info = env.step(action)
    points += reward
    if done:
      if points > highscore: # record high score
           highscore = points
      break
```

(Taken from the openAI gist)

We might want to decrease the size of our images because bigger images mean slower training and higher memory consumption. You might want to make the image grayscale because we might not need to know the difference between a green Kappa or a red Kappa. Another ‘pre-process’ we might take advantage of is utilizing a Convolutional Neural Network in order to extract/accentuate features from an image. Strictly speaking, passing your image through a “feed forward covnet” is not necessary, but doing so will improve results by reducing your image size and thus increasing training speed. 
After step 1 (image pre-processing), the next step involves the creation of our Q-Learning algorithm. Q-Learning algorithms come in all shapes and sizes, yet all variations function (or are based on) a basic principle. 

# The new Keras-Rl is a wonderful library which I will use in future projects 
