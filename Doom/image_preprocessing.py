# Image Preprocessing

# Importing the libraries
import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.spaces import Box

# Preprocessing the Images

class PreprocessImage(gym.ObservationWrapper):

    def __init__(self, env, height=64, width=64, grayscale=True, crop=lambda img: img):
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop
        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, shape=(n_colors, height, width), dtype=np.float32)

    def observation(self, img):
        img = self.crop(img)
        img = np.array(Image.fromarray(img).resize((self.img_size[1], self.img_size[0])))
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32') / 255.
        return img
