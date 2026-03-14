# Image Preprocessing

# Importing the libraries
import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.spaces import Box


class PreprocessImage(gym.ObservationWrapper):
    """Preprocesses image observations: resize, grayscale, normalize."""

    def __init__(self, env, height=64, width=64, grayscale=True, crop=lambda img: img):
        super().__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop
        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(n_colors, height, width), dtype=np.float32
        )

    def observation(self, img):
        img = self.crop(img)
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img = np.array(pil_img)
        if self.grayscale:
            if img.ndim == 3:
                img = img.mean(-1, keepdims=True)
            else:
                img = img[:, :, np.newaxis]
        img = np.transpose(img, (2, 0, 1))
        img = img.astype("float32") / 255.0
        return img
