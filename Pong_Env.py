import gym
import numpy as np
from PIL import Image
from skimage import util
import os
import time
import matplotlib.pyplot as plt
import cv2


class Pong_flip():
    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.action_space = self.env.action_space

    def reset(self):
        observation = self.env.reset()
        observation = observation[::-1]
        return observation

    def step(self,action):
        if action ==2 or action==4:
            action =3
        elif action == 3 or action==5:
            action =4
        observation,reward,done,_ = self.env.step(action)
        observation = observation[::-1]
        return observation,reward,done,_

    def render(self):
        self.env.render()


class Pong_noisy():
    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.action_space = self.env.action_space

    def reset(self):
        observation = self.env.reset()
        #add fixed gaussion white noise
        observation = util.random_noise(observation, mode='gaussian', seed=1)
        observation = (observation * 255).astype('int')
        return observation

    def step(self,action):
        observation,reward,done,_ = self.env.step(action)
        observation = util.random_noise(observation, mode='gaussian', seed=1)
        observation = (observation * 255).astype('int')
        return observation,reward,done,_

    def render(self):
        self.env.render()

class Pong_zoom():
    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.action_space = self.env.action_space

    def reset(self):
        observation = self.env.reset()
        observation = cv2_clipped_zoom(observation, 0.8)
        return observation

    def step(self,action):
        observation,reward,done,_ = self.env.step(action)
        observation = cv2_clipped_zoom(observation, 0.8)
        return observation,reward,done,_

    def render(self):
        self.env.render()


def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')#,constant_values=105)
    assert result.shape[0] == height and result.shape[1] == width
    return result



