import gym
import torch as th
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from torch import nn
from torchvision import models
import torch.nn.functional as F
import numpy as np

# Custom Feature Extractor using MobileNetV3 for depth image and a DNN for vector inputs
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # Define the output dimensions of your feature extractor
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=512)  # Adjust the output dimension if needed

        # Assume observation_space contains:
        # - depth image: (224, 224, 3)
        # - velocity, orientation, and heading vectors

        # Load pretrained MobileNetV3 and modify for depth image (3 channels)
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        
        # Modify the classifier to output the desired feature size (256)
        mobilenet.classifier = nn.Sequential(
            nn.Linear(mobilenet.classifier[0].in_features, 128),
            nn.ReLU()
        )

        self.depth_image_extractor = mobilenet

        # Linear layers for velocity, orientation, and heading vectors
        self.vector_extractor = nn.Sequential(
            nn.Linear(9, 128),  # Adjust input size for velocity (2), orientation (4), and heading (3)
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # The final output of this extractor will be a concatenation of the image and vector features
        self._features_dim = 256  # Adjust this based on the concatenation output

    def forward(self, observations):
        # Process depth image using MobileNetV3
        depth_image = observations['depth_image']  # Assumes observation is a dict with 'depth_image'
        # depth_image = depth_image.float()   
        # depth_image = depth_image / 255     
        if depth_image.shape[1] == 1:  # If the channel dimension is 1, repeat it to make it 3 channels
            depth_image = depth_image.repeat(1, 3, 1, 1)  # Convert (batch_size, 1, height, width) to (batch_size, 3, height, width)
        
        depth_image_features = self.depth_image_extractor(depth_image)
        
        # Process vector inputs (velocity, orientation, heading)
        vector_input = th.cat([
            observations['velocity'],
            observations['orientation'],
            observations['heading']
        ], dim=1)
        # vector_input = vector_input.float()
        vector_features = self.vector_extractor(vector_input)
        
        # Concatenate the features from both extractors
        return th.cat([depth_image_features, vector_features], dim=1)

