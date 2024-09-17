import gym
import gym_env
import datetime
import os
import torch as th
import numpy as np
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.logger import configure
from utils.custom_policy_sb3 import CNN_FC, CNN_GAP, CNN_GAP_BN, No_CNN, CNN_MobileNet, CNN_GAP_new
from configparser import ConfigParser

from utils.training_policies_new import CustomCombinedExtractor

class AirSimRLTrainer:
    def __init__(self, method, policy, env_name, algo, action_num, purpose, 
                 noise_type='NA', goal_distance=70, noise_intensity=0.1, gamma=0.99, 
                 learning_rate=5e-4, total_steps=3e5):
        self.method = method
        self.policy = policy
        self.env_name = env_name
        self.algo = algo
        self.action_num = action_num
        self.purpose = purpose
        self.noise_type = noise_type
        self.goal_distance = goal_distance
        self.noise_intensity = noise_intensity
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.env = None
        self.model = None
        
        self.HOME_PATH = os.getcwd()
        print(f"Current working directory: {self.HOME_PATH}")
        
        self._init_folders()
        self._init_env()
        self._init_policy()
        self._init_model()

    def _init_folders(self):
        now = datetime.datetime.now()
        now_string = now.strftime('%Y_%m_%d_%H_%M_')
        self.file_path = os.path.join(self.HOME_PATH, 'logs', f"{now_string}_{self.policy}_{self.purpose}")
        self.log_path = os.path.join(self.file_path, 'log')
        self.model_path = os.path.join(self.file_path, 'models')
        self.config_path = os.path.join(self.file_path, 'config')
        
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.config_path, exist_ok=True)

    def _init_env(self):
        config_file = 'configs/config_Maze_SimpleMultirotor_2D.ini'
        cfg = ConfigParser()
        cfg.read(config_file)
        self.env = gym.make('airsim-env-v0')
        self.env.set_config(cfg)
        
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)

    def _init_policy(self):
        feature_num_state = self.env.dynamic_model.state_feature_length
        
        policy_dict = {
            'CNN_FC': (CNN_FC, 25),
            'CNN_GAP': (CNN_GAP_new, 16),
            'CNN_GAP_BN': (CNN_GAP_BN, 32),
            'No_CNN': (No_CNN, 25),
            'CNN_MobileNet': (MultiInputActorCriticPolicy, 25),
        }
        
        if self.policy in policy_dict:
            self.policy_used, self.feature_num_cnn = policy_dict[self.policy]
        else:
            raise ValueError('Invalid policy selection')


        self.policy_kwargs = dict(
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(),  # Custom extractor handles features
            activation_fn=th.nn.ReLU
        )


        # self.policy_kwargs = dict(
        #     features_extractor_class=self.policy_used,
        #     features_extractor_kwargs=dict(
        #         features_dim=feature_num_state + self.feature_num_cnn,
        #         state_feature_dim=feature_num_state),
        #     activation_fn=th.nn.ReLU
        # )
        
    def _init_model(self):
        if self.algo == 'ppo':
            # self.policy_kwargs['net_arch'] = [64, 32]
            self.model = PPO(self.policy_used, 
                             self.env,
                             n_steps=2048,
                             batch_size=64,
                             policy_kwargs=self.policy_kwargs,
                             tensorboard_log=f"{self.log_path}_{self.env_name}_{self.algo}",
                             seed=0, 
                             verbose=2)
        
        
        elif self.algo == 'td3':
            self.policy_kwargs['net_arch'] = [64, 32]
            n_actions = self.env.action_space.shape[-1]
            action_noise = None
            
            if self.noise_type == 'NA':
                action_noise = NormalActionNoise(
                    mean=np.zeros(n_actions), sigma=self.noise_intensity * np.ones(n_actions))
            elif self.noise_type == 'OU':
                action_noise = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions), sigma=self.noise_intensity * np.ones(n_actions), theta=5)
            else:
                raise ValueError("Invalid noise_type")

            self.model = TD3('CnnPolicy', self.env,
                             action_noise=action_noise,
                             learning_rate=self.learning_rate,
                             gamma=self.gamma,
                             policy_kwargs=self.policy_kwargs, verbose=1,
                             learning_starts=2000,
                             batch_size=128,
                             train_freq=(200, 'step'),
                             gradient_steps=200,
                             tensorboard_log=f"{self.log_path}_{self.env_name}_{self.algo}",
                             buffer_size=50000, seed=0)
        else:
            raise ValueError('Invalid algo input')

    def train(self):
        tb_log_name = f"{self.algo}_{self.policy}_{self.purpose}"
        self.model.learn(total_timesteps=self.total_steps,
                         log_interval=1,
                         tb_log_name=tb_log_name)

    def save_model(self):
        model_name = f"{self.method}_{self.algo}_{self.action_num}_{self.policy}_{self.purpose}"
        self.model.save(os.path.join(self.model_path, model_name))
        print(f'Model is saved to: {self.model_path}/{model_name}')
        print(f'Log is saved to: {self.log_path}_{self.env_name}_{self.algo}')

def main():
    trainer = AirSimRLTrainer(
        method='pure_rl',                       # 1-pure_rl 2-generate_expert_data 3-bc_rl 4-offline_rl
        policy='CNN_MobileNet',                 # 1-cnn_fc 2-cnn_gap 3-no_cnn 4-cnn_mobile
        env_name='airsim_city',                 # 1-trees  2-cylinder
        algo='ppo',                             # 1-ppo 2-td3
        action_num='2d',                        # 2d or 3d
        purpose='multicopter',                  # input your training purpose
        total_steps=1e5
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')