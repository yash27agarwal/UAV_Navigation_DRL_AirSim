import gym
from gym import spaces
from configparser import NoOptionError
import keyboard
import numpy as np
import math
import cv2
import yaml
from .dynamics.multirotor_airsim_new import MultirotorDynamicsAirsim


class AirSimEnv(gym.Env):

    def __init__(self) -> None:
        super().__init__()
        print("init airsim gym env")

    def set_config(self, yaml_file):

        # training state
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0

        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        
        # Assign the values to the class attributes
        self.env_name = config.get('env_name')
        self.dynamic_name = config.get('dynamic_name')
        self.navigation_3d = config.get('navigation_3d')
        self.using_velocity_state = config.get('using_velocity_state')
        self.perception = config.get('perception')
        self.keyboard_debug = config.get('keyboard_debug')
        self.generate_q_map = config.get('generate_q_map')
        self.q_map_save_steps = config.get('q_map_save_steps')

        # Access nested values in the 'environment' section
        environment = config.get('environment', {})
        self.max_depth_meters = environment.get('max_depth_meters')
        self.crash_distance = environment.get('crash_distance')

        # Access nested values in the 'multirotor' section
        multirotor = config.get('multirotor', {})
        self.acc_xy_max = multirotor.get('acc_xy_max')
        self.v_xy_max = multirotor.get('v_xy_max')
        self.v_xy_min = multirotor.get('v_xy_min')
        self.v_z_max = multirotor.get('v_z_max')
        self.yaw_rate_max_deg = multirotor.get('yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)

        self.max_vertical_difference = 5

        print('Environment: ', self.env_name, "Dynamics: ", self.dynamic_name,
              'Perception: ', self.perception_type)
        
        self.dynamic_model = MultirotorDynamicsAirsim(yaml_file)
        
        # definition for NH tree envirnoment
        self.workspace_boundry_x =  (100, 300)
        self.workspace_boundry_y = (180, 400)
        self.workspace_boundry_z = (0.5, 5)
        self.height = 3
        self.goal_distance = 50
        self.max_episode_steps = 400
        self.accept_radius = 2
        self.env_center = ((self.workspace_boundry_x[0] + self.workspace_boundry_x[1]) / 2,
                            (self.workspace_boundry_y[0] + self.workspace_boundry_y[1]) / 2)
        
        self.dynamic_model.set_boundary_conditions(self.workspace_boundry_x, 
                                                   self.workspace_boundry_y, 
                                                   self.workspace_boundry_z)
        
        self.dynamic_model.set_goal_parameters(self.goal_distance, self.env_center)
        self.dynamic_model.set_height(self.height)
        self.dynamic_model.set_constraints(self.acc_xy_max, self.v_xy_max, 
                                                 self.v_xy_min, self.v_z_max,
                                                 self.yaw_rate_max_rad)
        self.dynamic_model.set_navigation_mode(self.navigation_3d)
        self.dynamic_model.set_crash_radius(self.crash_distance)
        
        self.observation_space = spaces.Dict({
                "depth_image" : spaces.Box(low=0, high=255,
                                            shape=(64, 64, 3), dtype=np.uint8),
                "velocity" : spaces.Box(low=0, high=1, 
                                        shape=(2,), dtype=np.float32),
                "orientation" : spaces.Box(low=0, high=1,
                                           shape=(4,), dtype=np.float32),
                "heading" : spaces.Box(low = 0, high=1, shape=(3,),
                                       dtype=np.float32)
            })
        
        if self.navigation_3d:
            self.action_space = spaces.Box(low=np.array([self.v_xy_min, -self.v_z_max, -self.yaw_rate_max_rad]),
                                           high=np.array([self.v_xy_max, self.v_z_max, self.yaw_rate_max_rad]),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=np.array([self.v_xy_min, -self.yaw_rate_max_rad]),
                                           high=np.array([self.v_xy_max, self.yaw_rate_max_rad]),
                                           dtype=np.float32)


    def reset(self):
        self.dynamic_model.reset()
        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.trajectory_list = []
        obs = self.get_obs()
        self.ep_boundary_x, self.ep_boundary_y, self.ep_boundary_z = self.dynamic_model.update_episode_boundary() 

        return obs
    
    def step(self, action):
        self.dynamic_model.set_action(action)
        position_ue4 = self.dynamic_model.get_position()
        self.trajectory_list.append(position_ue4)
        obs = self.get_obs()

        done = self.is_done()
        info = {
            'is_success': self.is_in_desired_pose(),
            'is_crash': self.is_crashed(),
            'is_not_in_workspace': self.is_not_inside_workspace(),
            'step_num': self.step_num
        }
        if done:
            print(info)

        reward = self.compute_reward_final(done, action)
        self.print_train_info_airsim(action, obs, reward, info)

        if self.keyboard_debug:
            action_copy = np.copy(action)
            action_copy[-1] = math.degrees(action_copy[-1])
            state_copy = np.copy(self.dynamic_model.state_raw)

            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(
                '=============================================================================')
            print('episode', self.episode_num, 'step',
                  self.step_num, 'total step', self.total_step)
            print('action', action_copy)
            print('state', state_copy)
            print('state_norm', self.dynamic_model.state_norm)
            print('reward {:.3f} {:.3f}'.format(
                reward, self.cumulated_episode_reward))
            print('done', done)
            keyboard.wait('a')
        
        self.step_num += 1
        self.total_step += 1
        return obs, reward, done, info
    
    def get_obs(self):
        image = self.get_depth_image()  
        image_resize = cv2.resize(image, (224, 224))
        self.min_distance_to_obstacles = image.min()
        self.dynamic_model.min_distance_to_obstacles(self.min_distance_to_obstacles)

        # cv2.imshow('Original Depth Image', image_resize.astype(np.uint8))
        image_scaled = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters  * 255
        image_scaled = 255 - image_scaled  # Invert the depth (closer = brighter)
        image_uint8 = image_scaled.astype(np.uint8)
        rgb_image = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        rgb_image = cv2.resize(rgb_image, (64,  64))
        # cv2.imshow('Processed Depth Image', image_uint8)

        velocity_vector, orientation_vector, desired_heading = self.dynamic_model.get_state_feature()
        obs = {
            "depth_image" : rgb_image,
            "velocity" : velocity_vector,
            "orientation" : orientation_vector,
            "heading" : desired_heading
        }

        # cv2.waitKey(1)  
        return obs
    
    def compute_reward_final(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if self.is_in_desired_pose():
            return reward_reach
        if self.is_crashed():
            return reward_crash
        if self.is_not_inside_workspace():
            return reward_outside
        
        if not done:
            if self.env_name == 'NH_center':
                distance_reward_coef = 500
            else:
                distance_reward_coef = 50
            # 1 - goal reward
            distance_now = self.dynamic_model.get_distance_to_goal_3d()
            reward_distance = distance_reward_coef * (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance   # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            z = current_pose[2]
            x_g = goal_pose[0]
            y_g = goal_pose[1]
            z_g = goal_pose[2]

            punishment_xy = np.clip(self.getDis(
                x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip((z - z_g)/5, 0, 1)

            punishment_pose = punishment_xy + punishment_z

            if self.min_distance_to_obstacles < 10:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 5, 0, 1)
            else:
                punishment_obs = 0

            punishment_action = 0

            # add yaw_rate cost
            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            self.dynamic_model._get_state_feature_old()

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = ((abs(action[1]) / self.dynamic_model.v_z_max)**2)
                z_err_cost = (
                    (abs(self.dynamic_model.state_raw[1]) / self.max_vertical_difference)**2)
                punishment_action += (v_z_cost + z_err_cost)

            punishment_action += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            reward = reward_distance - 0.1 * punishment_pose - 0.2 * \
                punishment_obs - 0.1 * punishment_action - 0.5 * yaw_error_cost
            
        return reward
    
    def get_depth_image(self):
        return self.dynamic_model.get_depth_image()
    
    def is_done(self):
        episode_done = False

        is_not_inside_workspace_now = self.is_not_inside_workspace()
        has_reached_des_pose = self.is_in_desired_pose()
        too_close_to_obstable = self.is_crashed()

        # We see if we are outside the Learning Space
        episode_done = is_not_inside_workspace_now or\
            has_reached_des_pose or\
            too_close_to_obstable or\
            self.step_num >= self.max_episode_steps

        return episode_done
    
    def is_not_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        current_position = self.dynamic_model.get_position()
        if current_position[0] < self.ep_boundary_x[0] or current_position[0] > self.ep_boundary_x[1] or \
            current_position[1] < self.ep_boundary_y[0] or current_position[1] > self.ep_boundary_y[1] or \
            current_position[2] < self.ep_boundary_z[0] or current_position[2] > self.ep_boundary_z[1]:
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):
        in_desired_pose = False
        if self.dynamic_model.get_distance_to_goal_3d() < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):
        return self.dynamic_model.is_crashed()
    
    def print_train_info_airsim(self, action, obs, reward, info):
        self.dynamic_model.format_and_send_train_info(
            self.episode_num,
            self.step_num,
            self.total_step,
            action,
            reward,
            self.cumulated_episode_reward,
            info,
            self.min_distance_to_obstacles
        )
        