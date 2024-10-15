import airsim
import math
import random
import yaml
import numpy as np

class MultirotorDynamicsAirsim():
    def __init__(self, yaml_file):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def set_boundary_conditions(self, workspace_boundry_x, 
                                workspace_boundry_y, 
                                workspace_boundry_z):
        self.workspace_boundry_x = workspace_boundry_x
        self.workspace_boundry_y = workspace_boundry_y
        self.workspace_boundry_z = workspace_boundry_z

    def set_constraints(self, acc_xy_max, v_xy_max, v_xy_min, v_z_max, yaw_rate_max_rad):
        self.acc_xy_max = acc_xy_max
        self.v_xy_max = v_xy_max
        self.v_xy_min = v_xy_min
        self.v_z_max = v_z_max
        self.yaw_rate_max_rad = yaw_rate_max_rad

    def set_goal_parameters(self, goal_distance, env_center):
        self.goal_distance = goal_distance
        self.env_center = env_center

    def set_height(self, height):
        self.height = height

    def reset(self):
        self.client.reset()
        pose = self.client.simGetVehiclePose()
        is_crashed = True
        
        while(is_crashed):
            pose.position.x_val = random.uniform(self.workspace_boundry_x[0], 
                                                self.workspace_boundry_x[1])
            pose.position.y_val = random.uniform(self.workspace_boundry_y[0], 
                                                self.workspace_boundry_y[1])
            pose.position.z_val = -self.height
            yaw = random.uniform(-math.pi, math.pi) 
            pose.orientation = airsim.to_quaternion(0, 0, yaw)
            self.client.simSetVehiclePose(pose, True)
            self.client.simPause(False)
            self.client.armDisarm(True)

            # take off
            self.client.moveToZAsync(pose.position.z_val, 2).join()
            self.client.simPause(True)
    
            collision_info = self.client.simGetCollisionInfo()
            if collision_info.has_collided:
                is_crashed = True
            else:
                is_crashed = False

        self.update_goal_pose(pose)
        self.start_position = [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def update_episode_boundary(self):
        ep_boundary_x = [self.start_position[0] - self.goal_distance, 
                         self.goal_position[0] + self.goal_distance,] 
        ep_boundary_y = [self.start_position[1] - self.goal_distance, 
                         self.goal_position[1] + self.goal_distance,] 
        ep_boundary_z = [0.5, 5] 
        return ep_boundary_x, ep_boundary_y, ep_boundary_z

    def update_goal_pose(self, pose):
        current_position = pose.position

        # Calculate direction vector from current position to the environment center (only in X and Y)
        direction_vector = airsim.Vector3r(
            self.env_center[0] - current_position.x_val,
            self.env_center[1] - current_position.y_val,
            0  # Ignore the Z direction
        )

        # Normalize the direction vector (only X and Y)
        direction_magnitude = math.sqrt(direction_vector.x_val**2 + direction_vector.y_val**2)

        # Avoid division by zero if the drone is at the center (X and Y plane)
        if direction_magnitude == 0:
            direction_magnitude = 1

        normalized_direction = airsim.Vector3r(
            direction_vector.x_val / direction_magnitude,
            direction_vector.y_val / direction_magnitude,
            0  # Z remains zero since we ignore it
        )

        # Scale the normalized direction vector by the goal distance (only X and Y)
        goal_x = current_position.x_val + normalized_direction.x_val * self.goal_distance
        goal_y = current_position.y_val + normalized_direction.y_val * self.goal_distance
        goal_z = current_position.z_val  # Keep the Z position unchanged
        self.goal_position = [goal_x, goal_y, goal_z]
    
    def get_depth_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)])

        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages(
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True))
        
        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float, responses[0].width,
            responses[0].height)
        depth_meter = depth_img * 100
        return depth_meter
    
    def get_state_feature(self):
        self.update_drone_state()

        x = self.position.x_val
        y = self.position.y_val
        z = -self.position.z_val

        # Get orientation (in quaternions)
        orientation_vector = np.array([self.orientation.w_val,
                                       self.orientation.x_val,
                                       self.orientation.y_val,
                                       self.orientation.z_val], np.float32)
        scaled_orientation_vector = (orientation_vector + 1) / 2 

        # Get velocity (in m/s)
        vx = self.velocity.x_val 
        vy = self.velocity.y_val
        vz = self.velocity.z_val

        v_xy = math.sqrt(pow(vx, 2) + pow(vy, 2))
        v_xy = (v_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) 
        v_z =  (vz + self.v_z_max) / (2*self.v_z_max)
        velocity_vector = np.array([v_xy, v_z], np.float32)

        desired_vector = np.array([x - self.goal_position[0], 
                                    y - self.goal_position[1],
                                    z - self.goal_position[2]], np.float32)
        magnitude = np.linalg.norm(desired_vector)
        if magnitude < 10e-5:
            desired_heading = np.array([0, 0, 0], np.float32)
        else:
            desired_heading = desired_vector / magnitude 

        scaled_desired_heading = (desired_heading + 1) / 2
        return velocity_vector, scaled_orientation_vector, scaled_desired_heading

    def set_action(self, action):
        self.v_xy_sp = action[0] * 0.7
        self.yaw_rate_sp = action[-1] * 2
        if self.navigation_3d:
            self.v_z_sp = float(action[1])
        else:
            self.v_z_sp = 0

        self.yaw = self.get_attitude()[2]
        self.yaw_sp = self.yaw + self.yaw_rate_sp * self.dt

        if self.yaw_sp > math.radians(180):
            self.yaw_sp -= math.pi * 2
        elif self.yaw_sp < math.radians(-180):
            self.yaw_sp += math.pi * 2

        vx_local_sp = self.v_xy_sp * math.cos(self.yaw_sp)
        vy_local_sp = self.v_xy_sp * math.sin(self.yaw_sp)

        self.client.simPause(False)
        if len(action) == 2:
            self.client.moveByVelocityZAsync(vx_local_sp, vy_local_sp, -self.start_position[2], self.dt,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))).join()
            # self.client.moveByVelocityZAsync(vx_local_sp, vy_local_sp, -self.start_position[2], self.dt,
            #                                 drivetrain=airsim.DrivetrainType.ForwardOnly,
            #                                 yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(0))).join()
        elif len(action) == 3:
            self.client.moveByVelocityAsync(vx_local_sp, vy_local_sp, -self.v_z_sp, self.dt,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))).join()
                        
        self.client.simPause(True)

    def update_drone_state(self):
        state = self.client.getMultirotorState()
        self.position = state.kinematics_estimated.position
        self.orientation = state.kinematics_estimated.orientation
        self.velocity = state.kinematics_estimated.linear_velocity
        self.angular_velocity = state.kinematics_estimated.angular_velocity

    def get_position(self):
        return [self.position.x_val, self.position.y_val, -self.position.z_val]

    def get_attitude(self):
        self.state_current_attitude = self.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)
    
    def set_navigation_mode(self, navigation_3d):
        self.navigation_3d = navigation_3d

    def get_distance_to_goal_3d(self):
        current_position = self.position
        goal_position = self.goal_position
        relative_pose_x = current_position.x_val - goal_position[0]
        relative_pose_y = current_position.y_val - goal_position[1]
        relative_pose_z = current_position.z_val - goal_position[2]
        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))
    
    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.crash_distance:
            is_crashed = True

        return is_crashed
    
    def set_crash_radius(self, radius):
        self.crash_distance = radius

    def set_min_distance_to_obstacles(self, distance):
        self.min_distance_to_obstacles = distance

    def _get_state_feature_old(self):
        '''
        @description: update and get current uav state and state_norm 
        @param {type} 
        @return: state_norm
                    normalized state range 0-255
        '''

        distance = self.get_distance_to_goal_2d()
        relative_yaw = self._get_relative_yaw()  # return relative yaw -pi to pi 
        relative_pose_z = self.get_position()[2] - self.goal_position[2]  # current position z is positive
        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255

        distance_norm = distance / self.goal_distance * 255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5) * 255

        # current speed and angular speed
        velocity = self.get_velocity()
        linear_velocity_xy = velocity[0]
        linear_velocity_norm = (linear_velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) * 255
        linear_velocity_z = velocity[1]
        linear_velocity_z_norm = (linear_velocity_z / self.v_z_max / 2 + 0.5) * 255
        angular_velocity_norm = (velocity[2] / self.yaw_rate_max_rad / 2 + 0.5) * 255
        # state: distance_h, distance_v, relative yaw, velocity_x, velocity_z, velocity_yaw
        self.state_raw = np.array([distance, relative_pose_z,  math.degrees(
            relative_yaw), linear_velocity_xy, linear_velocity_z,  math.degrees(velocity[2])])
        state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm,
                                linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])
        state_norm = np.clip(state_norm, 0, 255)

        if self.navigation_3d:
            if self.using_velocity_state == False:
                state_norm = state_norm[:3]
        else:
            state_norm = np.array(
                [state_norm[0], state_norm[2], state_norm[3], state_norm[5]])
            if self.using_velocity_state == False:
                state_norm = state_norm[:2]

        self.state_norm = state_norm

    def send_message(self, title, message):
        """Send the message to AirSim"""
        self.client.simPrintLogMessage(title, message)

    def format_and_send_train_info(self, episode_num, step_num, total_step, action, reward, cumulated_reward, info, min_distance):
        """Format and send all the training-related information to AirSim"""
        msg_train_info = "EP: {} Step: {} Total_step: {}".format(episode_num, step_num, total_step)
        
        self.send_message('Train: ', msg_train_info)
        self.send_message('Action: ', str(action))
        self.send_message('Reward: ', "{:4.4f} total: {:4.4f}".format(reward, cumulated_reward))
        self.send_message('Info: ', str(info))
        self.send_message('Min_depth: ', str(min_distance))
