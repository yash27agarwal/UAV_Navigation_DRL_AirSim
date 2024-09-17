import airsim
import numpy as np
import random

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()

# RRT Parameters
start = np.array([0.0, 0.0, -10.0])  # Starting position
goal = np.array([10.0, 0.0, -10.0])  # Goal position
step_size = 5.0
max_iterations = 1000

# Tree structure
nodes = [start]
parents = {0: None}

def sample():
    return np.array([random.uniform(-100, 100),
                     random.uniform(-100, 100),
                     -10.0])  # Fixed altitude for simplicity

def nearest_node(sample):
    distances = [np.linalg.norm(sample - node) for node in nodes]
    nearest_index = distances.index(min(distances))
    return nearest_index

def is_collision_free(p1, p2):
    # Convert to standard Python floats
    p1 = [float(coord) for coord in p1]
    p2 = [float(coord) for coord in p2]
    return client.simTestLineOfSightBetweenPoints(airsim.Vector3r(*p1), airsim.Vector3r(*p2))

found = False
for i in range(max_iterations):
    rand_sample = sample()
    nearest_index = nearest_node(rand_sample)
    nearest_point = nodes[nearest_index]
    
    direction = rand_sample - nearest_point
    length = np.linalg.norm(direction)
    if length == 0:
        continue
    direction = (direction / length) * min(step_size, length)
    new_point = nearest_point + direction
    
    if is_collision_free(nearest_point, new_point):
        nodes.append(new_point)
        parents[len(nodes) - 1] = nearest_index
        
        if np.linalg.norm(new_point - goal) < step_size:
            if is_collision_free(new_point, goal):
                nodes.append(goal)
                parents[len(nodes) - 1] = len(nodes) - 2
                found = True
                break

if found:
    # Extract path
    path_indices = []
    index = len(nodes) - 1
    while index is not None:
        path_indices.append(index)
        index = parents[index]
    path_indices.reverse()
    
    # Convert nodes to standard Python floats for AirSim
    path = []
    for i in path_indices:
        node = nodes[i]
        coords = [float(coord) for coord in node]
        path.append(airsim.Vector3r(*coords))
    
    # Move the drone along the path
    client.moveOnPathAsync(path, velocity=5).join()
else:
    print("Path not found")

# Land
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
