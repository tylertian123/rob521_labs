#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag


def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        if point.shape == (3,):
            point = point.reshape(3, 1)
        else:
            assert point.shape == (3, 1), f"Wrong shape {point.shape}! Make sure theta is included!"
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)
        self.origin = self.map_settings_dict["origin"]
        self.resolution = self.map_settings_dict["resolution"]

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        # TODO restrict the sampling to a bounding box within the root node to control expansion
        point = np.random.random((2, 1))
        point[0] = (point[0] * (self.bounds[0, 1] - self.bounds[0, 0])) + self.bounds[0, 0]
        point[1] = (point[1] * (self.bounds[1, 1] - self.bounds[1, 0])) + self.bounds[1, 0]
        return point
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        print("TO DO: Check that nodes are not duplicates")
        # TODO: what is this function actually used for?
        # what qualifies as a duplicate?
        raise NotImplementedError()
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        return min(range(len(self.nodes)), key=lambda i: np.hypot(self.nodes[i].point[0, 0] - point[0, 0], self.nodes[i].point[1, 0] - point[1, 0]))
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(node_i, point_s)
        robot_traj = self.trajectory_rollout(vel, rot_vel, node_i)
        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        #         
        # NOTE: Opting for simple proportional control. Could do PID if necessary.
        distance = np.linalg.norm(point_s[:2, 0] - node_i[:2, 0])
        angle_to_goal = np.arctan2(point_s[1, 0] - node_i[1, 0], point_s[0, 0] - node_i[0, 0])
        # normalize angle
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        
        # Proportional control for angular velocity
        max_threshold = np.pi / 4  # Threshold angle for maximum rotation
        if np.abs(angle_to_goal) > max_threshold:  # If the angle is large, use max rotation
            rot_vel = self.rot_vel_max
        else:
            rot_vel = self.rot_vel_max * (angle_to_goal / max_threshold)
        rot_vel = min(rot_vel, self.rot_vel_max)  # Cap at max rotation velocity
        rot_vel = rot_vel * np.sign(angle_to_goal)  # Ensure correct direction

        # Proportional control for linear velocity
        linear_vel = self.vel_max * distance
        linear_vel = min(linear_vel, self.vel_max)  # Cap at max velocity

        return linear_vel, rot_vel    
    
    def trajectory_rollout(self, vel, rot_vel, point):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        ## NOTE: Changed the function signature to include the start point
        # preallocated trajectory and steps (x, y, theta)
        startX, startY, startTheta = point.flatten()
        trajectory = np.zeros((3, self.num_substeps))
        steps = np.linspace(0, self.timestep, self.num_substeps)

        if rot_vel == 0:  # moving straight
            trajectory[0, :] = startX + vel * steps * np.cos(startTheta)
            trajectory[1, :] = startY + vel * steps * np.sin(startTheta)
            trajectory[2, :] = startTheta * np.ones(self.num_substeps)
        else:  # moving along a curve
            radius = vel / rot_vel
            trajectory[0, :] = startX + radius * (np.cos(startTheta + rot_vel * steps) - np.cos(startTheta))
            ## NOTE: MAYBE THE SIGN IS WRONG FOR Y? CHECK WHEN TESTING
            trajectory[1, :] = startY + radius * (np.sin(startTheta + rot_vel * steps) - np.sin(startTheta))
            trajectory[2, :] = startTheta + rot_vel * steps

        return trajectory
    
    # NOTE: this function returns indices in (row, col) order (i.e. (y, x), not (x, y))
    # It also assumes the y axis points up, which might not be the case for the occupancy grid
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        x, y, theta = self.origin
        translated = point - np.array([[x], [y]])
        # Apply rotation matrix
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([
            [c, -s],
            [s,  c]
        ])
        rotated = R @ translated
        indices = np.round(rotated / self.resolution).astype(int)
        # Switch x and y, since grid indexing is [row, col]
        indices[[0, 1]] = indices[[1, 0]]
        # TODO do we need to flip the y axis here?
        return indices

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        cell_coords = self.point_to_cell(points)
        radius_cells = self.robot_radius / self.resolution
        cells = [disk((x, y), radius_cells, shape=self.map_shape) for (x, y) in cell_coords.T]
        # Returns an array of 2-array-tuples, first one containing row indices, second one containing column indices
        return cells

    def collision_check(self, traj) -> int:
        """
        Performs collision checking along a path.
        Returns the index of the last "safe" point within traj.
        If len(traj) - 1 is returned, the entire trajectory is collision-free.
        If -1 is returned, none of the points are collision-free.
        
        :param traj: 2xN or 3xN array of the trajectory.
        :return: Index of the collision-free point within the trajectory.
        """
        # Strip theta
        occ_points = self.points_to_robot_circle(traj[:2])
        safe_i = -1
        for i, (occ_rows, occ_cols) in enumerate(occ_points):
            # TODO are True cells occupied or False cells occupied?
            if not np.all(self.occupancy_map[occ_rows, occ_cols]):
                safe_i = i - 1
                break
        else:
            safe_i = len(occ_points) - 1
        return safe_i
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self, max_iter=1000):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        # TODO tune this
        for iter_count in range(max_iter):
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            safe_i = self.collision_check(trajectory_o)
            if safe_i == -1:
                continue
            # Add the last point that didn't have a collision
            # No cost considered in RRT
            new_point = trajectory_o[:, safe_i].reshape((3, 1))
            self.nodes.append(Node(new_point, closest_node_id, 0))
            self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)
            
            if np.hypot(self.goal_point[0, 0] - new_point[0, 0], self.goal_point[1, 0] - new_point[1, 0]) <= self.stopping_dist:
                break
        else:
            raise RuntimeError(f"No path found after {iter_count + 1} iterations!")
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot
        # TODO tune this
        for iter_count in range(1000):
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            safe_i = self.collision_check(trajectory_o)

            # Add the last point that didn't have a collision
            best_point = trajectory_o[:, safe_i]
            # Find optimal parent in neighbourhood
            new_xy = best_point[:2] # Cut off theta for this one
            best_parent = closest_node_id
            # Tentative best cost, calculate using clipped path
            best_cost = self.nodes[closest_node_id].cost + self.cost_to_come(trajectory_o[:, :safe_i + 1])
            # Find everything within the radius
            r = self.ball_radius()
            lengths = np.array([np.hypot(new_xy[0, 0] - n.point[0, 0], new_xy[1, 0] - n.point[1, 0]) for n in self.nodes])
            indices = lengths >= r
            for i in indices:
                if i == closest_node_id:
                    continue
                traj = self.connect_node_to_point(self.nodes[i].point, new_xy)
                # Collision check to make sure the entire trajectory is collision-free
                if self.collision_check(traj) != len(traj) - 1:
                    continue
                edge_cost = self.cost_to_come(traj)
                if self.nodes[i].cost + edge_cost < best_cost:
                    best_cost = self.nodes[i].cost + edge_cost
                    best_parent = i
                    best_point = traj[-1]
            # Wire to optimal parent
            self.nodes.append(Node(best_point, best_parent, best_cost))
            self.nodes[best_parent].children_ids.append(len(self.nodes) - 1)

            #Close node rewire
            for i in indices:
                if i == closest_node_id:
                    continue
                # Collision check
                traj = self.connect_node_to_point(self.nodes[-1].point, self.nodes[i].point[:2])
                if self.collision_check(traj) != len(traj) - 1:
                    continue
                edge_cost = self.cost_to_come(traj)
                # Rewire
                if self.nodes[-1].cost + edge_cost < self.nodes[i].cost:
                    old_parent = self.nodes[i].parent_id
                    self.nodes[old_parent].children_ids.remove(i)
                    self.nodes[i].parent_id = len(self.nodes) - 1
                    self.nodes[-1].children_ids.append(i)
                    self.nodes[i].cost = self.nodes[-1].cost + edge_cost
                    self.nodes[i].point = traj[-1]
                    # Magically propagate cost?
                    self.update_children(i)


            if np.hypot(self.goal_point[0, 0] - self.nodes[-1].point[0, 0],
                        self.goal_point[1, 0] - self.nodes[-1].point[1, 0]) <= self.stopping_dist:
                print(f"Path found after {iter_count + 1} iterations")
                break
        else:
            raise RuntimeError(f"No path found after {iter_count + 1} iterations!")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[42.05], [-44]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    # nodes = path_planner.rrt_star_planning()
    nodes = path_planner.rrt_planning(max_iter=3000)
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
