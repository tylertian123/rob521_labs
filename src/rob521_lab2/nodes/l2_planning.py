#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import tqdm
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag

from pathlib import Path

MAPS_DIR = Path(__file__).absolute().parent.parent / "maps"


def load_map(filename):
    im = mpimg.imread(str(MAPS_DIR / filename))
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open(str(MAPS_DIR / filename), "r") as stream:
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
        # self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        # self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        # self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        # self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]
        # Constrain sampling to only within the nonempty areas of the map
        r, c = np.nonzero(self.occupancy_map < self.map_settings_dict["occupied_thresh"])
        min_row = np.min(r)
        min_col = np.min(c)
        max_row = np.max(r)
        max_col = np.max(c)
        
        self.bounds[0, 0] = self.origin[0] + min_col * self.resolution
        self.bounds[0, 1] = self.origin[0] + max_col * self.resolution
        self.bounds[1, 0] = self.origin[1] + (self.map_shape[0] - max_row) * self.resolution
        self.bounds[1, 1] = self.origin[1] + (self.map_shape[0] - min_row) * self.resolution
        
        self.tree_bounds = np.zeros((2, 2))

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.4 #rad/s (Feel free to change!)

        self.collision_disk = disk((0, 0), self.robot_radius / self.resolution)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]
        self.node_pos_np = np.zeros((3, 1))

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", map_filename, 1000, self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        radius = 5
        # With 10% probability, sample around the goal point
        if np.random.rand() < 0.1:
            xmin = self.goal_point[0, 0] - radius
            xmax = self.goal_point[0, 0] + radius
            ymin = self.goal_point[1, 0] - radius
            ymax = self.goal_point[1, 0] + radius
        else:
            # Sample within a box around the current extent of the tree to balance exploration and refinement
            xmin = max(self.bounds[0, 0], self.tree_bounds[0, 0] - radius)
            xmax = min(self.bounds[0, 1], self.tree_bounds[0, 1] + radius)
            ymin = max(self.bounds[1, 0], self.tree_bounds[1, 0] - radius)
            ymax = min(self.bounds[1, 1], self.tree_bounds[1, 1] + radius)
        point = np.random.random((2, 1))
        point[0] = (point[0] * (xmax - xmin)) + xmin
        point[1] = (point[1] * (ymax - ymin)) + ymin
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
        # NOTE this requires self.node_pos_np to be properly maintained!
        return np.argmin(np.sum(np.square(self.node_pos_np[:2, :len(self.nodes)] - point[:2, :]), axis=0))
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(node_i, point_s)
        robot_traj = self.trajectory_rollout(vel, rot_vel, node_i)
        return robot_traj
    
    def angle_to_goal(self, node, point):
        angle = np.arctan2(point[1, 0] - node[1, 0], point[0, 0] - node[0, 0]) - node[2, 0]
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        #         
        # NOTE: Opting for simple proportional control. Could do PID if necessary.
        distance = np.linalg.norm(point_s[:2, 0] - node_i[:2, 0])
        angle_to_goal = self.angle_to_goal(node_i, point_s)
        abs_angle_to_goal = np.abs(angle_to_goal)
        
        linear_vel_max = self.vel_max
        # Proportional control for angular velocity
        max_threshold = np.pi / 2  # Threshold angle for maximum rotation
        if abs_angle_to_goal > max_threshold:  # If the angle is large, use max rotation
            rot_vel = self.rot_vel_max
        else:
            rot_vel = self.rot_vel_max * (abs_angle_to_goal / max_threshold)
        rot_vel = rot_vel * np.sign(angle_to_goal)  # Ensure correct direction

        # Proportional control for linear velocity
        linear_vel = linear_vel_max * distance
        linear_vel = min(linear_vel, linear_vel_max)  # Cap at max velocity

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
            trajectory[0, :] = startX + vel * steps * np.sin(startTheta)
            trajectory[1, :] = startY + vel * steps * np.cos(startTheta)
            trajectory[2, :] = startTheta * np.ones(self.num_substeps)
        else:  # moving along a curve
            radius = vel / rot_vel
            trajectory[0, :] = startX + radius * (np.sin(startTheta + rot_vel * steps) - np.sin(startTheta))
            ## NOTE: MAYBE THE SIGN IS WRONG FOR Y? CHECK WHEN TESTING
            trajectory[1, :] = startY - radius * (np.cos(startTheta + rot_vel * steps) - np.cos(startTheta))
            trajectory[2, :] = startTheta + rot_vel * steps

        return trajectory
    
    # NOTE: this function returns indices in (row, col) order (i.e. (y, x), not (x, y))
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        x, y, theta = self.origin
        translated = point - np.array([[x], [y]])
        # Apply rotation matrix
        # c, s = np.cos(-theta), np.sin(-theta)
        # R = np.array([
        #     [c, -s],
        #     [s,  c]
        # ])
        # rotated = R @ translated
        rotated = translated
        indices = np.round(rotated / self.resolution).astype(int)
        # Switch x and y, since grid indexing is [row, col]
        indices[[0, 1]] = indices[[1, 0]]
        # Invert y axis
        indices[0] = self.occupancy_map.shape[0] - indices[0]
        return indices

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        cell_coords = np.round(self.point_to_cell(points)).astype(int).T
        # Returns an array of 2-array-tuples, first one containing row indices, second one containing column indices
        yield from ((np.clip(self.collision_disk[0] + row, 0, self.occupancy_map.shape[0] - 1),
                     np.clip(self.collision_disk[1] + col, 0, self.occupancy_map.shape[1] - 1)) for (row, col) in cell_coords)

    def collision_check(self, traj) -> int:
        """
        Performs collision checking along a path.
        Returns the index of the last "safe" point within traj.
        If traj.shape[1] - 1 is returned, the entire trajectory is collision-free.
        If -1 is returned, none of the points are collision-free.
        
        :param traj: 2xN or 3xN array of the trajectory.
        :return: Index of the collision-free point within the trajectory.
        """
        # Strip theta
        safe_i = -1
        for i, (occ_rows, occ_cols) in enumerate(self.points_to_robot_circle(traj[:2, :])):
            if not np.all(self.occupancy_map[occ_rows, occ_cols]):
                safe_i = i - 1
                break
        else:
            safe_i = traj.shape[1] - 1
        return safe_i
    
    def add_node(self, node: Node):
        self.nodes[node.parent_id].children_ids.append(len(self.nodes))
        self.node_pos_np[:, len(self.nodes)] = node.point.reshape(3,)
        self.nodes.append(node)
        self.tree_bounds[0, 0] = min(self.tree_bounds[0, 0], node.point[0, 0])
        self.tree_bounds[0, 1] = max(self.tree_bounds[0, 1], node.point[0, 0])
        self.tree_bounds[1, 0] = min(self.tree_bounds[1, 0], node.point[1, 0])
        self.tree_bounds[1, 1] = max(self.tree_bounds[1, 1], node.point[1, 0])

    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def nodes_within_radius(self, point, r):
        dist = np.sum(np.square(self.node_pos_np[:2, :len(self.nodes)] - point[:2].reshape(2, 1)), axis=0)
        return np.nonzero(dist <= r**2)[0]
    
    def connect_node_to_point(self, node_i, point_f):
        # NOTE: returns None if the connection is not possible within our timestep constraints
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        dx = point_f[0, 0] - node_i[0, 0]
        dy = point_f[1, 0] - node_i[1, 0]
        # Rotate into vehicle frame
        c, s = np.cos(node_i[2, 0]), np.sin(node_i[2, 0])
        dxv = dx * c + dy * s
        dyv = -dx * s + dy * c
        if abs(dyv) > 1e-8:
            # Radius of circle
            r = (dxv**2 + dyv**2) / (2 * dyv)
            # angle of arc
            a = 2 * np.arcsin(np.hypot(dxv, dyv) / (2 * r))

            T = max(a / self.rot_vel_max, np.hypot(dxv, dyv) / self.vel_max)
            omega = a / T
            v = omega * r
            if abs(omega) > self.rot_vel_max or abs(v) > self.vel_max:
                return None
        else:
            omega = 0
            v = dxv / self.timestep
            if abs(v) > self.vel_max:
                return None
        return self.trajectory_rollout(v, omega, node_i)
    
    def connect_node_to_point_v2(self, node_i, point_f):
        # Transform target into robot frame
        dx = point_f[0, 0] - node_i[0, 0]
        dy = point_f[1, 0] - node_i[1, 0]
        theta = node_i[2, 0]

        c, s = np.cos(theta), np.sin(theta)
        x_r =  c * dx + s * dy
        y_r = -s * dx + c * dy

        dist = np.hypot(x_r, y_r)
        if dist < 1e-3:
            return None

        best_traj = None
        best_time = np.inf

        omegas = np.linspace(-self.rot_vel_max, self.rot_vel_max, 11)

        for omega in omegas:
            if abs(omega) < 1e-3:
                # Straight line case
                v = x_r / self.timestep
                if abs(v) > self.vel_max or abs(y_r) > 0.1:
                    continue
                T = abs(x_r / v)
            else:
                # Circular arc case
                r = (x_r**2 + y_r**2) / (2 * y_r) if abs(y_r) > 1e-6 else None
                if r is None:
                    continue
                arc_angle = np.arctan2(x_r, r - y_r)
                T = abs(arc_angle / omega)
                v = omega * r
                if abs(v) > self.vel_max or T <= 0:
                    continue

            # Limit traje len
            if T > 3.0:
                continue

            steps = max(5, int(self.num_substeps * T / self.timestep))
            traj = np.zeros((3, steps))
            ts = np.linspace(0, T, steps)
            x0, y0, th0 = node_i.flatten()

            if abs(omega) < 1e-3:
                traj[0, :] = x0 + v * ts * np.cos(th0)
                traj[1, :] = y0 + v * ts * np.sin(th0)
                traj[2, :] = th0
            else:
                r = v / omega
                traj[0, :] = x0 + r * (np.sin(th0 + omega * ts) - np.sin(th0))
                traj[1, :] = y0 - r * (np.cos(th0 + omega * ts) - np.cos(th0))
                traj[2, :] = th0 + omega * ts

            if np.linalg.norm(traj[:2, -1] - point_f.flatten()) > 0.2:
                continue

            if T < best_time:
                best_time = T
                best_traj = traj

        return best_traj

    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        # Sum Euclidean distance between nodes
        diffs = np.diff(trajectory_o[:2], axis=1)
        dist = np.sum(np.sqrt(np.sum(diffs**2, axis=0)))
        # Use path length as cost (TODO find a better metric?)
        return dist
    
    def update_children(self, node_id, cost_delta):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        # TODO: This is fundamentally flawed, because the updated node will have a new theta value,
        # so the path to its children will be different, which changes the children's thetas,
        # and so on and so forth until the leaf nodes. Doing this propagation would be very expensive.
        # Furthermore there is no guarantee that these new paths will be collision-free or possible
        # given the maximum linear and angular velocity constraints. If a path ends up in collision,
        # do we just abandon that entire subtree?
        for i in self.nodes[node_id].children_ids:
            self.nodes[i].cost += cost_delta
            self.update_children(i, cost_delta)

    def draw_tree(self):
        self.window.clear(update=False)
        stack = [0]
        while stack:
            i = stack.pop()
            parent = self.nodes[i].parent_id
            if parent >= 0:
                self.window.add_line(self.nodes[parent].point[:2, 0].flatten(), self.nodes[i].point[:2, 0].flatten(), color=(255, 0, 255), update=False)
            self.window.add_point(self.nodes[i].point[:2, 0].flatten(), color=(0, 255, 0), update=False)
            stack.extend(self.nodes[i].children_ids)
        self.window.update()

    #Planner Functions
    def rrt_planning(self, max_iter=150000, visualize=True):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        # Preallocate space for vectorized closest node computation
        self.node_pos_np = np.zeros((3, max_iter + 1), dtype=np.float32)
        for iter_count in tqdm.trange(max_iter):
            #Sample map space
            point = self.sample_map_space()
            # self.window.add_point(point.flatten()[:2], radius=1, color=pygame_utils.COLORS['b'])

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            safe_i = self.collision_check(trajectory_o)
            if safe_i <= 0:
                continue
            # Add the last point that didn't have a collision
            # No cost considered in RRT
            new_point = trajectory_o[:, safe_i].reshape((3, 1))
            self.add_node(Node(new_point, closest_node_id, 0))

            # pygame visualization
            if visualize:
                # for t in range(safe_i):
                #     self.window.add_line(trajectory_o[:2, t].flatten(), trajectory_o[:2, t + 1].flatten(), color=(255, 0, 255))
                # self.window.add_se2_pose(new_point.flatten(), length=5, color=(0, 255, 0))
                self.window.add_line(trajectory_o[:2, 0].flatten(), trajectory_o[:2, safe_i].flatten(), color=(255, 0, 255))
                self.window.add_point(new_point[:2, 0].flatten(), color=(0, 255, 0))

            if np.hypot(self.goal_point[0, 0] - new_point[0, 0], self.goal_point[1, 0] - new_point[1, 0]) <= self.stopping_dist:
                break
        else:
            raise RuntimeError(f"No path found after {iter_count + 1} iterations!")
        return len(self.nodes) - 1
    
    def rrt_star_planning(self, max_iter=150000, visualize=True):
        #This function performs RRT* for the given map and robot
        # Preallocate space for vectorized closest node computation
        self.node_pos_np = np.zeros((3, max_iter + 1), dtype=np.float32)
        goal_node = -1
        for iter_count in tqdm.trange(max_iter):
            if visualize and iter_count % 100 == 0:
                self.draw_tree()
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
            new_xy = best_point[:2].reshape(2, 1) # Cut off theta for this one
            best_parent = closest_node_id
            # Tentative best cost, calculate using clipped path
            best_cost = self.nodes[closest_node_id].cost + self.cost_to_come(trajectory_o[:, :safe_i + 1])
            # Find everything within the radius
            indices = self.nodes_within_radius(new_xy, self.ball_radius())
            for i in indices:
                if i == closest_node_id:
                    continue
                traj = self.connect_node_to_point(self.nodes[i].point, new_xy)
                # Collision check to make sure the entire trajectory is collision-free
                if traj is None or self.collision_check(traj) != traj.shape[1] - 1:
                    continue
                edge_cost = self.cost_to_come(traj)
                if self.nodes[i].cost + edge_cost < best_cost:
                    best_cost = self.nodes[i].cost + edge_cost
                    best_parent = i
                    best_point[2] = traj[2, -1]
            # Wire to optimal parent
            self.add_node(Node(best_point, best_parent, best_cost))

            #Close node rewire
            for i in indices:
                if i == closest_node_id:
                    continue
                # Collision check
                traj = self.connect_node_to_point(self.nodes[-1].point, self.nodes[i].point[:2])
                if traj is None or self.collision_check(traj) != traj.shape[1] - 1:
                    continue
                edge_cost = self.cost_to_come(traj)
                # Rewire
                new_cost = self.nodes[-1].cost + edge_cost
                if new_cost < self.nodes[i].cost:
                    cost_delta = new_cost - self.nodes[i].cost
                    old_parent = self.nodes[i].parent_id
                    self.nodes[old_parent].children_ids.remove(i)
                    self.nodes[-1].children_ids.append(i)
                    self.nodes[i].parent_id = len(self.nodes) - 1
                    self.nodes[i].cost = new_cost
                    self.nodes[i].point[2, 0] = traj[2, -1]
                    # Magically propagate cost?
                    self.update_children(i, cost_delta)

            if not goal_node == -1 and \
                np.hypot(self.goal_point[0, 0] - self.nodes[-1].point[0, 0],
                         self.goal_point[1, 0] - self.nodes[-1].point[1, 0]) <= self.stopping_dist:
                print(f"Path found after {iter_count + 1} iterations")
                goal_node = len(self.nodes) - 1
        if goal_node == -1:
            raise RuntimeError(f"No path found after {iter_count + 1} iterations!")
        return goal_node
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    np.random.seed(0)
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[42.05], [-44]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    goal_node = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path(goal_node))

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)
    input("Done. Press enter to exit.")


if __name__ == '__main__':
    main()
