import numpy as np
import cv2
import math
import os
import habitat_sim
import random
import json
import tqdm

from collections import defaultdict

from rrt_unicycle import (
    PointHeading,
    RRTStarUnicycle
)

class RRTStarUnicyclePNG(RRTStarUnicycle):
    def __init__(
        self,
        png_path,
        meters_per_pixel,
        agent_radius,
        max_linear_velocity,
        max_angular_velocity,
        near_threshold,
        max_distance,
        goal_minimum_distance=0.2,
        critical_angle_lookup=None,
        directory=None
    ):
        assert max_distance<=near_threshold, (
            'near_threshold ({}) must be greater than or equal to max_distance ({})'.format(max_distance, near_threshold)
        )
        self._near_threshold = near_threshold
        self._max_distance   = max_distance
        self._max_linear_velocity  = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._goal_minimum_distance = goal_minimum_distance
        self._directory = directory

        if critical_angle_lookup is None:
            self._critical_angle_lookup = self._generate_critical_angle_lookup()
        else:
            self._critical_angle_lookup = critical_angle_lookup
        
        if self._directory is not None:
            self._vis_dir  = os.path.join(self._directory, 'visualizations')
            self._json_dir = os.path.join(self._directory, 'tree_jsons')
            for i in [self._vis_dir, self._json_dir]:
                if not os.path.isdir(i):
                    os.makedirs(i)

        img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        self._map = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        self._map[self._map>240]  = 255 
        self._map[self._map<=240] = 0
        blur_radius = int(round(agent_radius / meters_per_pixel))
        self._map = cv2.blur(self._map, (blur_radius,blur_radius))
        self._map_height = float(img.shape[0])*meters_per_pixel
        self._map_width  = float(img.shape[1])*meters_per_pixel
        self._top_down_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self._scale_x = lambda x: int(x/meters_per_pixel)
        # self._scale_y = lambda y: int((self._map_height-y)/meters_per_pixel)
        self._scale_y = lambda y: int((y)/meters_per_pixel)

        # TODO: Search the tree with binary search
        self.tree = {}
        self.grid_hash = defaultdict(list)
        self._best_goal_node = None
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self._start_iteration = 0
        self._start = None
        self._goal = None
        self._cost_from_parent = {}
        self._shortest_path_points = None
        self.x_min = 0
        self.y_min = 0

    def _is_navigable(self, pt):
        px = self._scale_x(pt.x)
        py = self._scale_x(pt.y)

        try:
            return self._map[py,px] == 255
        except IndexError:
            return False

    def _max_point(self, p1, p2):
        euclid_dist = self._euclid_2D(p1, p2)
        if euclid_dist <= self._max_distance:
            return p2, False

        new_x = p1.x+(p2.x-p1.x*self._max_distance/euclid_dist)
        new_y = p1.y+(p2.y-p1.y*self._max_distance/euclid_dist)
        new_z = p1.z
        p_new = PointHeading((new_x, new_z, new_y)) # MAY RETURN non navigable point

        return p_new, True

    def _visualize_tree(
        self,
        meters_per_pixel=0.01,
        show=False,
        path=None, # Option to visualize another path
        draw_all_edges=True,
        save_path=None
    ):
        # top_down_img = cv2.flip(self._top_down_img.copy(), 0)
        top_down_img = self._top_down_img.copy()

        # Draw all edges in orange
        if draw_all_edges:
            for node, node_parent in self.tree.items():
                if node_parent is None: # Start point has no parent
                    continue
                fine_path = [node_parent]+self._get_intermediate_pts(node_parent, node, resolution=0.01)+[node]
                for pt, next_pt in zip(fine_path[:-1], fine_path[1:]):
                    cv2.line(
                        top_down_img,
                        (self._scale_x(pt.x),       self._scale_y(pt.y)),
                        (self._scale_x(next_pt.x),  self._scale_y(next_pt.y)),
                        (0,102,255),
                        1
                    )

        # Draw best path to goal if it exists
        if path is not None or self._best_goal_node is not None:
            if path is not None:
                fine_path = self.make_path_finer(path)
            else:
                fine_path = self.make_path_finer(self._get_best_path())
            for pt, next_pt in zip(fine_path[:-1], fine_path[1:]):
                cv2.line(
                    top_down_img,
                    (self._scale_x(pt.x),       self._scale_y(pt.y)),
                    (self._scale_x(next_pt.x),  self._scale_y(next_pt.y)),
                    (0,255,0),
                    3
                )

        # Draw start point+heading in blue
        start_x, start_y = self._scale_x(self._start.x),  self._scale_y(self._start.y)
        cv2.circle(
            top_down_img,
            (start_x, start_y),
            8,
            (255,192,15),
            -1
        )
        LINE_SIZE = 10
        heading_end_pt = (int(start_x+LINE_SIZE*np.cos(self._start.heading)), int(start_y+LINE_SIZE*np.sin(self._start.heading)))
        cv2.line(
            top_down_img,
            (start_x, start_y),
            heading_end_pt,
            (0,0,0),
            3
        )

        # Draw goal point in red
        # cv2.circle(top_down_img, (self._scale_x(self._goal.x),  self._scale_y(self._goal.y)),  8, (0,255,255), -1)
        SQUARE_SIZE = 6
        cv2.rectangle(
            top_down_img,
            (self._scale_x(self._goal.x)-SQUARE_SIZE,  self._scale_y(self._goal.y)-SQUARE_SIZE),
            (self._scale_x(self._goal.x)+SQUARE_SIZE,  self._scale_y(self._goal.y)+SQUARE_SIZE),
            (0, 0, 255),
            -1
        ) 

        # Draw fastest waypoints
        if path is None:
            path = self._get_best_path()[1:-1]
        for i in path:
            cv2.circle(
                top_down_img,
                (self._scale_x(i.x), self._scale_y(i.y)),
                3,
                (0,0,255),
                -1
            )
            LINE_SIZE = 8
            heading_end_pt = (int(self._scale_x(i.x)+LINE_SIZE*np.cos(i.heading)), int(self._scale_y(i.y)+LINE_SIZE*np.sin(i.heading)))
            cv2.line(
                top_down_img,
                (self._scale_x(i.x), self._scale_y(i.y)),
                heading_end_pt,
                (0,0,0),
                1
            )

        if show:
            cv2.imshow('top_down_img', top_down_img)
            cv2.waitKey(1)

        if save_path is not None:
            cv2.imwrite(save_path, top_down_img)

        return top_down_img

    def generate_tree(
        self,
        start_position,
        start_heading,
        goal_position,
        json_path=None,
        iterations=5e4,
        visualize_on_screen=False,
        visualize_iterations=500,
        seed=0
    ):
        np.random.seed(seed)
        random.seed(seed)
        self._start = PointHeading(start_position, heading=start_heading)
        self._goal  = PointHeading(goal_position)
        self._start.y = self._map_height - self._start.y 
        self._goal.y  = self._map_height - self._goal.y  
        self.tree[self._start] = None
        self._cost_from_parent[self._start] = 0
        # self.times = defaultdict(list)
                
        self.add_to_grid_hash(self._start)
        self._load_tree_from_json(json_path=json_path)

        for iteration in tqdm.trange(int(iterations+1)):
        # for iteration in range(int(iterations+1)):
            if iteration < self._start_iteration:
                continue

            success = False
            while not success:
                # time0 = time.time()
                '''
                Choose random NAVIGABLE point.
                If a path to the goal is already found, with 80% chance, we sample near that path.
                20% chance, explore elsewhere.
                '''
                sample_random = np.random.rand() < 0.2
                found_valid_new_node = False
                while not found_valid_new_node:
                    if sample_random:
                        # rand_pt = PointHeading(self._pathfinder.get_random_navigable_point())
                        x_rand = np.random.rand()*self._map_width
                        y_rand = np.random.rand()*self._map_height
                        rand_pt = PointHeading((x_rand,0,y_rand))
                        if not self._is_navigable(rand_pt): # Must be on free space
                            continue

                        # Shorten distance
                        closest_pt = self._closest_tree_pt(rand_pt)
                        rand_pt, has_changed = self._max_point(closest_pt, rand_pt)
                        if not has_changed or self._is_navigable(rand_pt):
                            found_valid_new_node = True
                    else:
                        if self._best_goal_node is None:
                            sample_random = True; continue
                            # best_path_pt = random.choice(self._shortest_path_points)
                        else:
                            best_path = self._get_path_to_start(self._best_goal_node)
                            best_path_pt = random.choice(best_path)

                        rand_r = 1.5 * np.sqrt(np.random.rand()) # TODO make this adjustable
                        rand_theta = np.random.rand() * 2 * np.pi
                        x = best_path_pt.x + rand_r * np.cos(rand_theta)
                        y = best_path_pt.y + rand_r * np.sin(rand_theta)
                        z = best_path_pt.z
                        rand_pt = PointHeading((x, z, y)) # MAY RETURN NAN NAN NAN

                        if not self._is_navigable(rand_pt):
                            continue

                        if self._best_goal_node is None:
                            closest_pt = self._closest_tree_pt(rand_pt)
                            rand_pt, has_changed = self._max_point(closest_pt, rand_pt)
                            if not has_changed or self._is_navigable(rand_pt):
                                found_valid_new_node = True
                        else:
                            found_valid_new_node = True
                # time1 = time.time()

                # Find valid neighbors
                nearby_nodes = []
                for pt in self._get_near_pts(rand_pt):
                    if (
                        self._euclid_2D(rand_pt, pt) < self._near_threshold # within distance
                        and (rand_pt.x, rand_pt.y) != (pt.x, pt.y) # not the same point again
                        and self._path_exists(pt, rand_pt) # straight path exists
                    ):
                        nearby_nodes.append(pt)
                if not nearby_nodes:
                    continue
                # time2 = time.time()

                # Find best parent from valid neighbors        
                min_cost = float('inf')
                for idx, pt in enumerate(nearby_nodes):
                    cost_from_parent, final_heading = self._cost_from_to(pt, rand_pt, return_heading=True)
                    new_cost = self._cost_from_start(pt) + cost_from_parent
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_final_heading = final_heading
                        best_parent_idx = idx
                        best_cost_from_parent = cost_from_parent
                # Sometimes there is just one nearby node whose new_cost is NaN. Continue if so.
                if min_cost == float('inf'):
                    continue
                # time3 = time.time()

                # Add+connect new node to graph
                rand_pt.heading = best_final_heading
                try:
                    self.tree[rand_pt] = nearby_nodes.pop(best_parent_idx)
                    self._cost_from_parent[rand_pt] = best_cost_from_parent
                    self.add_to_grid_hash(rand_pt)
                except IndexError:
                    continue

                # Rewire
                for pt in nearby_nodes:
                    if pt == self._start:
                        continue
                    cost_from_new_pt = self._cost_from_to(
                        rand_pt,
                        pt, 
                        consider_end_heading=True
                    )
                    new_cost = self._cost_from_start(rand_pt)+cost_from_new_pt
                    if new_cost < self._cost_from_start(pt) and self._path_exists(rand_pt, pt):
                        self.tree[pt] = rand_pt
                        self._cost_from_parent[pt] = cost_from_new_pt

                # Update best path every so often
                if iteration % 50 == 0 or iteration % visualize_iterations == 0 :
                    min_costs = []
                    for idx, pt in enumerate(self._get_near_pts(self._goal)):
                        if (
                            self._euclid_2D(pt, self._goal) < self._near_threshold
                            and self._path_exists(pt, self._goal)
                        ):
                            min_costs.append((
                                self._cost_from_start(pt)+self._cost_from_to(pt, self._goal),
                                idx, # Tie-breaker for previous line when min is used
                                pt
                            ))
                    if len(min_costs) > 0:
                        self._best_goal_node = min(min_costs)[2]

                # Save tree and visualization to disk
                if (
                    iteration > 0 
                    and iteration % visualize_iterations == 0 
                    and self._directory is not None
                ):
                    img_path  = os.path.join(self._vis_dir,  '{}_{}.png'.format( iteration, os.path.basename(self._vis_dir)))
                    json_path = os.path.join(self._json_dir, '{}_{}.json'.format(iteration, os.path.basename(self._json_dir)))
                    self._visualize_tree(save_path=img_path, show=visualize_on_screen)
                    string_tree = self._string_tree()
                    with open(json_path, 'w') as f:
                        json.dump(string_tree, f)
                # self.times['time0'].append(time0)
                # self.times['time1'].append(time1)
                # self.times['time2'].append(time2)
                # self.times['time3'].append(time3)
                # self.times['time4'].append(time4)
                # self.times['time5'].append(time.time())
                # if len(self.times['time5']) == 50:
                #     for idx in range(50):
                #         avg1 = (self.times['time1'][idx]-self.times['time0'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #         avg2 = (self.times['time2'][idx]-self.times['time1'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #         avg3 = (self.times['time3'][idx]-self.times['time2'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #         avg4 = (self.times['time4'][idx]-self.times['time3'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #         avg5 = (self.times['time5'][idx]-self.times['time4'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #     print('{} 1:{} 2:{} 3:{} 4:{} 5:{}'.format(iteration, avg1, avg2, avg3, avg4, avg5))
                #     self.times = defaultdict(list)


                success = True