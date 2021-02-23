import numpy as np
import cv2
import math
import os
import habitat_sim
import random
import json
import tqdm

from collections import defaultdict

from rrt_unicycle import PointHeading
from rrt_unicycle_png import RRTStarUnicyclePNG


class RRTStarVanillaPNG(RRTStarUnicyclePNG):
    def __init__(
        self,
        png_path,
        meters_per_pixel,
        agent_radius,
        near_threshold,
        max_distance,
        goal_minimum_distance=0.2,
        directory=None
    ):
        assert max_distance<=near_threshold, (
            'near_threshold ({}) must be greater than or equal to max_distance ({})'.format(max_distance, near_threshold)
        )
        self._near_threshold = near_threshold
        self._max_distance   = max_distance
        self._goal_minimum_distance = goal_minimum_distance
        self._directory = directory
        
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
        self._map[self._map<255]  = 0
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
        self._start_iteration = 0
        self._start = None
        self._goal = None
        self._cost_from_parent = {}
        self._shortest_path_points = None
        self.x_min = 0
        self.y_min = 0

    def _path_exists(self, p1, p2):
        num_pts = int(self._euclid_2D(p1, p2)/0.01)
        for x,y in zip(np.linspace(p1.x, p2.x, num_pts), np.linspace(p1.y, p2.y, num_pts)):
            new_pt = PointHeading([x, 0, y])
            if not self._is_navigable(new_pt):
                return False

        return True


    def _get_intermediate_pts(self, pt, new_pt, precision=None, resolution=0.1):
        return []

    def _cost_from_to(
        self, 
        pt, 
        new_pt, 
        return_heading=False,
        consider_end_heading=False
    ):
        # theta is the angle from pt to new_pt (0 rad is east)
        theta = math.atan2((new_pt.y-pt.y), new_pt.x-pt.x)
        # theta_diff is the angle between the robot's heading at pt to new_pt
        # theta_diff = self._get_heading_error(pt.heading, theta)
        euclid_dist = self._euclid_2D(pt, new_pt)

        if return_heading:
            return euclid_dist, theta

        return euclid_dist
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
            if path is None:
                fine_path  = self._get_best_path()
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