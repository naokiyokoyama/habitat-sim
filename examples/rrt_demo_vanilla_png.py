#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import demo_runner as dr
from rrt_vanilla_png import RRTStarVanillaPNG
import os
import numpy as np
import habitat_sim

parser = argparse.ArgumentParser()
parser.add_argument("png_path", type=str)
parser.add_argument("start_x_pixel", type=int)
parser.add_argument("start_y_pixel", type=int)
parser.add_argument("start_heading_degrees", type=int)
parser.add_argument("goal_x_pixel", type=int)
parser.add_argument("goal_y_pixel", type=int)
parser.add_argument("out_dir", type=str)
parser.add_argument("--meters_per_pixel", type=float, default=0.008333333333)
parser.add_argument("--max_linear_velocity", type=float, default=0.25) # in m/s
parser.add_argument("--max_angular_velocity", type=float, default=np.pi/180*10) # in rad/s
parser.add_argument("--goal_minimum_distance", type=float, default=0.2) # in m
parser.add_argument("--near_threshold", type=float, default=1.5) # in m
parser.add_argument("--max_distance", type=float, default=1.5) # in m
parser.add_argument("--visualize_on_screen", action="store_true")
parser.add_argument("--iterations", type=int, default=5e3)
parser.add_argument("--visualize_iterations", type=int, default=500)
args = parser.parse_args()

episode_directory = os.path.join(args.out_dir, '{}_mlv={}_mav={}_md={}_nt={}'.format(
    os.path.splitext(os.path.basename(args.png_path))[0],
    args.max_linear_velocity,
    round(np.rad2deg(args.max_angular_velocity),2),
    args.max_distance,
    args.near_threshold,
))

start_position = (
    args.start_x_pixel*args.meters_per_pixel,
    0,
    args.start_y_pixel*args.meters_per_pixel,
)
start_heading = np.deg2rad(args.start_heading_degrees)
goal_position  = (
    args.goal_x_pixel*args.meters_per_pixel,
    0,
    args.goal_y_pixel*args.meters_per_pixel,
)

rrt_unicycle_png = RRTStarVanillaPNG(
    png_path              = args.png_path,
    meters_per_pixel      = args.meters_per_pixel,
    agent_radius          = 0.18,
    near_threshold        = args.near_threshold,
    max_distance          = args.max_distance,
    goal_minimum_distance = args.goal_minimum_distance,
    directory             = episode_directory
)

rrt_unicycle_png.generate_tree(
    start_position       = start_position,
    start_heading        = start_heading,
    goal_position        = goal_position,
    iterations           = args.iterations,
    visualize_on_screen  = args.visualize_on_screen,
    visualize_iterations = args.visualize_iterations,
)

