#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import demo_runner as dr
from rrt_unicycle_png import RRTStarUnicyclePNG
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

rrt_unicycle_png = RRTStarUnicyclePNG(
    png_path              = args.png_path,
    max_linear_velocity   = args.max_linear_velocity,
    meters_per_pixel      = args.meters_per_pixel,
    agent_radius          = 0.18,
    max_angular_velocity  = args.max_angular_velocity,
    near_threshold        = args.near_threshold,
    max_distance          = args.max_distance,
    goal_minimum_distance = args.goal_minimum_distance,
)

# rrt_unicycle._set_offsets()
rrt_unicycle_png._load_tree_from_json(
    json_path = '/Users/naoki/gt/path_planning/habitat-sim/../arium_pantry2aruna/arium_map_mlv=0.25_mav=10.0_md=1.5_nt=1.5/tree_jsons/5000_tree_jsons.json'
)

rrt_unicycle_png._visualize_tree(
    show=True,
    save_path=f'/Users/naoki/Downloads/help_me_man.png'
)