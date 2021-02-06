#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from rrt_unicycle import RRTStarUnicycle
import os
import json
import glob
import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("in_json_dir", type=str) # in m
parser.add_argument("out_json_dir", type=str) # in m
parser.add_argument("--max_linear_velocity", type=float, default=0.25) # in m/s
parser.add_argument("--max_angular_velocity", type=float, default=np.pi/180*10) # in rad/s
parser.add_argument("--goal_minimum_distance", type=float, default=0.2) # in m
parser.add_argument("--near_threshold", type=float, default=0.5) # in m
parser.add_argument("--max_distance", type=float, default=0.2) # in m
args = parser.parse_args()

os.makedirs(args.out_json_dir, exist_ok=True)
critical_angle_lookup = None
for json_path in tqdm.tqdm(sorted(glob.glob(os.path.join(args.in_json_dir, '*.json')))):
    rrt_unicycle = RRTStarUnicycle(
        pathfinder            = None,
        max_linear_velocity   = args.max_linear_velocity,
        max_angular_velocity  = args.max_angular_velocity,
        near_threshold        = args.near_threshold,
        max_distance          = args.max_distance,
        goal_minimum_distance = args.goal_minimum_distance,
        critical_angle_lookup = critical_angle_lookup,
    )
    new_json_path = os.path.join(args.out_json_dir, os.path.basename(json_path))
    critical_angle_lookup = rrt_unicycle._critical_angle_lookup
    rrt_unicycle._load_tree_from_json(json_path)
    best_path = rrt_unicycle.calculate_intermediate_points_from_start(rrt_unicycle._best_goal_node)
    best_path += [rrt_unicycle._goal]
    string_tree = rrt_unicycle._string_tree()
    best_path_str = [i._str_key() for i in best_path]
    string_tree['best_path'] = best_path_str

    with open(new_json_path, 'w') as f:
        json.dump(string_tree, f)
