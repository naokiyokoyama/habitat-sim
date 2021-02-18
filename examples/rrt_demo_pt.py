#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import demo_runner as dr
import rrt_unicycle as rrt_u
from rrt_pointturn import RRTStarPointTurn
import os
import numpy as np
import habitat_sim

parser = argparse.ArgumentParser()
parser.add_argument("dataset_json_gz", type=str)
parser.add_argument("dataset_dir", type=str)
parser.add_argument("episode_id", type=int)
parser.add_argument("out_dir", type=str)
parser.add_argument("--max_linear_velocity", type=float, default=0.25) # in m/s
parser.add_argument("--max_angular_velocity", type=float, default=np.pi/180*10) # in rad/s
parser.add_argument("--goal_minimum_distance", type=float, default=0.2) # in m
parser.add_argument("--near_threshold", type=float, default=1.5) # in m
parser.add_argument("--max_distance", type=float, default=1.5) # in m
parser.add_argument("--visualize_on_screen", action="store_true")
parser.add_argument("--iterations", type=int, default=5e3)
parser.add_argument("--visualize_iterations", type=int, default=500)

# parser.add_argument("--scene", type=str, default=dr.default_sim_settings["scene"])
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--max_frames", type=int, default=1000)
parser.add_argument("--save_png", action="store_true")
parser.add_argument("--sensor_height", type=float, default=1.5)
parser.add_argument("--disable_color_sensor", action="store_true")
parser.add_argument("--semantic_sensor", action="store_true")
parser.add_argument("--depth_sensor", action="store_true")
parser.add_argument("--print_semantic_scene", action="store_true")
parser.add_argument("--print_semantic_mask_stats", action="store_true")
parser.add_argument("--compute_shortest_path", action="store_true")
parser.add_argument("--compute_action_shortest_path", action="store_true")
parser.add_argument("--recompute_navmesh", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--silent", action="store_true")
parser.add_argument("--test_fps_regression", type=int, default=0)
parser.add_argument("--enable_physics", action="store_true")
parser.add_argument(
    "--physics_config_file",
    type=str,
    default=dr.default_sim_settings["physics_config_file"],
)
parser.add_argument("--disable_frustum_culling", action="store_true")
args = parser.parse_args()


def make_settings(scene):
    settings = dr.default_sim_settings.copy()
    settings["scene"] = scene

    settings["max_frames"] = args.max_frames
    settings["width"] = args.width
    settings["height"] = args.height
    settings["save_png"] = args.save_png
    settings["sensor_height"] = args.sensor_height
    settings["color_sensor"] = not args.disable_color_sensor
    settings["semantic_sensor"] = args.semantic_sensor
    settings["depth_sensor"] = args.depth_sensor
    settings["print_semantic_scene"] = args.print_semantic_scene
    settings["print_semantic_mask_stats"] = args.print_semantic_mask_stats
    settings["compute_shortest_path"] = args.compute_shortest_path
    settings["compute_action_shortest_path"] = args.compute_action_shortest_path
    settings["seed"] = args.seed
    settings["silent"] = args.silent
    settings["enable_physics"] = args.enable_physics
    settings["physics_config_file"] = args.physics_config_file
    settings["frustum_culling"] = not args.disable_frustum_culling
    settings["recompute_navmesh"] = args.recompute_navmesh

    return settings

start_position, start_heading, goal_position, scene_name = rrt_u.get_episode_info(args.episode_id, args.dataset_json_gz)
scene = os.path.join(args.dataset_dir, scene_name)

settings = make_settings(scene)
demo_runner = dr.DemoRunner(settings, dr.DemoRunnerType.EXAMPLE)
demo_runner.init_common()

navmesh_settings = habitat_sim.NavMeshSettings()
navmesh_settings.set_defaults()
navmesh_settings.agent_height = 0.88
navmesh_settings.agent_radius = 0.18
demo_runner._sim.recompute_navmesh(demo_runner._sim.pathfinder, navmesh_settings)

episode_directory = os.path.join(args.out_dir, '{}_{}_mlv={}_mav={}_md={}_nt={}'.format(
    args.episode_id,
    os.path.splitext(os.path.basename(scene_name))[0],
    args.max_linear_velocity,
    round(np.rad2deg(args.max_angular_velocity),2),
    args.max_distance,
    args.near_threshold,
))

rrt_pointturn = RRTStarPointTurn(
    pathfinder            = demo_runner._sim.pathfinder,
    max_linear_velocity   = args.max_linear_velocity,
    max_angular_velocity  = args.max_angular_velocity,
    near_threshold        = args.near_threshold,
    max_distance          = args.max_distance,
    goal_minimum_distance = args.goal_minimum_distance,
    directory             = episode_directory
)

rrt_pointturn.generate_tree(
    start_position       = start_position,
    start_heading        = start_heading,
    goal_position        = goal_position,
    iterations           = args.iterations,
    visualize_on_screen  = args.visualize_on_screen,
    visualize_iterations = args.visualize_iterations,
)

