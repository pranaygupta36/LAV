#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.envs.sensor_interface import SensorInterface, CallBack, SpeedometerReader, OpenDriveMapReader
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider
from srunner.tools.route_parser import RouteParser
from leaderboard.utils.route_manipulation import downsample_route, interpolate_trajectory

import numpy as np
from collections import deque
import cv2
import torch
import matplotlib
from lbc_libs.lbc_data_utils import *
from PIL import Image, ImageDraw, ImageFont
import argparse
import importlib



def setup_sensors(vehicle, world, agent, bev_sensor=None, debug_mode=False):
    """
    Create the sensors defined by the user and attach them to the ego-vehicle
    :param vehicle: ego vehicle
    :return:
    """
    bp_library = world.get_blueprint_library()
    #saving birds eye view separately
    agent_sensors = agent.sensors()
    if bev_sensor is not None:
        agent_sensors.append(bev_sensor)
    sensors_list = []
    for sensor_spec in agent_sensors:
        # These are the pseudosensors (not spawned)
        if sensor_spec['type'].startswith('sensor.opendrive_map'):
            # The HDMap pseudo sensor is created directly here
            sensor = OpenDriveMapReader(vehicle, sensor_spec['reading_frequency'])
        elif sensor_spec['type'].startswith('sensor.speedometer'):
            delta_time = world.get_settings().fixed_delta_seconds
            frame_rate = 1 / delta_time
            sensor = SpeedometerReader(vehicle, frame_rate)
        # These are the sensors spawned on the carla world
        else:
            bp = bp_library.find(str(sensor_spec['type']))
            if sensor_spec['type'].startswith('sensor.camera'):
                if sensor_spec['id'].startswith('BEV_RGB'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(1.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    bp.set_attribute('chromatic_aberration_offset', str(0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])
                else:    
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    bp.set_attribute('chromatic_aberration_offset', str(0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.lidar'):
                bp.set_attribute('range', str(85))
                bp.set_attribute('rotation_frequency', str(10))
                bp.set_attribute('channels', str(64))
                bp.set_attribute('upper_fov', str(10))
                bp.set_attribute('lower_fov', str(-30))
                bp.set_attribute('points_per_second', str(600000))
                bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                bp.set_attribute('dropoff_general_rate', str(0.45))
                bp.set_attribute('dropoff_intensity_limit', str(0.8))
                bp.set_attribute('dropoff_zero_intensity', str(0.4))
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.other.radar'):
                bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                bp.set_attribute('points_per_second', '1500')
                bp.set_attribute('range', '100')  # meters

                sensor_location = carla.Location(x=sensor_spec['x'],
                                                    y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])

            elif sensor_spec['type'].startswith('sensor.other.gnss'):
                bp.set_attribute('noise_alt_stddev', str(0.000005))
                bp.set_attribute('noise_lat_stddev', str(0.000005))
                bp.set_attribute('noise_lon_stddev', str(0.000005))
                bp.set_attribute('noise_alt_bias', str(0.0))
                bp.set_attribute('noise_lat_bias', str(0.0))
                bp.set_attribute('noise_lon_bias', str(0.0))

                sensor_location = carla.Location(x=sensor_spec['x'],
                                                    y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation()

            elif sensor_spec['type'].startswith('sensor.other.imu'):
                bp.set_attribute('noise_accel_stddev_x', str(0.001))
                bp.set_attribute('noise_accel_stddev_y', str(0.001))
                bp.set_attribute('noise_accel_stddev_z', str(0.015))
                bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                bp.set_attribute('noise_gyro_stddev_z', str(0.001))

                sensor_location = carla.Location(x=sensor_spec['x'],
                                                    y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])
            # create sensor
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)
            sensor = world.spawn_actor(bp, sensor_transform, vehicle)
        # setup callback
        sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, agent.sensor_interface))
        sensors_list.append(sensor)

    # Tick once to spawn the sensors
    world.tick()
    return sensors_list    

def get_forward_speed(transform=None, velocity=None):
    """ Convert the vehicle transform directly to forward speed """
    
    vel_np = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    speed = np.dot(vel_np, orientation)
    return speed 

def _set_global_plan(agent, global_plan_gps, global_plan_world_coord):
    """
    Set the plan (route) for the agent
    """
    ds_ids = downsample_route(global_plan_world_coord, 50)
    agent._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
    agent._global_plan = [global_plan_gps[x] for x in ds_ids]

    agent._plan_HACK = global_plan_world_coord
    agent._plan_gps_HACK = global_plan_gps
    return

def _read_route(world, agent, route_id, debug_mode=False):
    """
    Update the input route, i.e. refine waypoint list, and extract possible scenario locations
    """
    route_config = RouteParser.parse_routes_file('./leaderboard/data/longest6/longest6.xml',
                    './leaderboard/data/longest6/eval_scenarios.json', str(route_id))
    # for use there's only one route
    route_config = route_config[0]

    # prepare route's trajectory (interpolate and add the GPS route) 
    gps_route, route = interpolate_trajectory(world, route_config.trajectory)
    # this will be the global plan for the replay agent    
    _set_global_plan(agent, gps_route, route)
    ds_ids = downsample_route(route, 50)
    return

def build_projection_matrix(w = 1280, h = 720, fov=100):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(point, K, w2c, e2w):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    # point in ego vehicle's frame of reference 
    point_world =  np.dot(e2w, point)
    # transform to camera coordinates
    point_camera = np.dot(w2c, point_world)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def vis_bev(bev, ego_matrix, camera, pred_loc, det, cast_locs, cast_cmds, important_vehicles):
    K = build_projection_matrix()
    w2c = camera.get_transform().get_inverse_matrix()
    e2w = ego_matrix
    ego = get_image_point(np.array([0, 0, 3, 1]), K, w2c, e2w)
    bev = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)
    bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)


    for loc in pred_loc:
        loc = np.array([-1*loc[1], loc[0], 3, 1])
        bev = cv2.circle(bev, tuple(get_image_point(loc, K, w2c, e2w).astype(int)), 3, (255,0,0), -1)
    
    # for x, y, ww, hh, cos, sin in det[1]:

    #         p1 = tuple(([x,y] + [-ww,-hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
    #         p2 = tuple(([x,y] + [-ww, hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
    #         p3 = tuple(([x,y] + [ ww, hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
    #         p4 = tuple(([x,y] + [ ww,-hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))

    #         p1 = tuple(get_image_point(np.array([-1*(p1[1] - 280)/4, (p1[0] - 160)/4, 0, 1]), K, w2c, e2w))
    #         p2 = tuple(get_image_point(np.array([-1*(p2[1] - 280)/4, (p2[0] - 160)/4,  0, 1]), K, w2c, e2w))
    #         p3 = tuple(get_image_point(np.array([-1*(p3[1] - 280)/4, (p3[0] - 160)/4, 0, 1]), K, w2c, e2w))
    #         p4 = tuple(get_image_point(np.array([-1*(p4[1] - 280)/4, (p4[0] - 160)/4, 0, 1]), K, w2c, e2w))
            
    #         cv2.drawContours(bev, np.array([[p1,p2,p3,p4]]).astype(int), 0, (255,0,0), 2)
    important_vehicle_bb_image_locations = []
    # if len(det) > 0:                
    for v in important_vehicles:
        x, y, ww, hh, cos, sin = det[1][v[0]]
        p1 = tuple(([x,y] + [-ww,-hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
        p2 = tuple(([x,y] + [-ww, hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
        p3 = tuple(([x,y] + [ ww, hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
        p4 = tuple(([x,y] + [ ww,-hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
        # print(p1, p2, p3, p4)
        
        p1 = tuple(get_image_point(np.array([-1*(p1[1] - 280)/4, (p1[0] - 160)/4, 0, 1]), K, w2c, e2w))
        p2 = tuple(get_image_point(np.array([-1*(p2[1] - 280)/4, (p2[0] - 160)/4,  0, 1]), K, w2c, e2w))
        p3 = tuple(get_image_point(np.array([-1*(p3[1] - 280)/4, (p3[0] - 160)/4, 0, 1]), K, w2c, e2w))
        p4 = tuple(get_image_point(np.array([-1*(p4[1] - 280)/4, (p4[0] - 160)/4, 0, 1]), K, w2c, e2w))
        # print(p1, p2, p3, p4)
        for i, loc in enumerate(v[1]):
            if v[3] == 1:
                color = (255, 255, 0)
            elif v[3] == 2:
                color = (0, 255, 0)
            elif v[3] == 3:
                color = (0, 255, 255)
            elif v[3] == 4:
                color = (0, 0, 255)
            elif v[3] == 5:
                color = (0, 0, 255)
            else:
                color = (255, 0, 255)
            loc = np.array([-1*loc[1], loc[0], 3, 1])
            bev = cv2.circle(bev, tuple(get_image_point(loc, K, w2c, e2w).astype(int)), 3, color, -1)
        cv2.drawContours(bev, np.array([[p1,p2,p3,p4]]).astype(int), 0, color, 2)
        important_vehicle_bb_image_locations.append([p1, p2, p3, p4])

    return bev, important_vehicle_bb_image_locations

def vis_bev_projected(bev, ego_matrix, camera, pred_loc, det, cast_locs, cast_cmds, important_vehicles):
    K = build_projection_matrix()
    w2c = camera.get_transform().get_inverse_matrix()
    e2w = ego_matrix
    ego = get_image_point(np.array([0, 0, 3, 1]), K, w2c, e2w)
    bev = cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)
    bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)


    for loc in pred_loc:
        loc = np.array([-1*loc[1], loc[0], 3, 1])
        bev = cv2.circle(bev, tuple(get_image_point(loc, K, w2c, e2w).astype(int)), 3, (255,0,0), -1)
    
    # for x, y, ww, hh, cos, sin in det[1]:

    #         p1 = tuple(([x,y] + [-ww,-hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
    #         p2 = tuple(([x,y] + [-ww, hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
    #         p3 = tuple(([x,y] + [ ww, hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
    #         p4 = tuple(([x,y] + [ ww,-hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))

    #         p1 = tuple(get_image_point(np.array([-1*(p1[1] - 280)/4, (p1[0] - 160)/4, 0, 1]), K, w2c, e2w))
    #         p2 = tuple(get_image_point(np.array([-1*(p2[1] - 280)/4, (p2[0] - 160)/4,  0, 1]), K, w2c, e2w))
    #         p3 = tuple(get_image_point(np.array([-1*(p3[1] - 280)/4, (p3[0] - 160)/4, 0, 1]), K, w2c, e2w))
    #         p4 = tuple(get_image_point(np.array([-1*(p4[1] - 280)/4, (p4[0] - 160)/4, 0, 1]), K, w2c, e2w))
            
    #         cv2.drawContours(bev, np.array([[p1,p2,p3,p4]]).astype(int), 0, (255,0,0), 2)
    important_vehicle_bb_image_locations = []
    # if len(det) > 0:                
    for v in important_vehicles:
        # print(p1, p2, p3, p4)
        for i, loc in enumerate(v[1]):
            if v[-1] == 1:
                color = (255, 255, 0)
            elif v[-1] == 2:
                color = (0, 255, 0)
            elif v[-1] == 3:
                color = (0, 255, 255)
            elif v[-1] == 4:
                color = (0, 0, 255)
            elif v[-1] == 5:
                color = (0, 0, 255)
            else:
                color = (255, 0, 255)
            loc = np.array([-1*loc[1], loc[0], 3, 1])
            # print(v, loc, get_image_point(loc, K, w2c, e2w).astype(int))
            # print(tuple(get_image_point(loc, K, w2c, e2w).astype(int)))
            bev = cv2.circle(bev, tuple(get_image_point(loc, K, w2c, e2w).astype(int)), 3, color, -1)
        
        for i, loc in enumerate(v[2]):
            if v[3] == 1:
                color = (255, 255, 0)
            elif v[3] == 2:
                color = (0, 255, 0)
            elif v[3] == 3:
                color = (0, 255, 255)
            elif v[3] == 4:
                color = (0, 0, 255)
            else:
                color = (255, 0, 255)
            loc = np.array([-1*loc[1], loc[0], 3, 1])
            bev = cv2.circle(bev, tuple(get_image_point(loc, K, w2c, e2w).astype(int)), 3, color, -1)
    return bev

def get_speed(history_locations):
    locations = []
    for velocity in history_locations:
        locations.append([velocity.x, velocity.y, velocity.z])
    locations = np.array(locations)
    avg_velocities = np.linalg.norm(locations[1:] - locations[:-1], axis = -1).mean() #m/sec
    return avg_velocities

def counterfactual_plan_collide(ego_plan_locs, camera, ego_matrix, ego_vehicle, vehicles, actor_history, dist_threshold_static=1.0, dist_threshold_moving=2.5):
    # TODO: Do a proper occupancy map?
    important_vehicles = []
    modded_trajectories = []

    pixels_per_meter = 4
    brake_speed = 0.4
    other_cast_locs = []
    K = build_projection_matrix()
    w2c = camera.get_transform().get_inverse_matrix()
    e2w = ego_matrix
    ego = get_image_point(np.array([0, 0, 3, 1]), K, w2c, e2w)
    

    pos = np.array([0, 0, 2, 1])
    ego_pos = tuple(get_image_point(pos, K, w2c, ego_vehicle.get_transform().get_matrix()).astype(int))
    vids = []
    selected_vehicles = []
    ego_perturb_important = []
    modded_ego_trajectories = []
    for v in vehicles:
        if v.id == ego_vehicle.id:
            continue
        speed = get_speed(actor_history[v.id])
        if str(speed) == 'nan':
            continue
        waypoint_len = speed*80
        # waypoint_len = 20
        waypoints_y = np.linspace(0, waypoint_len, 20)
        waypoints_x = np.zeros(20)
        wps = np.vstack([waypoints_y, waypoints_x, np.zeros(20), np.ones(20)])
        wps_world = np.dot(v.get_transform().get_matrix(), wps)
        wps_ego = np.dot(np.linalg.inv(e2w), wps_world)
        wps_ego = np.vstack([wps_ego[1, :], -1*wps_ego[0, :]])
        other_cast_locs.append([wps_ego.T])
        vids.append(v.id)
        selected_vehicles.append(v)
        # print(v, speed, wps_ego.T)
    
    perturbed_ego_trajs = []
    for ego_perturb in range(5):
        if ego_perturb == 1:
            perturbed_ego_trajs.append([ego_plan_locs[1], ego_perturb])
        elif ego_perturb == 2:
            # speed up
            dir_vecs = ego_plan_locs[1:] - ego_plan_locs[:-1]
            speed_ratio = 1.25
            mod_traj = [ego_plan_locs[0]]
            for j in range(len(dir_vecs)):
                mod_traj.append(mod_traj[-1] + speed_ratio*dir_vecs[j])
            mod_traj = np.array(mod_traj)
            perturbed_ego_trajs.append([mod_traj, ego_perturb])
        elif ego_perturb == 3:
            #left lane change
            lane_width = 3
            straight_line_length = 10

            x1 = np.linspace(0, straight_line_length/10, 2)
            y1 = np.zeros(2)

            x2 = straight_line_length/10 + np.linspace(0, lane_width, 7)
            y2 = np.linspace(0, -lane_width, 7)

            x3 = straight_line_length/10 + lane_width + np.linspace(0, straight_line_length*2, 13)
            y3 = -lane_width + np.zeros(13)

            x = np.concatenate([x1, x2[1:], x3[1:]])
            y = np.concatenate([y1, y2[1:], y3[1:]])
            mod_traj = np.vstack([y, -x]).T
            perturbed_ego_trajs.append([mod_traj, ego_perturb])

        elif ego_perturb == 4:
            #right lane change
            lane_width = 3
            straight_line_length = 10

            x1 = np.linspace(0, straight_line_length/10, 2)
            y1 = np.zeros(2)

            x2 = straight_line_length/10 + np.linspace(0, lane_width, 7)
            y2 = np.linspace(0, lane_width, 7)

            x3 = straight_line_length/10 + lane_width + np.linspace(0, straight_line_length*2, 13)
            y3 = lane_width + np.zeros(13)

            x = np.concatenate([x1, x2[1:], x3[1:]])
            y = np.concatenate([y1, y2[1:], y3[1:]])
            mod_traj = np.vstack([y, -x]).T
            perturbed_ego_trajs.append([mod_traj, ego_perturb])

        else:
            mod_traj = np.copy(ego_plan_locs)
            perturbed_ego_trajs.append([mod_traj, ego_perturb])

    other_cast_locs = np.array(other_cast_locs)
    for vehicle, other_trajs in enumerate(other_cast_locs):
        init_x, init_y = other_trajs[0,0]
        for other_traj in other_trajs:
            for i in range(6):
                if i == 1:
                    #Full brake # red
                    for epl in perturbed_ego_trajs:
                        # spd = np.linalg.norm(ego_plan_locs[1:]-ego_plan_locs[:-1], axis=-1).mean()
                        dist = np.linalg.norm(epl[0] - other_traj[1], axis = -1).min()
                        intersection_point = np.argmin(np.linalg.norm(epl[0] - other_traj[1], axis = -1))
                        if dist < dist_threshold:
                            if epl[1] == 1:
                                important_vehicles.append([vids[vehicle], [other_traj[1]], [epl[0]], epl[1], i, intersection_point])
                            else:
                                important_vehicles.append([vids[vehicle], [other_traj[1]], epl[0], epl[1], i, intersection_point])
                        if epl[1] == 1:
                            modded_trajectories.append([vids[vehicle], [other_traj[1]], [epl[0]], epl[1], i, intersection_point])
                        else:
                            modded_trajectories.append([vids[vehicle], [other_traj[1]], epl[0], epl[1], i, intersection_point])
                    continue
                elif i == 2:
                    #Speed UP # Green
                    dir_vecs = other_traj[1:] - other_traj[:-1]
                    speed_ratio = 1.25
                    mod_traj = [other_traj[0]]
                    for j in range(len(dir_vecs)):
                        mod_traj.append(mod_traj[-1] + speed_ratio*dir_vecs[j])
                    mod_traj = np.array(mod_traj)
                    

                elif i == 3: # Yellow
                    # Does not change direction, only for turning vehicles(1, 0.78562605, 0)
                    dist_btw_points = np.linalg.norm(other_traj[1:] - other_traj[:-1], axis = -1).mean()
                    dir_vec = other_traj[1] - other_traj[0]
                    mod_traj = [other_traj[0], other_traj[1]]
                    for j in range(1, len(other_traj)-1):
                        mod_traj.append(other_traj[1] + dir_vec*dist_btw_points*j)
                    mod_traj = np.array(mod_traj)

                elif i == 4: 
                    # simulate a lane change 
                    speed = get_speed(actor_history[vids[vehicle]])
                    if speed == 0:
                        lane_width = 0
                    else:
                        lane_width = 3
                    straight_line_length = speed*40
                    # straight_line_length = 10

                    x1 = np.linspace(0, straight_line_length/10, 2)
                    y1 = np.zeros(2)

                    x2 = straight_line_length/10 + np.linspace(0, lane_width, 7)
                    y2 = np.linspace(0, -lane_width, 7)

                    x3 = straight_line_length/10 + lane_width + np.linspace(0, straight_line_length*2, 13)
                    y3 = -lane_width + np.zeros(13)

                    x = np.concatenate([x1, x2[1:], x3[1:]])
                    y = np.concatenate([y1, y2[1:], y3[1:]])

                    mod_traj = np.dot(selected_vehicles[vehicle].get_transform().get_matrix(), np.vstack([x, y, np.zeros(20), np.ones(20)]))
                    mod_traj = np.dot(np.linalg.inv(e2w), mod_traj).T

                    mod_traj = np.vstack([mod_traj[:, 1], -mod_traj[:, 0]]).T
                    # print(mod_traj)

                    # Trajectory tilts left
                    # theta = np.radians(30)                        
                    # rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    # dir_vecs = other_traj[1:] - other_traj[:-1]
                    # mod_traj = [other_traj[0]]
                    # for j in range(len(dir_vecs)):
                    #     rot_vec = np.dot(rot, dir_vecs[j])
                    #     mod_traj.append(mod_traj[-1] + rot_vec)
                    # mod_traj = np.array(mod_traj)

                elif i == 5: 
                    speed = get_speed(actor_history[vids[vehicle]])
                    if speed == 0:
                        lane_width = 0
                    else:
                        lane_width = 3
                    
                    straight_line_length = speed*40
                    # straight_line_length = 10

                    x1 = np.linspace(0, straight_line_length/10, 2)
                    y1 = np.zeros(2)

                    x2 = straight_line_length/10 + np.linspace(0, lane_width, 7)
                    y2 = np.linspace(0, lane_width, 7)

                    x3 = straight_line_length/10 + lane_width + np.linspace(0, straight_line_length*2, 13)
                    y3 = lane_width + np.zeros(13)

                    x = np.concatenate([x1, x2[1:], x3[1:]])
                    y = np.concatenate([y1, y2[1:], y3[1:]])

                    mod_traj = np.dot(selected_vehicles[vehicle].get_transform().get_matrix(), np.vstack([x, y, np.zeros(20), np.ones(20)]))
                    mod_traj = np.dot(np.linalg.inv(e2w), mod_traj).T

                    mod_traj = np.vstack([mod_traj[:, 1], -mod_traj[:, 0]]).T

                    # Trajectory tilts right
                    # theta = np.radians(-30)                        
                    # rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    # dir_vecs = other_traj[1:] - other_traj[:-1]
                    # mod_traj = [other_traj[0]]
                    # for j in range(len(dir_vecs)):
                    #     rot_vec = np.dot(rot, dir_vecs[j])
                    #     mod_traj.append(mod_traj[-1] + rot_vec)
                    # mod_traj = np.array(mod_traj)

                
                else: # pink
                    mod_traj = np.copy(other_traj)
                
                spd = np.linalg.norm(mod_traj[1:]-mod_traj[:-1], axis=-1).mean()
                dist_threshold = dist_threshold_static if spd < brake_speed else dist_threshold_moving
                for epl in perturbed_ego_trajs:
                    dist = np.linalg.norm(mod_traj-epl[0], axis=-1).min() # TODO: outer norm?
                    intersection_point = np.argmin(np.linalg.norm(mod_traj-epl[0], axis=-1))
                    if dist < dist_threshold:
                        if epl[1] == 1:
                            important_vehicles.append([vids[vehicle], mod_traj, [epl[0]], epl[1], i, intersection_point])
                        else:
                            important_vehicles.append([vids[vehicle], mod_traj, epl[0], epl[1], i, intersection_point])
                    if epl[1] == 1:
                        modded_trajectories.append([vids[vehicle], mod_traj, [epl[0]], epl[1], i, intersection_point])
                    else:
                        modded_trajectories.append([vids[vehicle], mod_traj, epl[0], epl[1], i, intersection_point])

        
    return important_vehicles, modded_trajectories

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-s', '--start',
        metavar='S',
        default=0.0,
        type=float,
        help='starting time (default: 0.0)')
    argparser.add_argument(
        '-d', '--duration',
        metavar='D',
        default=0.0,
        type=float,
        help='duration (default: 0.0)')
    argparser.add_argument(
        '-f', '--recorder-filename',
        metavar='F',
        default="test1.log",
        help='recorder filename (test1.log)')
    argparser.add_argument(
        '-c', '--camera',
        metavar='C',
        default=0,
        type=int,
        help='camera follows an actor (ex: 82)')
    argparser.add_argument(
        '-x', '--time-factor',
        metavar='X',
        default=1.0,
        type=float,
        help='time factor (default 1.0)')
    argparser.add_argument(
        '-i', '--ignore-hero',
        action='store_true',
        help='ignore hero vehicles')
    argparser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="./team_code_v2/config.yaml")
    argparser.add_argument("--agent", type=str, help="Path to Agent's file", default="./team_code_v2/lav_agent_cr.py")
    argparser.add_argument("--route-id", type=str, help="Route ID", default="1")
    argparser.add_argument("--plant-agent-config", type=str, help="Path to Agent's configuration file", default="")
    args = argparser.parse_args()


    try:
        
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        # set the time factor for the replayer
        client.set_replayer_time_factor(args.time_factor)

        # set to ignore the hero vehicles or not
        client.set_replayer_ignore_hero(args.ignore_hero)

        # replay the session
        print(client.replay_file(args.recorder_filename, args.start, args.duration, args.camera))
        world = client.get_world()
        new_settings = world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = 1./20 #chck in args
        world.apply_settings(new_settings) 
        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(world)

        # print(CarlaDataProvider)
        world.tick()
        assert(world.get_settings().synchronous_mode)
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        module_agent = importlib.import_module(module_name)
        agent_class_name = getattr(module_agent, 'get_entry_point')()
        agent_instance = getattr(module_agent, agent_class_name)(args.agent_config)
        # config.agent = self.agent_instance

        # plant_agent_class_name = getattr(self.module_plant_agent, 'get_entry_point')()
        # self.plant_agent_instance = getattr(self.module_plant_agent, plant_agent_class_name)(args.plant_agent_config)
        # config.plant_agent = self.plant_agent_instance
        
        # rbe_agent_class_name = getattr(self.module_rbe_agent, 'get_entry_point')()
        # self.rbe_agent_instance = getattr(self.module_rbe_agent, rbe_agent_class_name)(args.plant_agent_config)
        # config.rbe_agent = self.rbe_agent_instance

        other_actors = world.get_actors()
        other_actors_vehicles = other_actors.filter('*vehicle*')
        other_actors_pedestrians = other_actors.filter('*pedestrian*')
        other_actors = list(other_actors_vehicles) + list(other_actors_pedestrians)
        #Route 1 -> 259 12000 -> 12018
        #Route 8 -> 231 10884 -> 10899
        #Route 13 -> 171 7344 -> 7346
        #Route 18 -> 207 33430 -> 33450
        #Route 23 -> 207 32793 ->  32814
        #Route 30 -> 182 19901 -> 19919
        route_id = int(args.route_id)
        ego_id_dict = {1:259, 8:206, 13:171, 18:207, 23:207, 30:182}
        for actor in other_actors:
            print(actor, actor.id, actor.type_id)
            if int(actor.id) == ego_id_dict[int(args.route_id)]:
                ego_vehicle = actor
        _spectator = world.get_spectator()
            
        _agent = AgentWrapper(agent_instance)
        bev_sensor = {'type': 'sensor.camera.rgb', 'x': 0, 'y': 0, 'z': 50, 'roll': 0, 'pitch': -90, 'yaw': 0,
            'width': 1280, 'height': 720, 'fov': 100, 'id': 'BEV_RGB'}
        sensor_list = setup_sensors(ego_vehicle, world, agent_instance, bev_sensor, False)
        step = 0
        _read_route(world, agent_instance, route_id) # this calls self.set_global_plan()
        ego_speed = 0
        actor_history = {}
        while True:
            step+=1 
            print(step)
            ego_trans = ego_vehicle.get_transform()
            _spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),carla.Rotation(pitch=-90)))
            input_data = _agent._agent.sensor_interface.get_data()
            timestamp = GameTime.get_time()
            ego_speed = get_forward_speed( ego_vehicle.get_transform(), ego_vehicle.get_velocity())
            ego_action, ego_plan_locs, og_det, other_cast_locs, other_cast_cmds, important_vehicles, modded_trajectories, rgb = _agent._agent.run_step(input_data, timestamp, ego_speed, step)    
            print(ego_action)
            # print(other_cast_locs)
            # print(ego_plan_locs)
            ego_matrix = np.array(ego_vehicle.get_transform().get_matrix())
            _, spectator_view = input_data.get('BEV_RGB')

            spec_camera = sensor_list[-1]
            
            if os.path.exists("./lidar_point_clouds/" + str(route_id+1)) == False:
                os.mkdir("./lidar_point_clouds/" + str(route_id+1))

            if os.path.exists("./lidar_point_clouds/" + str(route_id+1) + '/recording_data/') == False:
                os.mkdir("./lidar_point_clouds/" + str(route_id+1) + '/recording_data/')
            
            if os.path.exists("./lidar_point_clouds/" + str(route_id+1) + '/recording_data/bev') == False:
                os.mkdir("./lidar_point_clouds/" + str(route_id+1) + '/recording_data/bev')
            
            if os.path.exists("./lidar_point_clouds/" + str(route_id+1) + '/recording_data/bev_projected') == False:
                os.mkdir("./lidar_point_clouds/" + str(route_id+1) + '/recording_data/bev_projected')
            
            if os.path.exists("./lidar_point_clouds/" + str(route_id+1) + '/recording_data/data') == False:
                os.mkdir("./lidar_point_clouds/" + str(route_id+1) + '/recording_data/data')
            
            bev_viz, modded_trajectory_vehicles_bb_locations = vis_bev(spectator_view, ego_matrix, spec_camera, ego_plan_locs, og_det, other_cast_locs, other_cast_cmds, important_vehicles)   
            cv2.imwrite("./lidar_point_clouds/" + str(route_id+1) + "/recording_data/bev/" + str(step) + ".png", bev_viz)
            
            
            vehicle_data = {}
            other_actors = world.get_actors()
            other_actors_vehicles = other_actors.filter('*vehicle*')
            other_actors_pedestrians = other_actors.filter('*pedestrian*')
            other_actors = list(other_actors_vehicles) + list(other_actors_pedestrians)
            for actor in other_actors:
                if actor.id not in actor_history:
                    actor_history[actor.id] = [actor.get_transform().location]
                else:
                    actor_history[actor.id].append(actor.get_transform().location)
                    actor_history[actor.id] = actor_history[actor.id][-5:]

            projected_modded_trajectories = []
            projected_important_vehicles = []
            ego_perturb_important = []
            modded_ego_trajectories = []

            if len(ego_plan_locs) > 0:
                projected_important_vehicles, projected_modded_trajectories = counterfactual_plan_collide(ego_plan_locs, spec_camera, ego_matrix, ego_vehicle, other_actors, actor_history)
            bev_viz = vis_bev_projected(bev_viz, ego_matrix, spec_camera, ego_plan_locs, og_det, other_cast_locs, other_cast_cmds, projected_important_vehicles)   
            cv2.imwrite("./lidar_point_clouds/" + str(route_id+1) + "/recording_data/bev_projected/projected_" + str(step) + ".png", bev_viz)

            for actor in other_actors:
                vehicle_data[actor.id] = [str(actor.type_id), [actor.get_acceleration().x, actor.get_acceleration().y, actor.get_acceleration().z], 
                [actor.get_transform().location.x, actor.get_transform().location.y, actor.get_transform().location.z], 
                [actor.get_transform().rotation.yaw, actor.get_transform().rotation.pitch, actor.get_transform().rotation.roll], 
                [actor.get_velocity().x, actor.get_velocity().y, actor.get_velocity().z], actor.type_id]

            data = np.array([ego_plan_locs, other_cast_locs, other_cast_cmds, modded_trajectories, ego_perturb_important, modded_ego_trajectories, projected_important_vehicles, modded_trajectory_vehicles_bb_locations, important_vehicles, vehicle_data])
            np.save('./lidar_point_clouds/' + str(route_id+1) + '/recording_data/data/' + str(step) + '.npy', data)
            # _, spectator_view = input_data.get('BEV_RGB')
            
 
            # timestamp = GameTime.get_time()

            # if not self._agent._agent.wallclock_t0:
            #     self._agent._agent.wallclock_t0 = GameTime.get_wallclocktime()
            # wallclock = GameTime.get_wallclocktime()
            # wallclock_diff = (wallclock - self._agent._agent.wallclock_t0).total_seconds()

            # # print('======[Agent] Wallclock_time = {} / {} / Sim_time = {} / {}x'.format(wallclock, wallclock_diff, timestamp, timestamp/(wallclock_diff+0.001)))
            world.tick()

    finally:
        pass


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
