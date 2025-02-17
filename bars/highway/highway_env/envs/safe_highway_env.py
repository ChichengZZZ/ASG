import math
import random
from typing import Tuple, Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class SafeHighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "TrafficFlow",
                "vehicles_count": 11,
                # "features": ["x", "y", "vx", "vy", "heading"],
                "features": ["x", "y", "vx", "vy", "heading", 'lat_off', 'ang_off'],
                # "features_range": {
                #     "x": [-100, 100],
                #     "y": [-100, 100],
                #     "vx": [-20, 20],
                #     "vy": [-20, 20],
                # },
                "absolute": False,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                # "type": "DiscreteMetaAction",
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 3, np.pi / 3],
                # "longitudinal": True,
                # "lateral": True,
                # "dynamical": True,
            },
            "lanes_count": 4,
            "simulation_frequency": 10,  # [Hz]
            "policy_frequency": 10,  # [Hz]
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 100,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -2,    # The reward received when colliding with a vehicle.
            "on_road_reward": -1,
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [5, 15],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        vehicles_count = random.choice(range(20, 50))
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(vehicles_count, num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=random.choice(range(5, 15)),
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"], speed=random.choice(range(5, 15)),)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        reward = 0.0
        forward_speed = self.vehicle.speed
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward += self.config['high_speed_reward'] * scaled_speed
        av_lane = self.road.network.get_lane(self.vehicle.lane_index)
        av_lon, lat = av_lane.local_coordinates(self.vehicle.position)
        av_angle = av_lane.local_angle(self.vehicle.heading, av_lon)

        if abs(av_angle) >= math.pi / 6:
            reward = -(lat / av_lane.width)**2
        else:
            reward += (1 - (lat / av_lane.width) ** 2) * 0.4
        cost = self._cost()
        if cost >= 45:
            reward = -((cost - 45) / 5)
        if self.vehicle.crashed or not self.vehicle.on_road:
            reward = self.config['collision_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(not self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _cost(self):
        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        front_ttc = 50
        if front_vehicle is not None:  # 计算与前车的ttc
            av_lane = self.road.network.get_lane(self.vehicle.lane_index)
            av_lon, _ = av_lane.local_coordinates(self.vehicle.position)
            front_veh_lon, _ = av_lane.local_coordinates(front_vehicle.position)
            av_angle = av_lane.local_angle(self.vehicle.heading, av_lon)
            front_veh_angle = av_lane.local_angle(front_vehicle.heading, front_veh_lon)

            bv_speed = front_vehicle.speed * np.cos(front_veh_angle)
            av_speed = self.vehicle.speed * np.cos(av_angle)
            if bv_speed < av_speed:
                front_ttc = min(front_ttc, (front_veh_lon - av_lon) / utils.not_zero(av_speed - bv_speed))

        return 50 - front_ttc

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]

        self._simulate(action)

        obs = self.observation_type.observe()
        # print(self.vehicle.position)
        # print(self.vehicle.heading)
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        info['cost'] = self._cost()
        info['ttc'] = 50 - info['cost']
        info['acceleration'] = self.vehicle.action['acceleration']
        info['steering'] = self.vehicle.action['steering']
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info