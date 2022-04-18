import numpy as np
from gym.envs.registration import register
from highway_env import utils
from highway_env.envs import HighwayEnv, Action
from highway_env.vehicle.controller import ControlledVehicle


class HighwayEnvLocal(HighwayEnv):
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
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 30,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -5,
            ##############yaeli
            "right_lane_reward_1": 5,
            "high_speed_reward_1": 0,
            "lane_change_reward_1": 0,

            "right_lane_reward_2": 0,
            "high_speed_reward_2": 0,
            "lane_change_reward_2": 0,

            "right_lane_reward_3": 0,
            "high_speed_reward_3": 0,
            "lane_change_reward_3": 0,
            "real_time_rendering": True,
            "reward_speed_range": [6, 10],
            "offroad_terminal": False,

        })
        return config

    def _reward(self, action: Action) -> float:
        max_value = max(self.config["lane_change_reward_1"] + self.config["high_speed_reward_1"] +
                        self.config["right_lane_reward_1"],
                        self.config["lane_change_reward_2"] + self.config["high_speed_reward_2"] +
                        self.config["right_lane_reward_2"],
                        self.config["lane_change_reward_3"] + self.config["high_speed_reward_3"] +
                        self.config["right_lane_reward_3"])
        lane_change = action == 0 or action == 2
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle,
                                                               ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"],
                                  [0, 1])
        """reward 1"""
        reward_1 = \
            + self.config["lane_change_reward_1"] * lane_change \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward_1"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward_1"] * np.clip(scaled_speed, 0, 1)

        reward_1 = utils.lmap(reward_1,
                              [self.config["collision_reward"],
                               max_value],
                              # self.config["lane_change_reward_1"] + self.config[
                              #    "high_speed_reward_1"] + self.config[
                              #   "right_lane_reward_1"]],
                              [0, 1])
        reward_1 = 0 if not self.vehicle.on_road else reward_1

        """reward 2"""
        reward_2 = \
            + self.config["lane_change_reward_2"] * lane_change \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward_2"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward_2"] * np.clip(scaled_speed, 0, 1)

        reward_2 = utils.lmap(reward_2,
                              [self.config["collision_reward"],
                               max_value],
                              #   self.config["lane_change_reward_2"] + self.config[
                              #      "high_speed_reward_2"] + self.config["right_lane_reward_2"]],
                              [0, 1])
        reward_2 = 0 if not self.vehicle.on_road else reward_2

        """reward 3"""
        reward_3 = \
            + self.config["lane_change_reward_3"] * lane_change \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward_3"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward_3"] * np.clip(scaled_speed, 0, 1)
        reward_3 = utils.lmap(reward_3,
                              [self.config["collision_reward"],
                               max_value],
                              # self.config["lane_change_reward_3"] + self.config[
                              #   "high_speed_reward_3"] + self.config[
                              #  "right_lane_reward_3"]],
                              [0, 1])
        reward_3 = 0 if not self.vehicle.on_road else reward_3

        return np.array([reward_1, reward_2, reward_3])

register(
    id='highway_local-v0',
    #entry_point='highway_env_local.envs.highway_env_local:HighwayEnvLocal',
    entry_point='multi_head.highway_env_local.envs.highway_env_local:HighwayEnvLocal',
)