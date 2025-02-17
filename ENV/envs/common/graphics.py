from __future__ import division, print_function, absolute_import

import os

import numpy as np
import pygame
from gym.spaces import Discrete

from NGSIM_env.road.graphics import WorldSurface, RoadGraphics
from NGSIM_env.vehicle.graphics import VehicleGraphics


class EnvViewer(object):
    """
    一个用于渲染高速公路驾驶环境的观察者。
    """
    SAVE_IMAGES = False

    def __init__(self, env, offscreen=False):
        self.env = env
        self.offscreen = offscreen

        pygame.init()
        pygame.display.set_caption("NGSIM-env")
        panel_size = (self.env.config["screen_width"], self.env.config["screen_height"])

        # 不需要显示器来绘制物体。忽略display.set_mode()指令可以在没有处理屏幕显示的情况下在表面上进行绘制，适用于云计算等场景。
        if not self.offscreen:
            self.screen = pygame.display.set_mode([self.env.config["screen_width"], self.env.config["screen_height"]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = env.config.get("scaling", self.sim_surface.INITIAL_SCALING)
        self.sim_surface.centering_position = env.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()

        self.enabled = True
        if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0

    def set_agent_display(self, agent_display):
        """
            设置由代理提供的显示回调函数，以便它们可以在专用的代理表面上显示其行为，甚至可以在模拟表面上显示。
            ：param agent_display：由代理提供的在表面上显示的回调函数。
        """
        if self.agent_display is None:
            if self.env.config["screen_width"] > self.env.config["screen_height"]:
                self.screen = pygame.display.set_mode((self.env.config["screen_width"], 2 * self.env.config["screen_height"]))
            else:
                self.screen = pygame.display.set_mode((2 * self.env.config["screen_width"], self.env.config["screen_height"]))

        # wjc
        self.agent_surface = pygame.Surface((self.env.config["screen_width"], self.env.config["screen_height"]))

        self.agent_display = agent_display

    def set_agent_action_sequence(self, actions):
        """
        设置代理选择的动作序列，以便可以显示出来。
        :param actions: 动作列表，遵循环境的动作空间规范。
        """
        return
        if isinstance(self.env.action_space, Discrete):
            actions = [self.env.ACTIONS[a] for a in actions]
        if len(actions) > 1:
            self.vehicle_trajectory = self.env.vehicle.predict_trajectory(actions,
                                                                          1 / self.env.config["policy_frequency"],
                                                                          1 / 3 / self.env.config["policy_frequency"],
                                                                          1 / self.env.SIMULATION_FREQUENCY)

    def handle_events(self):
        """
        通过将 pygame 事件转发到显示器和环境车辆来处理。
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.vehicle:
                VehicleGraphics.handle_event(self.env.vehicle, event)

    def display(self):
        """
        在 pygame 窗口上显示道路和车辆。
        """
        if not self.enabled:
            return
        info = pygame.display.Info()
        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency = self.env.SIMULATION_FREQUENCY, offscreen=self.offscreen)

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if self.env.config["screen_width"] > self.env.config["screen_height"]:
                self.screen.blit(self.agent_surface, (0, self.env.config["screen_height"]))
            else:
                self.screen.blit(self.agent_surface, (self.env.config["screen_width"], 0))

        if not self.offscreen:
            # wjc
            try:
                self.screen.blit(self.sim_surface, (0, 0))
                self.clock.tick(self.env.SIMULATION_FREQUENCY)
                pygame.display.flip()
            except Exception as e:
                print(e)
        if self.SAVE_IMAGES:
            pygame.image.save(self.screen, "Pics/ngsim_highway_env_{}.png".format(self.frame))
            self.frame += 1

    def get_image(self):
        """
        :return：渲染的图像作为 RGB 数组返回。
        """
        surface = self.screen if not self.offscreen else self.sim_surface
        data = pygame.surfarray.array3d(surface)
        return np.moveaxis(data, 0, 1)

    def window_position(self):
        """
        :return: 显示窗口中心的世界位置。
        """
        if self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self):
        """
        关闭 pygame 窗口。
        """
        pygame.quit()
