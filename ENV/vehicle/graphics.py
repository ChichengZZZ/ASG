from __future__ import division, print_function

import itertools

import numpy as np
import pygame

from NGSIM_env.vehicle.dynamics import Vehicle, Obstacle
from NGSIM_env.vehicle.control import ControlledVehicle, MDPVehicle
from NGSIM_env.vehicle.behavior import IDMVehicle, LinearVehicle
from NGSIM_env.vehicle.humandriving import NGSIMVehicle, HumanLikeVehicle, InterActionVehicle, IntersectionHumanLikeVehicle

class VehicleGraphics(object):
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    @classmethod
    def display(cls, vehicle, surface, his_length=4, his_width=2, transparent=False, offscreen=False):
        """
        在pygame表面上显示一个车辆。

        车辆以彩色的旋转矩形表示。

        :param vehicle: 要绘制的车辆
        :param surface: 要在上面绘制车辆的表面
        :param transparent: 是否应该将车辆绘制为轻微透明
        :param offscreen: 是否应该进行离屏渲染

        """
        v = vehicle
        if v.LENGTH:
            s = pygame.Surface((surface.pix(v.LENGTH), surface.pix(v.LENGTH)), pygame.SRCALPHA)  # per-pixel alpha
            rect = pygame.Rect(0, surface.pix(v.LENGTH)/2 - surface.pix(v.WIDTH)/2, surface.pix(v.LENGTH), surface.pix(v.WIDTH))
        else:
            v_length, v_width = his_length, his_width
            s = pygame.Surface((surface.pix(v_length), surface.pix(v_length)), pygame.SRCALPHA)  # per-pixel alpha
            rect = pygame.Rect(0, surface.pix(v_length)/2 - surface.pix(v_width)/2, surface.pix(v_length), surface.pix(v_width))

        pygame.draw.rect(s, cls.get_color(v, transparent), rect, width=0, border_radius=5)
        # font = pygame.font.SysFont('Times New Roman', 15)
        # textSurface = font.render(f'{vehicle.vehicle_ID}', True, (255, 0, 0))


        #pygame.draw.rect(s, cls.BLACK, rect, 1, border_radius=1)
        if not offscreen:  # convert_alpha throws errors in offscreen mode TODO() Explain why
            try:
                s = pygame.Surface.convert_alpha(s)
            except Exception as e:
                print(e)
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        sr = pygame.transform.rotate(s, -h * 180 / np.pi)

        if v.LENGTH:
            surface.blit(sr, (surface.pos2pix(v.position[0] - v.LENGTH/2, v.position[1] - v.LENGTH/2)))
            # surface.blit(textSurface, (surface.pos2pix(v.position[0] - v.LENGTH/2, v.position[1] - v.LENGTH/2)))
        else:
            surface.blit(sr, (surface.pos2pix(v.position[0] - v_length/2, v.position[1] - v_length/2)))
            # surface.blit(textSurface, (surface.pos2pix(v.position[0] - v_length/2, v.position[1] - v_length/2)))
        # try:
        #     # if vehicle.is_ego and len(ls_points) > 1:
        #     # if vehicle.is_ego:
        # ls_points = [surface.vec2pix(tuple(pt)) for pt in vehicle.planned_trajectory]
        # pygame.draw.aalines(surface, "GREEN", False, ls_points)
        # elif len(ls_points) > 1:
            # pygame.draw.aalines(surface, (100, 200, 255), False, ls_points)
        # except:
        #     print(vehicle.vehicle_ID)


    @classmethod
    def display_trajectory(cls, states, surface, offscreen=False):
        """
        在pygame的界面上显示车辆的整个轨迹。

        :param states: 要显示的车辆状态列表
        :param surface: 用于绘制车辆未来状态的界面
        :param offscreen: 是否应该在屏幕外进行渲染
        """

        for vehicle in states:
            cls.display(vehicle, surface, transparent=True, offscreen=offscreen)
    
    @classmethod
    def display_NGSIM_trajectory(cls, surface, vehicle):
        if isinstance(vehicle, NGSIMVehicle) and not vehicle.overtaken and vehicle.appear:
            trajectories = vehicle.ngsim_traj[vehicle.sim_steps:vehicle.sim_steps+50, 0:2]
            points = []
            for i in range(trajectories.shape[0]):
                if trajectories[i][0] >= 0.1:
                    point = surface.pos2pix(trajectories[i][0], trajectories[i][1]) 
                    pygame.draw.circle(surface, cls.GREEN, point, 2)
                    points.append(point)
                else:
                    break
            if len(points) >= 2:
                pygame.draw.lines(surface, cls.GREEN, False, points)

    @classmethod
    def display_planned_trajectory(cls, surface, vehicle):
        if isinstance(vehicle, HumanLikeVehicle) and not vehicle.human:
            trajectory = vehicle.planned_trajectory
            points = []
            for i in range(trajectory.shape[0]):
                point = surface.pos2pix(trajectory[i][0], trajectory[i][1]) 
                pygame.draw.circle(surface, cls.RED, point, 2)
                points.append(point)
            pygame.draw.lines(surface, cls.RED, False, points) 

    @classmethod
    def display_history(cls, vehicle, surface, frequency=3, duration=3, simulation=10, offscreen=False):
        """
        在pygame的界面上显示车辆的整个轨迹。

        :param vehicle: 要显示的车辆状态
        :param surface: 用于绘制车辆未来状态的界面
        :param frequency: 历史中显示位置的频率
        :param duration: 显示历史的长度
        :param simulation: 模拟频率
        :param offscreen: 是否应该在屏幕外进行渲染 """

        for v in itertools.islice(vehicle.history, None, int(simulation * duration), int(simulation / frequency)):
            cls.display(v, surface, vehicle.LENGTH, vehicle.WIDTH, transparent=True, offscreen=offscreen)

    @classmethod
    def get_color(cls, vehicle, transparent=False):
        color = cls.DEFAULT_COLOR
        if getattr(vehicle, "color", None) and not vehicle.crashed:
            color = vehicle.color
        elif vehicle.crashed:
            color = cls.RED
        elif isinstance(vehicle, NGSIMVehicle):
            color = cls.BLUE
        # elif isinstance(vehicle, NGSIMVehicle) and not vehicle.overtaken:
        #     color = cls.BLUE
        elif isinstance(vehicle, InterActionVehicle):
            color = cls.BLUE
        elif isinstance(vehicle, HumanLikeVehicle):
            color = cls.BLUE
        elif isinstance(vehicle, IDMVehicle):
            color = cls.BLUE
        elif isinstance(vehicle, Obstacle):
            color = cls.GREEN
        elif isinstance(vehicle, InterActionVehicle):
            color = cls.GREEN

        if vehicle.is_ego and not hasattr(vehicle, 'color'):
            color = cls.EGO_COLOR

        if transparent:
            color = (color[0], color[1], color[2], 30)
        return color

    @classmethod
    def handle_event(cls, vehicle, event):
        """
        根据车辆类型处理一个pygame事件

        :param vehicle: 接收事件的车辆
        :param event: pygame事件
        """

        if isinstance(vehicle, ControlledVehicle):
            cls.control_event(vehicle, event)
        if isinstance(vehicle, Vehicle):
            cls.dynamics_event(vehicle, event)

    @classmethod
    def control_event(cls, vehicle, event):
        """
        将pygame键盘事件映射到控制决策

        :param vehicle: 接收事件的车辆
        :param event: pygame事件 """

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                vehicle.act("FASTER")
            if event.key == pygame.K_LEFT:
                vehicle.act("SLOWER")
            if event.key == pygame.K_DOWN:
                vehicle.act("LANE_RIGHT")
            if event.key == pygame.K_UP:
                vehicle.act("LANE_LEFT")

    @classmethod
    def dynamics_event(cls, vehicle, event):
        """
        将pygame键盘事件映射到动力学激励

        :param vehicle: 接收事件的车辆
        :param event: pygame事件 """

        action = vehicle.action.copy()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                action['steering'] = 45 * np.pi / 180
            if event.key == pygame.K_LEFT:
                action['steering'] = -45 * np.pi / 180
            if event.key == pygame.K_DOWN:
                action['acceleration'] = -6
            if event.key == pygame.K_UP:
                action['acceleration'] = 5
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                action['steering'] = 0
            if event.key == pygame.K_LEFT:
                action['steering'] = 0
            if event.key == pygame.K_DOWN:
                action['acceleration'] = 0
            if event.key == pygame.K_UP:
                action['acceleration'] = 0
        if action != vehicle.action:
            vehicle.act(action)
