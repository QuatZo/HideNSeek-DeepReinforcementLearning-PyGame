import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces.utils import flatten_space, flatten

import pygame
import math
import copy
import random
import sys
import os
import numpy as np

from game_env.hidenseek_gym.controllable import Hiding, Seeker
from game_env.hidenseek_gym.fixed import Wall
from game_env.hidenseek_gym.supportive import Point, Collision


class HideNSeekEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    def __init__(self, config, width, height, seeker, hiding, walls):
        self.default_cfg = config

        self.map_path = config['game']['map']
        self.fps = config['game']['fps']
        self.clock = pygame.time.Clock()
        self.screen = None

        self.dt = self.clock.tick_busy_loop(self.fps)
        self.cfg = config['game']
        self.duration = config['game']['duration']

        self.width = width
        self.height = height

        self.walls_group = pygame.sprite.Group()
        self.env_walls = walls
        self.walls_group.add(walls)

        self.player_seek = seeker
        self.player_hide = hiding
        self.players_group = pygame.sprite.Group()
        self.players_group.add(self.player_seek)
        self.players_group.add(self.player_hide)

        self.screen_lite = pygame.Surface((self.width, self.height))
        if self.walls_group:
            for wall in walls:
                wall_p = [(p.x, p.y) for p in wall.get_abs_vertices()]
                _ = [pygame.draw.polygon(
                    self.screen_lite, (255, 255, 255), wall_p)]

        self.p_hide_cfg = config['hiding']
        self.p_seek_cfg = config['seeker']
        self.agent_env = {}
        self.action_space = spaces.Discrete(6)  # for both agents
        '''
        0 - NOOP 
        1 - FORWARD MOVEMENT
        2 - BACKWARD MOVEMENT
        3 - ROTATE RIGHT (clockwise)
        4 - ROTATE LEFT (counter-clockwise)
        5 - SPECIAL (ADD/DELETE WALL)
        '''

        self.observation_space_n = [
            spaces.Box(low=0, high=1, shape=(self.width, self.height)),
            spaces.Box(low=0, high=1, shape=(self.width, self.height)),
        ]

        self.flatten_observation_space_n = [flatten_space(
            space) for space in self.observation_space_n]

    def reset(self):
        self.duration = self.cfg['duration']
        self.screen = None
        self.agent_env = {}

        self.walls_group = pygame.sprite.Group()
        self.walls_group.add(self.env_walls)

        self.player_seek.reset()
        self.player_hide.reset()

        self.player_seek.update_vision({'walls': [], 'enemy': None, })
        self.player_hide.update_vision({'walls': [], 'enemy': None, })

        self._calc_local_env()

        self.player_seek.update_vision(self.agent_env['p_seek'])
        self.player_hide.update_vision(self.agent_env['p_hide'])

        self.players_group = pygame.sprite.Group()
        self.players_group.add(self.player_seek)
        self.players_group.add(self.player_hide)

        return [
            self._get_agent_obs(self.player_seek, self.agent_env['p_seek']),
            self._get_agent_obs(self.player_hide, self.agent_env['p_hide'])
        ]

    def game_over(self):
        if self.duration <= 0:
            return True, "HIDING"

        if Collision.aabb(self.player_seek.pos, (self.player_seek.width, self.player_seek.height), self.player_hide.pos, (self.player_hide.width, self.player_hide.height)):
            if Collision.sat(self.player_seek.get_abs_vertices(), self.player_hide.get_abs_vertices()):
                return True, "SEEKER"
        return False, None

    def _can_create_wall(self, wall, enemy):
        # check if dynamically created POV lines are shorter than eyesight -- if yes, then it's not possible to create a Wall
        local_wall_edges = self.player_hide.reduce_wall_edges(
            self.agent_env['p_hide']['walls'])
        wall_vertices = wall.get_abs_vertices()
        wall_edges = [wall_vertices[0], wall.pos,
                      wall_vertices[3]]  # only closer edges & center

        vision_ray_points = [[self.player_hide.pos, wall_edge]
                             for wall_edge in wall_edges] + [[self.player_hide.pos, self.player_hide.vision_top]]
        for ray in vision_ray_points:
            ray_dist = ray[0].distance(ray[1])
            for local_wall_edge in local_wall_edges:
                p = Collision.line_intersection(ray, local_wall_edge)
                if p and p.distance(ray[0]) < ray_dist:
                    return False

        for _wall in self.agent_env['p_hide']['walls']:
            if Collision.aabb(wall.pos, (wall.width, wall.height), _wall.pos, (_wall.width, _wall.height)):
                if Collision.sat(wall.get_abs_vertices(), _wall.get_abs_vertices()):
                    return False

        if enemy and Collision.aabb(enemy.pos, (enemy.width, enemy.height), wall.pos, (wall.width, wall.height)):
            if Collision.sat(self.player_hide.get_abs_vertices(), enemy.get_abs_vertices()):
                return False
        return True

    def _add_wall(self):
        if self.player_hide.walls_counter < self.player_hide.walls_max and not self.player_hide.wall_timer:
            wall_pos = copy.deepcopy(self.player_hide.pos)
            wall_size = (max(int(self.player_hide.width / 10), 2),
                         max(int(self.player_hide.height / 2), 2))  # minimum 2x2 Wall
            vision_arc_range = np.sqrt((self.player_hide.vision_top.x - self.player_hide.pos.x) * (self.player_hide.vision_top.x - self.player_hide.pos.x) + (
                self.player_hide.vision_top.y - self.player_hide.pos.y) * (self.player_hide.vision_top.y - self.player_hide.pos.y))
            # vision arc range - 1.5 wall width, so the wall is always created inside PoV.
            wall_pos.x = wall_pos.x + vision_arc_range - \
                (1.5 * wall_size[0])
            wall_pos = Point.triangle_unit_circle_relative(
                self.player_hide.direction, self.player_hide.pos, wall_pos)

            wall = Wall(self.player_hide, wall_pos.x,
                        wall_pos.y, wall_size, self.cfg['graphics_path_wall_owner'])
            wall._rotate(self.player_hide.direction, wall_pos)
            if self._can_create_wall(wall, self.agent_env['p_hide']['enemy']):
                self.player_hide.walls_counter += 1
                self.walls_group.add(wall)
                self.player_hide.wall_timer = copy.deepcopy(
                    self.player_hide.wall_timer_init)
                return True
            else:
                del wall

        return False

    def _remove_wall(self):
        if self.agent_env['p_seek']['walls'] and not self.player_seek.wall_timer:
            # remove randomly selected wall in local env
            delete_wall = random.choice(self.agent_env['p_seek']['walls'])
            self.player_seek.wall_timer = self.player_seek.wall_timer_init
            if delete_wall.owner:
                delete_wall.owner.walls_counter -= 1
                self.walls_group.remove(delete_wall)
                del delete_wall
                return True

        return False

    def _reduce_agent_cooldown(self, agent):
        if agent.wall_timer > 0:
            agent.wall_timer -= 1
        # for negative it's 0, for positive - higher than 0, needed if time-based cooldown (i.e. 5s) instead of frame-based (i.e. 500 frames)
        agent.wall_timer = max(agent.wall_timer, 0)

    def _calc_local_env(self):
        self.agent_env['p_seek'] = {
            'walls': Collision.get_objects_in_local_env(self.walls_group, self.player_seek.pos, self.player_seek.vision_radius, self.player_seek.direction, self.player_seek.ray_objects),
            'enemy': self.player_hide if Collision.get_objects_in_local_env([self.player_hide], self.player_seek.pos, self.player_seek.vision_radius, self.player_seek.direction, self.player_seek.ray_objects) else None,
        }
        self.agent_env['p_hide'] = {
            'walls': Collision.get_objects_in_local_env(self.walls_group, self.player_hide.pos, self.player_hide.vision_radius, self.player_hide.direction, self.player_hide.ray_objects),
            'enemy': self.player_seek if Collision.get_objects_in_local_env([self.player_seek], self.player_hide.pos, self.player_hide.vision_radius, self.player_hide.direction, self.player_hide.ray_objects) else None,
        }

    def _get_agent_obs(self, agent, local_env):
        return flatten(self.observation_space_n[0], self._draw_obs(agent, local_env))

    def _rotate_agent(self, agent, turn):
        """
        Rotates the object, accordingly to the value, along its axis.

        Parameters
        ----------
            agent : object
            turn : int, [-1,1]
                in which direction should agent rotate (clockwise or counterclockwise)
            local_env : dict
                contains Player Local Environment

        Returns
        -------
            None
        """
        agent.image_index = 0
        agent.direction += agent.speed_rotate * turn
        agent.direction = agent.direction % (2 * math.pi)
        return True

    def _calc_action_reward(self, agent, action, success=True):
        agent_str = str(agent)[1:-1].lower()  # hiding, seeker
        if action == 0:
            reward = self.default_cfg[agent_str]['rewards']['noop']
        elif action in [1, 2]:
            reward = self.default_cfg[agent_str]['rewards']['move']
        elif action in [3, 4]:
            reward = self.default_cfg[agent_str]['rewards']['rotate']
        elif action == 5:
            reward = self.default_cfg[agent_str]['rewards']['special']

        return reward

    def _perform_agent_action(self, agent, action, local_env):
        if action == 0:
            '''
            agent.image_index = 0
            agent.image = agent.images[agent.image_index]
            '''
            return self._calc_action_reward(agent, action)
        elif action in [1, 2]:
            # (1 - 1.5) * 2 = -1, so for Forward it needs to be * (-1)
            x = math.cos(agent.direction) * agent.speed * \
                (action - 1.5) * 2 * (-1)
            # (1 - 1.5) * 2 = -1, so for Forward it needs to be * (-1)
            y = math.sin(agent.direction) * agent.speed * \
                (action - 1.5) * 2 * (-1)
            old_pos = copy.deepcopy(agent.pos)
            new_pos = agent.pos + Point((x, y))

            self._move_agent(agent, new_pos)

            # outside game window
            if (
                new_pos.x < 0
                or new_pos.x > self.width
                or new_pos.y < 0
                or new_pos.y > self.height
            ):
                self._move_agent(agent, old_pos)
                return self._calc_action_reward(agent, action, success=False)

            for wall in local_env['walls']:
                if Collision.aabb(new_pos, (agent.width, agent.height), wall.pos, (wall.width, wall.height)):
                    if Collision.sat(agent.get_abs_vertices(), wall.get_abs_vertices()):
                        self._move_agent(agent, old_pos)
                        return self._calc_action_reward(agent, action, success=False)
            return self._calc_action_reward(agent, action)
        elif action in [3, 4]:
            # (3 - 3.5) * 2 = -1, so for Clockwise Rotate it needs to be * (-1)
            did_rotate = self._rotate_agent(agent, (action - 3.5) * 2 * (-1))
            return self._calc_action_reward(agent, action, success=did_rotate)
        elif action == 5:
            if isinstance(agent, Seeker):
                did_remove = self._remove_wall()
                return self._calc_action_reward(agent, action, success=did_remove)
            else:
                did_add = self._add_wall()
                return self._calc_action_reward(agent, action, success=did_add)

        raise Exception(
            f"Unknown action, available action space: {self.action_space}")

    def _move_agent(self, agent, new_pos):
        """
        Algorithm which moves the Player object to given direction, if not outside map (game screen)

        Parameters
        ----------
            new_pos : hidenseek.ext.supportive.Point
                Point object of the new position

        Returns
        -------
            None
        """

        old_pos = copy.deepcopy(agent.pos)
        agent.pos = new_pos

        if old_pos != agent.pos:  # if moving
            agent.image_index = (agent.image_index + 1) % len(agent.sprites)
            if not agent.image_index:
                agent.image_index += 1
            agent.rect.center = (agent.pos.x, agent.pos.y)
        else:  # if not moving
            agent.image_index = 0

        agent.image = agent.sprites[agent.image_index]

    def step(self, action_n):
        obs_n = list()
        reward_n = list()
        info_n = {'n': []}

        self.dt = self.clock.tick_busy_loop(self.fps)

        self._reduce_agent_cooldown(self.player_seek)
        self._reduce_agent_cooldown(self.player_hide)

        if self.cfg['reverse']:
            reward_hiding = self._perform_agent_action(
                self.player_hide, action_n[1], self.agent_env['p_hide'])
            reward_seeker = self._perform_agent_action(
                self.player_seek, action_n[0], self.agent_env['p_seek'])
        else:
            reward_seeker = self._perform_agent_action(
                self.player_seek, action_n[0], self.agent_env['p_seek'])
            reward_hiding = self._perform_agent_action(
                self.player_hide, action_n[1], self.agent_env['p_hide'])

        reward_n = [
            reward_seeker,
            reward_hiding,
        ]

        self._calc_local_env()

        self.player_seek.update_vision(self.agent_env['p_seek'])
        self.player_hide.update_vision(self.agent_env['p_hide'])

        done = self.game_over()

        obs_n = [
            self._get_agent_obs(self.player_seek, self.agent_env['p_seek']),
            self._get_agent_obs(self.player_hide, self.agent_env['p_hide'])

        ]

        # End Game Rewards
        if done[0]:
            endscore = ['lose', 'win']
            if self.cfg['continuous_reward']:
                score = [
                    -max(self.default_cfg['seeker']['rewards'][endscore[0]],
                         self.default_cfg['game']['duration'] / 2),
                    max(self.default_cfg['hiding']['rewards'][endscore[1]],
                        self.default_cfg['game']['duration'] / 2)
                ]
            else:
                score = [
                    -max(self.default_cfg['seeker']['rewards'][endscore[0]],
                         self.default_cfg['game']['duration'] / 2),
                    max(self.default_cfg['hiding']['rewards'][endscore[1]],
                        self.default_cfg['game']['duration'] / 2)
                ]

            if done[1] == 'SEEKER':
                endscore = endscore[::-1]
                if self.cfg['continuous_reward']:
                    score = [
                        self.default_cfg['seeker']['rewards'][endscore[0]] +
                        self.default_cfg['game']['duration'] - self.duration,
                        -(self.default_cfg['hiding']['rewards'][endscore[1]] +
                          self.default_cfg['game']['duration'] - self.duration)
                    ]
                else:
                    score = [
                        max(self.default_cfg['seeker']['rewards'][endscore[0]],
                            self.default_cfg['game']['duration'] / 2),
                        -max(self.default_cfg['hiding']['rewards'][endscore[1]],
                             self.default_cfg['game']['duration'] / 2)
                    ]
            reward_n = [reward_n[i] + score[i] for i in range(len(score))]
        self.duration -= 1

        return obs_n, reward_n, done, info_n

    def _get_state(self):
        state = np.fliplr(np.rot90(pygame.surfarray.pixels3d(
            pygame.display.get_surface()).astype(np.uint8), 3))
        return state

    def _draw_agent_vision(self, agent, screen):
        pygame.draw.line(screen, (0, 255, 0), (agent.pos.x, agent.pos.y),
                         (agent.vision_top.x, agent.vision_top.y), 1)
        ray_obj = agent.ray_points  # without square object
        for obj in ray_obj:
            pygame.draw.line(screen,
                             (255, 85, 55),
                             (agent.pos.x, agent.pos.y),
                             (obj.x, obj.y)
                             )

    def _draw_agent(self, agent, screen):
        """
        Function used only in HideNSeek class. Draws Agent POV on given Screen

        Parameters
        ----------
            agent : hidenseek.objects.controllable.Player
                agent instance, may be Player, Hiding or Seeker
            screen : pygame.Display
                game window

        Returns
        -------
            None
        """
        # Copy and then rotate the original image.
        copied_sprite = agent.sprites[agent.image_index].copy()
        copied_sprite = pygame.transform.scale(
            copied_sprite, (agent.width, agent.height))
        copied_sprite = pygame.transform.rotate(
            copied_sprite, -agent.direction * 180 / math.pi)

        copied_sprite_rect = copied_sprite.get_rect()
        copied_sprite_rect.center = (agent.pos.x, agent.pos.y)
        screen.blit(copied_sprite, copied_sprite_rect)

        agent.image = pygame.Surface((agent.width, agent.height))
        agent.image.set_colorkey((0, 0, 0))

    def _draw_obs(self, agent, local_env):
        # 0 - background
        # 63 - agent
        # 127 - enemy
        # 191 - enemy walls
        # 255 - wall
        surf_temp = self.screen_lite.copy()

        agent_v = [(v.x, v.y) for v in agent.get_abs_vertices()]
        _ = pygame.draw.polygon(surf_temp, (63, 63, 63), agent_v)
        if 'enemy' in local_env and local_env['enemy']:
            enemy_v = [(v.x, v.y)
                       for v in local_env['enemy'].get_abs_vertices()]
            _ = pygame.draw.polygon(surf_temp, (127, 127, 127), enemy_v)

        if 'walls' in local_env and local_env['walls']:
            for wall in local_env['walls']:
                wall_v = [(w.x, w.y) for w in wall.get_abs_vertices()]
                _ = pygame.draw.polygon(surf_temp, (191, 191, 191), wall_v)

        return np.fliplr(np.rot90(pygame.surfarray.pixels_red(surf_temp).astype(np.uint8), 3)) / 255.0

    def render(self, mode='human', close=False):
        """
        Renders game based on the mode. Raises Exception if unexpected render mode.

        Parameters
        ----------
            mode : string
                mode in which game should be rendered (graphic, console, rgb_array)            
            close : boolean
                whether pygame instance should be shutdown

        Returns
        -------
            None
        """
        if mode == 'rgb_array':
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        if mode == 'human' or mode == 'rgb_array':
            if close:
                pygame.quit()
                return

            if not self.screen:
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.width, self.height), 0, 32)

            self.screen.fill((0, 0, 0))
            if self.walls_group:
                self.walls_group.draw(self.screen)

            if self.player_hide and self.player_seek:
                if self.default_cfg['video']['draw_pov']:
                    self._draw_agent_vision(self.player_seek, self.screen)
                    self._draw_agent_vision(self.player_hide, self.screen)
                self._draw_agent(self.player_hide, self.screen)
                self._draw_agent(self.player_seek, self.screen)

            if self.players_group:
                self.players_group.draw(self.screen)

            pygame.display.update()
            img = self._get_state()
            return img
        elif mode == 'console':
            pass
        else:
            raise Exception(
                "Unexpected render mode, available: 'human', 'console', 'rgb_array'")
