import math
import os
import pygame
import copy
from game_env.hidenseek_gym.supportive import Point


class Wall(pygame.sprite.Sprite):
    """
    Wall Class for Hide'n'Seek Game, inherits from pygame.sprite.Sprite

    Attributes
    ----------
        owner : None, hidenseek.objects.controllable.Hiding, hidenseek.objects.controllable.Seeker
            Wall owner, None for game environment
        width : int
            width of the Wall
        height : int
            height of the Wall
        pos : hidenseek.ext.supportive.Point
            object position on the game display
        image : pygame.Surface
            object image surface on which image/shape will be drawn
        rect : pygame.Rect
            object Rectangle, to be drawn
        polygon_points : list of tuples
            vertices, used for collision check in SAT

    Methods
    -------
        get_abs_vertices():
            returns absolute vertices coordinates (in game screen coordinates system)
        _rotate(angle, position):
            rotates the Wall by Angle and moves its center to Position
    """

    def __init__(self, owner, x, y, size, img_path, direction=0):
        """
        Constructs all neccesary attributes for the Wall Object

        Parameters
        ----------
            owner : None, hidenseek.objects.controllable.Hiding, hidenseek.objects.controllable.Seeker
                Wall owner, None for game environment
            x : float
                center of the rectangle in 'x' axis for absolute coordinate system (game screen)
            y : float
                center of the rectangle in 'y' axis for absolute coordinate system (game screen)
            size : tuple
                Wall size, at least 2x2
        """

        super().__init__()

        self.owner = owner

        self.width = size[0]
        self.height = size[1]

        self.pos = Point((x, y))
        self.pos_init = Point((x, y))

        image = pygame.Surface((self.width, self.height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        self.image = image

        self.filling = [pygame.image.load(os.path.join(img_path, file_))
                        for file_ in os.listdir(img_path)]

        self.rect = self.image.get_rect()
        self.rect.center = (self.pos.x, self.pos.y)

        filling_width = self.filling[0].get_width()
        filling_height = self.filling[0].get_height()

        img_full_size_w = self.width / filling_width
        img_rounded_size_w = math.ceil(img_full_size_w)
        img_full_size_h = self.height / filling_height
        img_rounded_size_h = math.ceil(img_full_size_h)

        blit_list = [(self.filling[0], (filling_width * i, j * filling_height)) for i in range(0, img_rounded_size_w) for j in range(0, img_rounded_size_h)]
        image.blits(blit_list)
        
        self.polygon_points = [Point((self.rect.left, self.rect.top)), Point((self.rect.right, self.rect.top)), Point(
            (self.rect.right, self.rect.bottom)), Point((self.rect.left, self.rect.bottom))]

        self.direction = direction

    def __str__(self):
        return str(self.pos)

    def __repr__(self):
        return self.__str__()

    def get_abs_vertices(self):
        """
        Returns absolute coordinates of Vertices in Polygon

        Parameters
        ----------
            None

        Returns
        -------
            points : list of hidenseek.ext.supportive.Point
                self.pylogon_points mapped to the absolute coordinates system
        """

        return self.polygon_points

    def _rotate(self, angle, position):
        """
        Rotates the sprite by creating new Rectangle and updates its polygon points

        Parameters
        ----------
            angle : float
                player direction in radians
            position : Point
                center of the wall

        Returns
        -------
            None
        """
        self.direction = angle
        # Copy and then rotate the original image.
        copied_image = self.image.copy()
        self.image = pygame.transform.rotozoom(
            copied_image, -angle*180/math.pi, 1)
        self.image.set_colorkey((0, 0, 0))

        # Create a new rect with the center of the sprite.
        self.rect = self.image.get_rect()
        self.rect.center = (position.x, position.y)
        self.width = self.rect.width
        self.height = self.rect.height

        # Update the polygon points for collisions
        self.polygon_points = [Point.triangle_unit_circle_relative(
            angle, self.pos, polygon_point) for polygon_point in self.polygon_points]
