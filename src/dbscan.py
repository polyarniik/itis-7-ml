import sys
from typing import TypeVar
import numpy as np
import pygame

MAX_NEIGHBORS_DISTANCE = 50
MIN_NEIGHBORS = 3
CIRCLE_RADIUS = 5
BLACK = 'black'
GREEN = 'green'
YELLOW = 'yellow'
RED = 'red'

TPoint= TypeVar("TPoint", bound="Point")

class Point:
    radius: int = CIRCLE_RADIUS
    color: str = BLACK
    neighbors: list[TPoint]
    
    def __init__(self, x, y, color=None, radius=None):
        self.x = x
        self.y = y
        self.color = color or self.color
        self.radius = radius or self.radius
        self.neighbors = []

    def add_neighbor(self, point: TPoint):
        self.neighbors.append(point)

    def distance(self, point):
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)
        


class DBSCAN:
    EPS = MAX_NEIGHBORS_DISTANCE
    MIN_SAMPLES = MIN_NEIGHBORS
    points: list[Point]

    def __init__(self, points: list[Point]) -> None:
        self.points = points

    def get_points_cluster(self):
        for point in self.points:
            point.color = RED   

        for i, point_i in enumerate(self.points):
            point_i.neighbors.clear()
            for j, point_j in enumerate(self.points):
                if i != j and point_i.distance(point_j) < MAX_NEIGHBORS_DISTANCE:
                    point_i.add_neighbor(point_j)
            if len(point_i.neighbors) >= MIN_NEIGHBORS:
                point_i.color = GREEN

        for point in self.points:
            if point.color != GREEN:
                for neighbor in point.neighbors:
                    if neighbor.color == GREEN:
                        point.color = YELLOW
                        break
        
        return self.points


class Game:
    points: list

    def __init__(self) -> None:
        self.points = []

    def draw_point(self, point: Point):
        pygame.draw.circle(self.screen, point.color, (point.x, point.y), point.radius)


    def start(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 800))
        self.screen.fill(color='white')
        pygame.display.update()

        while True: 
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    point = Point(x=event.pos[0], y=event.pos[1])
                    self.draw_point(point)
                    self.points.append(point)
                    pygame.display.update()
                elif pygame.key.get_pressed()[pygame.K_RETURN]:
                    self.points = DBSCAN(self.points).get_points_cluster()
                    for point in self.points:
                        self.draw_point(point)
                    pygame.display.update()
                elif event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:
                    pygame.quit()
                    sys.exit()


Game().start()