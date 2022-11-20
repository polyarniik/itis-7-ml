import sys
import pygame
import numpy as np
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt

CIRCLE_RADIUS = 5
BLACK = 'black'
GREEN = 'green'
YELLOW = 'yellow'
RED = 'red'


class Point:
    radius: int = CIRCLE_RADIUS
    color: str = RED

    def __init__(self, x, y, color=None, radius=None):
        self.x = x
        self.y = y
        self.color = color or self.color
        self.radius = radius or self.radius
        self.neighbors = []

    def distance(self, point):
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)


class Game:
    def __init__(self) -> None:
        self.points: list[Point] = []
        self.is_svm_start = False
        self.svm = svm.SVC(kernel='linear')

    def draw_point(self, point: Point):
        pygame.draw.circle(self.screen, point.color, (point.x, point.y), point.radius)

    def show_svm(self):
        data_frame = pd.DataFrame(
            data={
                'x': [x.x for x in self.points],
                'y': [x.y for x in self.points],
                'target': [1 if x.color == GREEN else 0 for x in self.points],
            }
        )
        self.svm.fit(data_frame[['x', 'y']].values, data_frame['target'].values)
        fig, ax = plt.subplots(figsize=(12, 7))
        xx = np.linspace(-1, max(data_frame['x']) + 1, len(self.points))
        yy = np.linspace(0, max(data_frame['y']) + 1, len(self.points))
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        colors = np.where(data_frame['target'].values == 1, GREEN, RED)
        ax.scatter(data_frame['x'], data_frame['y'], c=colors)
        Z = self.svm.decision_function(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, colors=BLACK, levels=[0], alpha=0.5, linestyles=['-'])
        plt.show()

    def start(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 800))
        self.screen.fill(color='white')
        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    point = Point(x=event.pos[0], y=event.pos[1])
                    self.points.append(point)
                    if event.button == 1:
                        point.color = GREEN
                        self.draw_point(point)
                    elif event.button == 3:
                        point.color = RED
                        self.draw_point(point)
                    pygame.display.update()
                elif pygame.key.get_pressed()[pygame.K_RETURN]:
                    self.show_svm()
                elif event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:
                    pygame.quit()
                    sys.exit()


Game().start()
