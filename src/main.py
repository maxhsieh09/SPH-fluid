import numpy as np
import cv2
import pygame
from pygame.locals import *

from sim_optimized import SPH2D

pygame.init()

outline_color = (10, 100, 200)
RED = np.array((250, 10, 0))
BLUE = np.array((10, 100, 250))
NEUTRAL = np.array((50, 50, 50))

def pressure_color(pressure, max_pressure):
    if pressure < 0:
        factor = -pressure / max_pressure
        factor = np.clip(factor, 0, 1)
        return BLUE * factor + (1 - factor) * NEUTRAL
    else:
        factor = pressure / max_pressure
        factor = np.clip(factor, 0, 1)
        return RED * factor + (1 - factor) * NEUTRAL


def visualize(sim: SPH2D):
    max_size = 800
    scale = max_size / max(sim.bounds)
    screen_size = (int(sim.bounds[0] * scale), int(sim.bounds[1] * scale))
    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()
    running = True

    sim.step()
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        screen.fill((255, 255, 255))

        for i in range(sim.positions.shape[0]):
            pos = sim.positions[i] * scale
            color = pressure_color(sim.pressures[i], 10000)

            pygame.draw.circle(screen, color, pos, 3)
            #pygame.draw.circle(screen, outline_color, pos, int(sim.kernel_radius * scale), 1)

        sim.step()
        pygame.display.flip()
        #clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    sim = SPH2D(particle_spacing=0.008, smooth_factor=2, dt=0.002, stiffness=20, viscosity=0.2)
    for x in np.linspace(0.1, 0.4, 37):
        for y in np.linspace(0.4, 0.7, 37):
            sim.add_particle(np.array([x, y]) + np.random.uniform(-0.001, 0.001, 2))

    visualize(sim)
