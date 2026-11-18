import numpy as np
import cv2
import pygame
from pygame.locals import *

from sim_optimized import SPH2D

pygame.init()

outline_color = (10, 100, 200)
RED = np.array((250, 10, 0))
BLUE = np.array((82, 165, 255))
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
    font = pygame.font.Font(None, 30)
    running = True

    sim.step()
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        screen.fill((255, 255, 255))

        for i in range(sim.positions.shape[0]):
            pos = np.astype(sim.positions[i] * scale, int)
            #color = pressure_color(sim.pressures[i], 80000)
            color = BLUE

            pygame.draw.circle(screen, color, pos, 3)
            #pygame.draw.circle(screen, outline_color, pos, int(sim.kernel_radius * scale), 1)

        text = font.render(f"FPS: {int(clock.get_fps())}", True, (0, 0, 0))
        screen.blit(text, (10, 10))

        for i in range(8):
            sim.step()
        
        pygame.display.flip()
        clock.tick()

    pygame.quit()


if __name__ == "__main__":
    sim = SPH2D(
        particle_spacing=0.005,
        smooth_factor=2,
        dt=0.001,
        stiffness=40,
        viscosity=0.1,
        wall_viscosity=0
    )

    #for x in np.linspace(0.1, 0.4, 36):
    #    for y in np.linspace(0.4, 0.7, 36):
    #        sim.add_particle(np.array([x, y]) + np.random.uniform(-0.001, 0.001, 2))

    dx = sim.particle_spacing

    xs = np.arange(0.05, 0.3, dx)
    ys = np.arange(0.2, 0.65, dx)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    positions = np.column_stack((X.ravel(), Y.ravel()))
    positions += np.random.uniform(-0.001, 0.001, (positions.shape[0], 2))
    sim.add_particles(positions)

    visualize(sim)
