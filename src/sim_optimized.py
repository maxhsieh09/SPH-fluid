import numpy as np
from scipy.spatial import KDTree
from numba import njit

SMALL_NUMBER = 1e-6


@njit(cache=True)
def cubic_kernel(r, h):
    q = 2.0 * r / h
    factor = 40.0 / (7.0 * np.pi * h**2)

    if q >= 2.0:
        return 0.0
    elif q > 1.0:
        return factor * 0.25 * (2.0 - q)**3
    else:
        return factor * (1.0 - 1.5*q*q + 0.75*q*q*q)


@njit(cache=True)
def compute_density_numba(
    positions, pairs, n_fluid, h
):
    # Fluid densities, including the self contribution
    rho = np.full(n_fluid, cubic_kernel(0.0, h))

    for k in range(len(pairs)):
        i, j = pairs[k]

        dx = positions[i, 0] - positions[j, 0]
        dy = positions[i, 1] - positions[j, 1]

        r = np.sqrt(dx*dx + dy*dy)
        w = cubic_kernel(r, h)

        # Fluid-fluid pair: both receive a contribution
        if i < n_fluid:
            rho[i] += w

        if j < n_fluid:
            rho[j] += w

    return rho


@njit(cache=True)
def compute_forces_numba(
    positions, velocities, pairs,
    rho, pressure, n_fluid,
    h, rho0, stiffness, alpha=0.1
):
    acc = np.zeros((n_fluid, 2))

    # Spiky kernel derivative factor (2D)
    factor = 10.0 / (np.pi * h**5)

    # Numerical speed of sound
    c = np.sqrt(stiffness)

    for k in range(len(pairs)):
        i, j = pairs[k]

        # Skip boundary-boundary interactions
        if i >= n_fluid and j >= n_fluid:
            continue

        # Relative position: x_j - x_i
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]

        r2 = dx*dx + dy*dy

        if r2 < 1e-12 or r2 >= h*h:
            continue

        r = np.sqrt(r2)

        slope = -3.0 * factor * (h-r)**2

        i_fluid = i < n_fluid
        j_fluid = j < n_fluid

        # Pressure interaction
        if i_fluid and j_fluid:
            term = pressure[i] / rho[i]**2 + pressure[j] / rho[j]**2

            # Relative velocity: v_j - v_i
            dvx = velocities[j, 0] - velocities[i, 0]
            dvy = velocities[j, 1] - velocities[i, 1]

            dot = dvx*dx + dvy*dy

            if dot < 0.0:
                rho_avg = 0.5 * (rho[i] + rho[j])
                mu = h * dot / (r2 + 0.01*h*h)

                pi_ij = -alpha * c * mu / max(rho_avg, 1e-6)

                term += pi_ij

        elif i_fluid:
            # Simplified ghost boundary pressure
            term = pressure[i] / rho[i]**2 + pressure[i] / rho0**2

        else:
            term = pressure[j] / rho[j]**2 + pressure[j] / rho0**2

        scale = term * slope / r

        ax = scale * dx
        ay = scale * dy

        if i_fluid:
            acc[i, 0] += ax
            acc[i, 1] += ay

        if j_fluid:
            acc[j, 0] -= ax
            acc[j, 1] -= ay

    return acc


class SPH2D:
    def __init__(self, particle_spacing=0.008, smooth_factor=2, dt=0.002, stiffness=20, viscosity=0.2):
        self.particle_spacing = particle_spacing
        self.kernel_radius = particle_spacing * smooth_factor
        self.target_density = (1 / self.particle_spacing) ** 2
        self.dt = dt
        self.stiffness = stiffness
        self.viscosity = viscosity
        self.bounds = np.array([1., 0.7])
        self.reflectivity = 0
        self.gravity = 9.8

        self.positions = np.zeros((0, 2))
        self.velocities = np.zeros((0, 2))
        self.densities = np.zeros(0)
        self.pressures = np.zeros(0)

        self.grid_size = self.kernel_radius
        self.cell_to_indices = {}

        self.create_boundary_particles()

    def add_particle(self, x, v=None):
        if v is None:
            v = np.zeros(2)
        self.positions = np.vstack((self.positions, x))
        self.velocities = np.vstack((self.velocities, v))
        self.densities = np.append(self.densities, self.target_density)
        self.pressures = np.append(self.pressures, 0)

        self.update_grid()

    def create_boundary_particles(self):
        dx = self.particle_spacing
        width, height = self.bounds

        wall = []

        # Two layers outside each wall
        offsets = [-0.5 * dx, -1.5 * dx]

        xs = np.arange(-1.5 * dx, width + 2 * dx, dx)
        ys = np.arange(-1.5 * dx, height + 2 * dx, dx)

        for d in offsets:
            # Left and right
            for y in ys:
                wall.append([d, y])
                wall.append([width - d, y])

            # Bottom and top
            for x in xs:
                wall.append([x, d])
                wall.append([x, height - d])

        # Remove duplicate corner samples
        self.boundary_positions = np.unique(np.round(np.array(wall), 10), axis=0)

    def pos_to_cell(self, pos):
        return tuple(np.astype(pos // self.grid_size, int).tolist())

    def update_grid(self):
        self.all_positions = np.vstack((
            self.positions,
            self.boundary_positions
        ))

        self.cell_to_indices.clear()

        for i, pos in enumerate(self.all_positions):
            cell = self.pos_to_cell(pos)
            self.cell_to_indices.setdefault(cell, []).append(i)

    def neighbor_indices(self, idx, include_self=False):
        cell = self.pos_to_cell(self.all_positions[idx])
        result = []
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if (i, j) in self.cell_to_indices:
                    for k in self.cell_to_indices[(i, j)]:
                        if k != idx or include_self:
                            result.append(k)
        return result

    def build_pairs(self):
        all_positions = np.vstack((self.positions, self.boundary_positions))

        tree = KDTree(all_positions)
        pairs = tree.query_pairs(self.kernel_radius, output_type="ndarray")

        return np.ascontiguousarray(pairs, dtype=np.int64)

    def calculate_density(self):
        for i in range(self.positions.shape[0]):
            neighbors = self.neighbor_indices(i, include_self=True)
            dist = np.linalg.norm(self.all_positions[neighbors] - self.positions[i], axis=1)
            self.densities[i] = np.sum(self.kernel.get(dist, self.kernel_radius))

    def apply_xsph(self, epsilon=0.05):
        n = len(self.positions)

        # Boundary particles have zero velocity
        all_velocities = np.vstack((
            self.velocities,
            np.zeros_like(self.boundary_positions)
        ))

        new_velocities = self.velocities.copy()

        for i in range(n):
            neighbors = np.asarray(self.neighbor_indices(i), dtype=int)

            if len(neighbors) == 0:
                continue

            delta = (self.all_positions[neighbors] - self.positions[i])

            dist = np.linalg.norm(delta, axis=1)
            weights = self.kernel.get(dist, self.kernel_radius)

            rho_j = np.full(len(neighbors), self.target_density)

            fluid_mask = neighbors < n
            rho_j[fluid_mask] = self.densities[neighbors[fluid_mask]]

            velocity_diff = (all_velocities[neighbors] - self.velocities[i])

            correction = np.sum(
                (weights / np.maximum(rho_j, SMALL_NUMBER))[:, None] * velocity_diff,
                axis=0
            )

            new_velocities[i] += epsilon * correction

        self.velocities[:] = new_velocities

    def artificial_viscosity(self, idx, alpha=0.1):
        neighbors = np.asarray(self.neighbor_indices(idx), dtype=int)

        # Only include moving fluid particles initially
        neighbors = neighbors[neighbors < len(self.positions)]

        if len(neighbors) == 0:
            return np.zeros(2)

        # r_ij = x_i - x_j
        rij = self.positions[idx] - self.positions[neighbors]

        # v_ij = v_i - v_j
        vij = self.velocities[idx] - self.velocities[neighbors]

        r2 = np.sum(rij**2, axis=1)
        r = np.sqrt(r2)

        dot = np.sum(vij * rij, axis=1)

        h = self.kernel_radius
        c = np.sqrt(self.stiffness)

        rho_avg = 0.5 * (self.densities[idx] + self.densities[neighbors])

        mu = h * dot / (r2 + 0.01 * h**2)

        # Only damp approaching particles
        pi_ij = np.where(
            dot < 0,
            -alpha * c * mu / np.maximum(rho_avg, SMALL_NUMBER),
            0.0
        )

        # Kernel gradient with respect to x_i
        slope = self.spiky_kernel.get_derivative(r, h)

        grad = slope[:, None] * rij / np.maximum(r[:, None], SMALL_NUMBER)

        # Your original solver implicitly uses particle mass = 1
        return -np.sum(pi_ij[:, None] * grad, axis=0)

    def resolve_collisions(self):
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.bounds[0])
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.bounds[1])

        self.velocities[self.positions[:, 0] <= 0, 0] *= -self.reflectivity
        self.velocities[self.positions[:, 0] >= self.bounds[0], 0] *= -self.reflectivity
        self.velocities[self.positions[:, 1] <= 0, 1] *= -self.reflectivity
        self.velocities[self.positions[:, 1] >= self.bounds[1], 1] *= -self.reflectivity

    def apply_wall_friction(self):
        margin = self.kernel_radius
        dt = self.dt
        gamma = 100.0

        x = self.positions[:, 0]
        y = self.positions[:, 1]

        near_vertical = (
            (x < margin) |
            (x > self.bounds[0] - margin)
        )

        near_horizontal = (
            (y < margin) |
            (y > self.bounds[1] - margin)
        )

        # Dampen tangential velocities
        factor = np.exp(-gamma * dt)

        self.velocities[near_vertical, 1] *= factor
        self.velocities[near_horizontal, 0] *= factor

    def compute_accelerations(self):
        n = len(self.positions)

        all_positions = np.ascontiguousarray(
            np.vstack((self.positions, self.boundary_positions))
        )
        all_velocities = np.ascontiguousarray(
            np.vstack((self.velocities, np.zeros_like(self.boundary_positions)))
        )

        tree = KDTree(all_positions)

        pairs = np.ascontiguousarray(
            tree.query_pairs(self.kernel_radius, output_type="ndarray"),
            dtype=np.int64
        )

        self.densities = compute_density_numba(
            all_positions,
            pairs,
            n,
            self.kernel_radius
        )

        self.pressures = self.stiffness * np.maximum(self.densities - self.target_density, 0.0)

        acc = compute_forces_numba(
            all_positions, all_velocities, pairs,
            self.densities, self.pressures,
            n, self.kernel_radius,
            self.target_density,
            self.stiffness,
            alpha=self.viscosity
        )

        acc[:, 1] += self.gravity

        return acc


    def step(self):
        dt = self.dt

        x0 = self.positions.copy()
        v0 = self.velocities.copy()

        # Beginning-of-step acceleration
        a0 = self.compute_accelerations()

        # Predict next positions and velocities
        self.positions = x0 + dt*v0 + 0.5*dt*dt*a0
        self.velocities = v0 + dt*a0

        self.resolve_collisions()

        # Evaluate forces using predicted end-of-step state
        a1 = self.compute_accelerations()

        # Correct velocity
        self.velocities = v0 + 0.5*dt*(a0 + a1)

        #self.apply_wall_friction()
        self.resolve_collisions()
