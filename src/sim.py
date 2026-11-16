import numpy as np

SMALL_NUMBER = 1e-6


class CubicKernel:
    def __init__(self, dim):
        self.dim = dim
        if dim == 3:
            self.factor = 1 / np.pi
        elif dim == 2:
            self.factor = 10 / 7 / np.pi
        else:
            self.factor = 2 / 3

    def get(self, r, h):
        s = h / 2
        q = r / s
        factor = self.factor / s ** self.dim
        return np.where(
            q > 2, 0,
            np.where(
                q > 1,
                (2 - q) ** 3 * 0.25,
                (1 - 1.5 * q ** 2 * (1 - q * 0.5))
            )
        ) * factor

    def get_derivative(self, r, h):
        q = r * 2 / h
        factor = self.factor / h ** self.dim
        return np.where(
            q > 2, 0,
            np.where(
                q > 1,
                -0.75 * (2 - q) ** 2,
                -3 * q * (1 - q * 0.75)
            )
        ) * factor


class SpikyKernel:
    def __init__(self, dim):
        self.dim = dim
        if dim == 3:
            self.factor = 15 / np.pi
        elif dim == 2:
            self.factor = 10 / np.pi
        else:
            raise NotImplementedError

    def get(self, r, h):
        factor = self.factor / h ** (self.dim + 3)
        return np.where(r > h, 0, factor * (h - r) ** 3)

    def get_derivative(self, r, h):
        factor = self.factor / h ** (self.dim + 3)
        return np.where(r > h, 0, -3 * factor * (h - r) ** 2)


class SPH2D:
    def __init__(self, particle_spacing=0.02, smooth_factor=2, dt=0.003, stiffness=40):
        self.particle_spacing = particle_spacing
        self.kernel_radius = particle_spacing * smooth_factor
        self.target_density = (1 / self.particle_spacing) ** 2
        self.dt = dt
        self.stiffness = stiffness
        self.bounds = np.array([1., 0.7])
        self.reflectivity = 0
        self.gravity = 9.8

        self.kernel = CubicKernel(2)
        self.spiky_kernel = SpikyKernel(2)

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

    def calculate_density(self):
        for i in range(self.positions.shape[0]):
            neighbors = self.neighbor_indices(i, include_self=True)
            dist = np.linalg.norm(self.all_positions[neighbors] - self.positions[i], axis=1)
            self.densities[i] = np.sum(self.kernel.get(dist, self.kernel_radius))

    def pressure_acceleration(self, idx):
        neighbors = np.asarray(self.neighbor_indices(idx), dtype=int)

        if len(neighbors) == 0:
            return np.zeros(2)

        n_fluid = len(self.positions)
        
        dist = np.linalg.norm(self.all_positions[neighbors] - self.positions[idx], axis=1)
        direction = (self.all_positions[neighbors] - self.positions[idx]) / (dist[:, np.newaxis] + SMALL_NUMBER)
        slope = self.spiky_kernel.get_derivative(dist, self.kernel_radius)

        rho_i = max(self.densities[idx], SMALL_NUMBER)
        #rho_i = self.target_density
        p_i = self.pressures[idx]

        # Initialize ghost boundary values
        rho_j = np.full(len(neighbors), self.target_density)
        #rho_j = self.target_density
        p_j = np.full(len(neighbors), p_i)

        # Overwrite values for fluid neighbors
        fluid_mask = neighbors < n_fluid
        fluid_indices = neighbors[fluid_mask]

        rho_j[fluid_mask] = self.densities[fluid_indices]
        p_j[fluid_mask] = self.pressures[fluid_indices]

        rho_j = np.maximum(rho_j, SMALL_NUMBER)

        pressure_term = p_i / rho_i**2 + p_j / rho_j**2

        return np.sum((pressure_term * slope)[:, None] * direction, axis=0)

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

    def compute_accelerations(self):
        self.update_grid()
        self.calculate_density()

        self.pressures = self.stiffness * np.maximum(self.densities - self.target_density, 0)

        accelerations = np.zeros_like(self.positions)

        for i in range(len(self.positions)):
            accelerations[i] = self.pressure_acceleration(i) + self.artificial_viscosity(i, alpha=0.3)

        accelerations[:, 1] += self.gravity

        return accelerations

    def step(self):
        dt = self.dt

        a0 = self.compute_accelerations()
        self.velocities += 0.5 * dt * a0
        self.positions += dt * self.velocities
        self.resolve_collisions()

        a1 = self.compute_accelerations()
        self.velocities += 0.5 * dt * a1
        #self.apply_xsph(epsilon=0.1)
        self.resolve_collisions()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    kernel = SpikyKernel(2)

    r = np.linspace(0, 1, 200)
    plt.plot(r, kernel.get(r, 0.5))
    plt.plot(r, kernel.get_derivative(r, 0.5) * 0.1)
    plt.show()
