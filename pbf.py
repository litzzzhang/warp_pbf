import math

import numpy as np

import warp as wp
import warp.render


wp.init()


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x


@wp.func
def sixth(x: float):
    return x * x * x * x * x * x


@wp.func
def ninth(x: float):
    return sixth(x) * cube(x)


@wp.func
def poly6_kernel(r: wp.vec3f, h: float):
    # W(r, h) = 315 / (64*pi*h^9) * (h^2 - |r|^2)^3,  0 <= |r| <= h
    coeff = 315.0 / (64.0 * wp.pi)
    squared_diff = square(h) - wp.length_sq(r)
    if squared_diff <= 0.0:
        return 0.0
    else:
        return (coeff / ninth(h)) * cube(squared_diff)


@wp.func
def grad_poly6_kernel(r: wp.vec3f, h: float):
    # ∇W(r, h) for poly6
    coeff = 945.0 / (32.0 * wp.pi)
    squared_diff = square(h) - wp.length_sq(r)
    if squared_diff <= 0.0:
        return wp.vec3f()
    else:
        return -r * (coeff / ninth(h)) * square(squared_diff)


@wp.func
def spiky_kernel(r: wp.vec3f, h: float):
    # W(r, h) = 15 / (pi*h^6) * (h - |r|)^3
    coeff = 15.0 / wp.pi
    dist = wp.length(r)
    diff = h - dist
    if diff <= 0.0:
        return 0.0
    else:
        return (coeff / sixth(h)) * cube(diff)


@wp.func
def grad_spiky_kernel(r: wp.vec3f, h: float):
    # ∇W(r, h) = -45 / (pi*h^6) * (h - |r|)^2 * r / |r|
    coeff = 45.0 / wp.pi
    dist = wp.length(r)
    diff = h - dist
    if diff <= 0.0:
        return wp.vec3f()
    else:
        return -r * (coeff / (sixth(h) * wp.max(dist, 1.0e-12))) * square(diff)


@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3f),
    particle_rho: wp.array(dtype=float),
    particle_m: wp.array(dtype=float),
    smoothing_length: float,
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)
    x = particle_x[i]

    rho = float(0.0)

    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    for index in neighbors:
        r = x - particle_x[index]
        rho += particle_m[index] * poly6_kernel(r, smoothing_length)

    particle_rho[i] = rho


@wp.func
def compute_constraint(rho: float, rest_density: float):
    return rho / rest_density - 1.0


@wp.kernel
def compute_lambda(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3f),
    particle_rho: wp.array(dtype=float),
    particle_m: wp.array(dtype=float),
    particle_lambda: wp.array(dtype=float),
    rest_density: float,
    smoothing_length: float,
):
    # epsilon (CFM-like regularization)
    epsilon_cfm = 1000.0

    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    x_i = particle_x[i]
    rho_i = particle_rho[i]

    # C_i = rho_i / rho0 - 1
    constraint = compute_constraint(rho_i, rest_density)

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    # gradient at i, and sum of gradient norms squared over all involved particles
    grad_ci_i = wp.vec3f()
    sum_grad_c_sq = float(0.0)

    for j in neighbors:
        if j == i:
            continue

        x_j = particle_x[j]
        m_j = particle_m[j]

        r = x_i - x_j
        gradW = grad_spiky_kernel(r, smoothing_length)

        # ∇_{x_i} C_i
        grad_ci_i += (m_j / rest_density) * gradW

        # ∇_{x_j} C_i
        grad_ci_j = (-m_j / rest_density) * gradW

        sum_grad_c_sq += wp.length_sq(grad_ci_j)

    sum_grad_c_sq += wp.length_sq(grad_ci_i)

    denom = sum_grad_c_sq + epsilon_cfm

    particle_lambda[i] = -constraint / denom


@wp.kernel
def compute_delta_x(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3f),
    particle_rho: wp.array(dtype=float),
    particle_m: wp.array(dtype=float),
    particle_lambda: wp.array(dtype=float),
    particle_dx: wp.array(dtype=wp.vec3f),
    rest_density: float,
    smoothing_length: float,
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    x_i = particle_x[i]
    lambda_i = particle_lambda[i]

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    # Artificial tensile pressure constants (typical PBF values)
    corr_n = 4.0
    corr_h = 0.30  # delta_q = 0.3 * h
    corr_k = 0.0005

    dq = corr_h * smoothing_length * wp.vec3f(1.0, 0.0, 0.0)
    corr_w = poly6_kernel(dq, smoothing_length)

    delta_x = wp.vec3f()

    for j in neighbors:
        if j == i:
            continue

        x_j = particle_x[j]
        m_j = particle_m[j]
        lambda_j = particle_lambda[j]

        r = x_i - x_j

        kernel_val = poly6_kernel(r, smoothing_length)
        if corr_w > 0.0:
            ratio = kernel_val / corr_w
            corr_coeff = -corr_k * wp.pow(ratio, corr_n)
        else:
            corr_coeff = 0.0

        gradW = grad_spiky_kernel(r, smoothing_length)

        coeff = (lambda_i + lambda_j + corr_coeff) * m_j
        delta_x += coeff * gradW

    # Δp_i = (1 / rho0) * Σ_j (λ_i + λ_j + s_corr) m_j ∇W
    delta_x *= 1.0 / rest_density

    particle_dx[i] = delta_x


@wp.kernel
def apply_bounds(
    particle_x: wp.array(dtype=wp.vec3f),
    width: float,
    height: float,
    length: float,
    moving_width: float,
):
    tid = wp.tid()

    x = particle_x[tid]

    x_max = wp.min(width, moving_width)

    # box: [0, width] x [0, height] x [0, length]
    if x[0] < 0.0:
        x = wp.vec3f(0.0, x[1], x[2])
    if x[0] > x_max:
        x = wp.vec3f(x_max, x[1], x[2])

    if x[1] < 0.0:
        x = wp.vec3f(x[0], 0.0, x[2])
    if x[1] > height:
        x = wp.vec3f(x[0], height, x[2])

    if x[2] < 0.0:
        x = wp.vec3f(x[0], x[1], 0.0)
    if x[2] > length:
        x = wp.vec3f(x[0], x[1], length)

    particle_x[tid] = x


@wp.kernel
def predict_positions(
    particle_x: wp.array(dtype=wp.vec3f),
    particle_v: wp.array(dtype=wp.vec3f),
    gravity: wp.vec3f,
    dt: float,
):
    tid = wp.tid()

    v = particle_v[tid]
    v = v + gravity * dt
    x = particle_x[tid] + v * dt

    particle_v[tid] = v
    particle_x[tid] = x


@wp.kernel
def save_positions(
    particle_x: wp.array(dtype=wp.vec3f),
    particle_x_prev: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    particle_x_prev[tid] = particle_x[tid]


@wp.kernel
def update(
    particle_x: wp.array(dtype=wp.vec3f),
    particle_x_prev: wp.array(dtype=wp.vec3f),
    particle_v: wp.array(dtype=wp.vec3f),
    particle_dx: wp.array(dtype=wp.vec3f),
    damping: float,
    dt: float,
):
    tid = wp.tid()

    x_prev = particle_x_prev[tid]

    # apply constraint correction
    x_new = particle_x[tid] + particle_dx[tid]

    # update velocity from position change
    v_new = (x_new - x_prev) / dt
    v_new = damping * v_new

    particle_x[tid] = x_new
    particle_v[tid] = v_new


@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3f),
    spacing: wp.vec3f,
    offset: wp.vec3f,
    count_x: int,
    count_y: int,
    count_z: int,
):
    tid = wp.tid()

    total_particles = count_x * count_y * count_z
    if tid >= total_particles:
        return

    # x fastest, then z, then y (matches Taichi layout)
    x_index = wp.float(tid % count_x)
    yz = tid // count_x
    z_index = wp.float(yz % count_z)
    y_index = wp.float(yz // count_z)

    pos = wp.vec3f(
        offset[0] + spacing[0] * x_index,
        offset[1] + spacing[1] * y_index,
        offset[2] + spacing[2] * z_index,
    )

    particle_x[tid] = pos


class Example:
    def __init__(self, verbose: bool = False, device: str | None = None):
        self.verbose = verbose

        # render config
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0

        self.device = device or wp.get_preferred_device()
        self.renderer = wp.render.OpenGLRenderer("PBF")

        self.smoothing_length = 1.10  # match Taichi kernel radius
        self.rest_density = 1.0
        self.damping = 0.98
        self.dt = 0.0083

        # gravity (world units)
        self.gravity = wp.vec3f(0.0, -9.8, 0.0)

        # domain box
        self.width = 30.0
        self.height = 15.0
        self.length = 15.0

        # Aim the Warp OpenGLRenderer camera toward the fluid block so the
        # full domain is comfortably in frame (per renderer docs).
        scene_center = np.array(
            [self.width * 0.5, self.height * 0.5, self.length * 0.5],
            dtype=np.float32,
        )
        camera_offset = np.array(
            [0.0, self.height * 0.25, self.length * 1.75],
            dtype=np.float32,
        )
        camera_pos = scene_center + camera_offset
        camera_front = scene_center - camera_pos
        norm = np.linalg.norm(camera_front)
        if norm > 0.0:
            camera_front /= norm

        self.renderer.camera_pos = tuple(camera_pos.tolist())
        self.renderer.camera_front = tuple(camera_front.tolist())
        self.renderer.camera_up = (0.0, 1.0, 0.0)
        self.renderer.update_view_matrix()

        self.boundary_extent = self.width - 1.0e-3
        self.boundary_phase = 0.0
        self.boundary_period_steps = 90.0
        self.boundary_reference_dt = 1.0 / 20.0
        self.boundary_velocity_strength = 4.5

        # particle mass (simplified)
        self.particle_mass = 1.0

        # Taichi scene block dimensions (30 x 10 x 15)
        self.num_particles_x = 30
        self.num_particles_y = 10
        self.num_particles_z = 15
        self.n = self.num_particles_x * self.num_particles_y * self.num_particles_z
        offset_x = 0.0
        offset_y = self.height * 0.1
        offset_z = 0.0
        self.particle_offset = wp.vec3f(offset_x, offset_y, offset_z)

        desired_spacing = self.smoothing_length * 1.05

        def compute_spacing(available: float, count: int):
            if count <= 1:
                return 0.0
            effective_available = max(available, self.smoothing_length)
            return min(desired_spacing, effective_available / (count - 1))

        available_x = self.boundary_extent - offset_x - self.smoothing_length
        available_y = self.height - offset_y - self.smoothing_length
        available_z = self.length - offset_z - self.smoothing_length

        spacing_x = compute_spacing(available_x, self.num_particles_x)
        spacing_y = compute_spacing(available_y, self.num_particles_y)
        spacing_z = compute_spacing(available_z, self.num_particles_z)

        self.particle_spacing = wp.vec3f(spacing_x, spacing_y, spacing_z)

        # solver substeps per frame
        self.sim_step_to_frame_ratio = 4

        # allocate arrays
        self.m = wp.full(
            shape=self.n, value=self.particle_mass, dtype=float, device=self.device
        )
        self.x = wp.empty(self.n, dtype=wp.vec3f, device=self.device)
        self.x_prev = wp.empty(self.n, dtype=wp.vec3f, device=self.device)
        self.v = wp.zeros(self.n, dtype=wp.vec3f, device=self.device)
        self.rho = wp.zeros(self.n, dtype=float, device=self.device)
        self.lambda_ = wp.zeros(self.n, dtype=float, device=self.device)
        self.dx = wp.zeros(self.n, dtype=wp.vec3f, device=self.device)

        # init positions
        wp.launch(
            kernel=initialize_particles,
            dim=self.n,
            inputs=[
                self.x,
                self.particle_spacing,
                self.particle_offset,
                self.num_particles_x,
                self.num_particles_y,
                self.num_particles_z,
            ],
            device=self.device,
        )

        # hash grid
        grid_res = int(
            max(self.width, self.height, self.length) / self.smoothing_length
        )
        grid_res = max(grid_res, 8)
        self.grid = wp.HashGrid(grid_res, grid_res, grid_res, device=self.device)

    def move_boundary(self):
        step_increments = self.dt / self.boundary_reference_dt
        self.boundary_phase += step_increments
        if self.boundary_phase >= 2.0 * self.boundary_period_steps:
            self.boundary_phase -= 2.0 * self.boundary_period_steps

        sin_arg = (self.boundary_phase * math.pi) / self.boundary_period_steps
        velocity = -math.sin(sin_arg) * self.boundary_velocity_strength
        new_extent = self.boundary_extent + velocity * self.dt
        min_extent = 2.0 * self.smoothing_length
        max_extent = self.width - 1.0e-3
        self.boundary_extent = float(np.clip(new_extent, min_extent, max_extent))

    def step(self):
        with wp.ScopedTimer("step", active=self.verbose):
            for _ in range(self.sim_step_to_frame_ratio):
                self.move_boundary()
                # save old positions for velocity update
                wp.launch(
                    kernel=save_positions,
                    dim=self.n,
                    inputs=[self.x, self.x_prev],
                    device=self.device,
                )

                # predict positions with gravity
                wp.launch(
                    kernel=predict_positions,
                    dim=self.n,
                    inputs=[self.x, self.v, self.gravity, self.dt],
                    device=self.device,
                )

                # build hash grid
                with wp.ScopedTimer("grid build", active=self.verbose):
                    self.grid.build(self.x, self.smoothing_length)

                with wp.ScopedTimer("solver", active=self.verbose):
                    # density
                    wp.launch(
                        kernel=compute_density,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.rho,
                            self.m,
                            self.smoothing_length,
                        ],
                        device=self.device,
                    )

                    # lambda
                    wp.launch(
                        kernel=compute_lambda,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.rho,
                            self.m,
                            self.lambda_,
                            self.rest_density,
                            self.smoothing_length,
                        ],
                        device=self.device,
                    )

                    # Δx
                    wp.launch(
                        kernel=compute_delta_x,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.rho,
                            self.m,
                            self.lambda_,
                            self.dx,
                            self.rest_density,
                            self.smoothing_length,
                        ],
                        device=self.device,
                    )

                    # apply corrections & update velocities
                    wp.launch(
                        kernel=update,
                        dim=self.n,
                        inputs=[
                            self.x,
                            self.x_prev,
                            self.v,
                            self.dx,
                            self.damping,
                            self.dt,
                        ],
                        device=self.device,
                    )

                    # boundaries
                    wp.launch(
                        kernel=apply_bounds,
                        dim=self.n,
                        inputs=[
                            self.x,
                            self.width,
                            self.height,
                            self.length,
                            self.boundary_extent,
                        ],
                        device=self.device,
                    )

                self.sim_time += self.dt

    def render(self):
        with wp.ScopedTimer("render", active=self.verbose):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.x.numpy(),
                radius=self.smoothing_length * 0.5,
                name="points",
                colors=(0.2, 0.5, 0.9),
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override the default Warp device."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out additional status messages during execution.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device or wp.get_preferred_device()):
        example = Example(verbose=args.verbose, device=args.device)

        while example.renderer.is_running():
            example.render()
            if not example.renderer.is_running():
                break
            example.step()
