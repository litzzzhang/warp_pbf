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
):
    tid = wp.tid()

    x = particle_x[tid]

    # box: [0, width] x [0, height] x [0, length]
    if x[0] < 0.0:
        x = wp.vec3f(0.0, x[1], x[2])
    if x[0] > width:
        x = wp.vec3f(width, x[1], x[2])

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
    smoothing_length: float,
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()

    # [0, width/4] x [0, height] x [0, length/4]
    nr_x = wp.int32(width / (4.0 * smoothing_length))
    nr_y = wp.int32(height / smoothing_length)
    nr_z = wp.int32(length / (4.0 * smoothing_length))

    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)

    pos = smoothing_length * wp.vec3f(x, y, z)

    # jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * smoothing_length * wp.vec3f(
        wp.randn(state), wp.randn(state), wp.randn(state)
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

        self.smoothing_length = 0.10  # kernelRadius
        self.rest_density = 8000.0
        self.damping = 0.98
        self.dt = 0.0083

        # gravity (world units)
        self.gravity = wp.vec3f(0.0, -9.8, 0.0)

        # domain box
        self.width = 5.0
        self.height = 10.0
        self.length = 5.0

        # particle mass (simplified)
        self.particle_mass = 1.0

        nr_x = int(self.width / (2.0 * self.smoothing_length))
        nr_y = int(self.height / self.smoothing_length)
        nr_z = int(self.length / (2.0 * self.smoothing_length))
        self.n = nr_x * nr_y * nr_z

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
                self.smoothing_length,
                self.width,
                self.height,
                self.length,
            ],
            device=self.device,
        )

        # hash grid
        grid_res = int(
            max(self.width, self.height, self.length) / self.smoothing_length
        )
        grid_res = max(grid_res, 8)
        self.grid = wp.HashGrid(grid_res, grid_res, grid_res, device=self.device)

    def step(self):
        with wp.ScopedTimer("step", active=self.verbose):
            for _ in range(self.sim_step_to_frame_ratio):
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
                        inputs=[self.x, self.width, self.height, self.length],
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
        "--num_frames", type=int, default=4800, help="Total number of frames."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out additional status messages during execution.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device or wp.get_preferred_device()):
        example = Example(verbose=args.verbose, device=args.device)

        for _ in range(args.num_frames):
            example.render()
            example.step()
