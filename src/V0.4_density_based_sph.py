import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, debug=True)
WIDTH = 1200
HEIGHT = 500

particles = 5000

# BASIC VARIABLES
n = particles
delta_time = 0.0008
particle_radius = 0.05


# VECTORS
velocity = ti.Vector.field(2, dtype=float, shape=n)
position = ti.Vector.field(2, dtype=float, shape=n)
force = ti.Vector.field(2, dtype=float, shape=n)

# SCALARS
densities = ti.field(dtype=float, shape=n)
mass = ti.field(dtype=float, shape=n)
pressures = ti.field(dtype=float, shape=n)

# CONSTANTS
gravity = tm.vec2(0, -9.8)
collision_dampening = 0.5
samplePoint = 2 * particle_radius
smoothing_radius = 0.05
rest_density = 300.0
gas_constant = 5000.0

# SPATIAL HASHING
cell_size = smoothing_radius
grid_resolution = int(1 / cell_size) + 1
max_particles_per_cell = 200
grid_count = ti.field(int, shape=(grid_resolution, grid_resolution))
grid_particles = ti.field(int, shape=(grid_resolution, grid_resolution, max_particles_per_cell))



# FUNCTION RESET FORCES
@ti.func
def reset_forces(i):
    force[i] = mass[i] * gravity

# FUNCTION SMOOTH KERNEL
@ti.func
def smoothing_kernel(distance_between_particles, smoothing_radius):

    kernel_value = 0.0

    if distance_between_particles < smoothing_radius:

        smoothing_radius_squared = smoothing_radius * smoothing_radius
        distance_squared = distance_between_particles * distance_between_particles

        distance_difference = smoothing_radius_squared - distance_squared

        normalization_factor = 4.0 / (tm.pi * smoothing_radius**8)

        kernel_value = normalization_factor * distance_difference**3

    return kernel_value

# FUNCTION CALCULATE DENSITY
@ti.func
def calculate_density(i):
    density = 0.0
    cell = (position[i] / cell_size).cast(int)

    for offset in ti.static(ti.ndrange((-1, 2), (-1, 2))):
        neighbor_cell = cell + ti.Vector(offset)

        if 0 <= neighbor_cell[0] < grid_resolution and 0 <= neighbor_cell[1] < grid_resolution:

            count = grid_count[neighbor_cell[0], neighbor_cell[1]]

            for k in range(count):
                j = grid_particles[neighbor_cell[0], neighbor_cell[1], k]

                r = (position[i] - position[j]).norm()

                if r < smoothing_radius:
                    density += mass[j] * smoothing_kernel(r, smoothing_radius)

    return density

# FUNCTION SPIKY GRADIENT
@ti.func
def spiky_kernel_gradient(r_vec):
    gradient = tm.vec2(0.0)

    r = r_vec.norm()

    if r > 1e-5 and r < smoothing_radius:
        factor = -30.0 / (tm.pi * smoothing_radius**5)
        gradient = factor * (smoothing_radius - r)**2 * (r_vec / r)

    return gradient



# FUNCTION CALCULATE PRESSURE
@ti.func
def calculate_pressure(i):
    pressures[i] = gas_constant * (densities[i] - rest_density)

# FUNCTION CALCULATE PRESSURE FORCE
@ti.func
def calculate_pressure_force(i):
    cell = (position[i] / cell_size).cast(int)

    for offset in ti.static(ti.ndrange((-1, 2), (-1, 2))):
        neighbor_cell = cell + ti.Vector(offset)

        if 0 <= neighbor_cell[0] < grid_resolution and 0 <= neighbor_cell[1] < grid_resolution:

            count = grid_count[neighbor_cell[0], neighbor_cell[1]]

            for k in range(count):
                j = grid_particles[neighbor_cell[0], neighbor_cell[1], k]

                if i != j:
                    r_vec = position[i] - position[j]
                    r = r_vec.norm()

                    if r < smoothing_radius:
                        gradient = spiky_kernel_gradient(r_vec)

                        force[i] += -mass[j] * (
                                            pressures[i] / (densities[i] * densities[i]) +
                                            pressures[j] / (densities[j] * densities[j])
                                        ) * gradient

# CALCULATE VISCOSITY 
@ti.func
def compute_viscosity_force(i):
    cell = (position[i] / cell_size).cast(int)

    for offset in ti.static(ti.ndrange((-1, 2), (-1, 2))):
        neighbor_cell = cell + ti.Vector(offset)

        if 0 <= neighbor_cell[0] < grid_resolution and 0 <= neighbor_cell[1] < grid_resolution:

            count = grid_count[neighbor_cell[0], neighbor_cell[1]]

            for k in range(count):
                j = grid_particles[neighbor_cell[0], neighbor_cell[1], k]

                if i != j:
                    r = (position[i] - position[j]).norm()

                    if r < smoothing_radius:
                        laplacian = 40.0 / (tm.pi * smoothing_radius**5) * (smoothing_radius - r)

                        force[i] += 0.08 * mass[j] * (velocity[j] - velocity[i]) / densities[j] * laplacian

# FUNCTION BOX COLLISION
@ti.func 
def box_collision(i):
    # POSITION X - HORIZONTAL LEFT
    if position[i].x < particle_radius:
        position[i].x = particle_radius
        velocity[i].x *= -collision_dampening
    # POSITION X - HORIZONTAL RIGHT
    if position[i].x > 1 - particle_radius:
        position[i].x = 1- particle_radius
        velocity[i].x *= -collision_dampening

    # POSITION Y - VERTICAL BOTTOM
    if position[i].y < particle_radius:
        position[i].y = particle_radius
        velocity[i].y *= -collision_dampening
    # POSITION X - VERTICAL TOP
    if position[i].y > 1 - particle_radius:
        position[i].y = 1- particle_radius
        velocity[i].y *= -collision_dampening

# INITIALIZE PARTICLES
@ti.kernel
def init():
    for i in range(n):
        mass[i] = 20.0
        position[i] = tm.vec2(ti.random(), ti.random())

# INITIALIZE PARTICLES BUT IN SQUARE FORMATION
@ti.kernel
def init_organized():
    particles_per_row = int(ti.sqrt(n))
    spacing = 0.4 / particles_per_row  

    for i in range(n):
        row = i // particles_per_row
        col = i % particles_per_row

        x = 0.3 + col * spacing
        y = 0.6 + row * spacing 

        position[i] = tm.vec2(x, y)
        velocity[i] = tm.vec2(0.0, 0.0)
        mass[i] = 1.0

#BUILD GRID
@ti.kernel
def build_spatial_grid():
    for i in range(n):
        cell = (position[i] / cell_size).cast(int)

        cell[0] = ti.max(0, ti.min(cell[0], grid_resolution - 1))
        cell[1] = ti.max(0, ti.min(cell[1], grid_resolution - 1))

        index = ti.atomic_add(grid_count[cell[0], cell[1]], 1)

        if index < max_particles_per_cell:
            grid_particles[cell[0], cell[1], index] = i

# CLEAR GRID
@ti.kernel
def clear_grid():
    for I in ti.grouped(grid_count):
        grid_count[I] = 0

@ti.kernel
def compute_densities():
    for i in range(n):
        densities[i] = calculate_density(i)
        pressures[i] = gas_constant * (densities[i] - rest_density)
        if pressures[i] < 0: pressures[i] = 0 

@ti.kernel
def compute_forces():
    for i in range(n):
        force[i] = gravity * densities[i]
        calculate_pressure_force(i)
        compute_viscosity_force(i)

@ti.kernel
def integrate():
    for i in range(n):
        # SEMI IMPLICIT EULER INTEGRATION
        acceleration = force[i] / densities[i]
        velocity[i] += acceleration * delta_time
        position[i] += velocity[i] * delta_time
        
        # BOX COLLISION
        box_collision(i)


# GUI 
gui = ti.GUI("Fluid Sim", res=(WIDTH, HEIGHT), background_color=0x03045e)
reset_btn = gui.button("Reset")

# MAIN LOOP
init_organized()
while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == reset_btn:
            init()
    clear_grid()
    build_spatial_grid()
    compute_densities()
    compute_forces()
    integrate()

    pos = position.to_numpy()
    #gui.circles(pos, radius=7, color=0x0096c7)
    gui.circles(pos, radius=3, color=0x90e0ef)
    #gui.circles(pos, radius=2, color=0x8ecae6)

    gui.show()