import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, debug=True)

particles = 5000

# basic forces 
n = particles
delta_time = 0.002
gravity = tm.vec2(0, -9.8)
energy_loss = -0.8
radius = 0.02
stiffness = 100
cohesion_strength = 1.2
viscosity = 1.0

# basic vectors
position = ti.Vector.field(2, dtype=float, shape=n)
velocity = ti.Vector.field(2, dtype=float, shape=n)
force = ti.Vector.field(2, dtype=float, shape=n)
mass = ti.field(dtype=float, shape=n)

interaction_radius = 2 * radius

# grid and cells
cell_size = interaction_radius
grid_res = int(1 / cell_size) + 1
grid_count = ti.field(int, shape=(grid_res, grid_res))
max_particles_per_cell = 50
grid_particles = ti.field(int, shape=(grid_res, grid_res, max_particles_per_cell))

@ti.func
def add_particle_to_grid(i):
    cell = (position[i] / cell_size).cast(int)
    cell = ti.max(0, ti.min(cell, grid_res - 1))
    idx = ti.atomic_add(grid_count[cell[0], cell[1]], 1)

    if idx < max_particles_per_cell:
        grid_particles[cell[0], cell[1], idx] = i

@ti.func
def look_neighbors(i):
    # descobrir célula da partícula i
    cell = (position[i] / interaction_radius).cast(int)
    cell = ti.max(0, ti.min(cell, grid_res - 1))

    # olhar 3x3 células ao redor
    for offset in ti.static(ti.ndrange((-1, 2), (-1, 2))):
        neighbor_cell = cell + ti.Vector(offset).cast(int)

        # verificar se célula é válida
        if 0 <= neighbor_cell[0] < grid_res and 0 <= neighbor_cell[1] < grid_res:

            count = ti.min(grid_count[neighbor_cell[0], neighbor_cell[1]], max_particles_per_cell)

            for k in range(count):

                j = grid_particles[neighbor_cell[0], neighbor_cell[1], k]

                if i < j:  # evitar duplicar interação
                    particle_collision(i, j)

@ti.func
def reset_forces(i):
    force[i] = mass[i] * gravity

@ti.func
def particle_collision(i, j):
    vdiff = position[i] - position[j]
    distance = vdiff.norm()
    
    if distance > 1e-5:
        
        normal = vdiff / distance
        # repulsion
        collision_radius = 2 * radius
        if distance < collision_radius:
            penetration = collision_radius - distance
            F_rep = stiffness * penetration * normal
            force[i] += F_rep
            force[j] -= F_rep

        #cohesion
        cohesion_radius = 3 * radius
        if distance < cohesion_radius:
            q = distance / cohesion_radius
            F_coh = -cohesion_strength * (1 - q) * normal
            force[i] += F_coh
            force[j] -= F_coh
            
        # viscosity
        if distance < cohesion_radius:
            vel_diff = velocity[i] - velocity[j]
            F_visc = -viscosity * vel_diff
            force[i] += F_visc
            force[j] -= F_visc

@ti.func
def border_collision(i):
    if position[i].x < radius:
        position[i].x = radius
        velocity[i].x *= energy_loss
    if position[i].x > 1 - radius:
        position[i].x = 1 - radius
        velocity[i].x *= energy_loss

    if position[i].y < radius:
        position[i].y = radius
        velocity[i].y *= energy_loss
    if position[i].y > 1 - radius:
        position[i].y = 1 - radius
        velocity[i].y *= energy_loss

# initialize particles
@ti.kernel
def init():
    for i in range(n):
        position[i] = tm.vec2(ti.random(), ti.random())
        velocity[i] = tm.vec2(0, 0)
        mass[i] = 0.1

# particle behavior
@ti.kernel
def update():
    #reset grid
    for I in ti.grouped(grid_count):
        grid_count[I] = 0

    # reset forces and apply gravity
    for i in range(n):
        reset_forces(i)
    
     # insert into grid
    for i in range(n):
        add_particle_to_grid(i)

    # particle collision
    for i in range(n):
        look_neighbors(i)

    #integration
    for i in range(n):
        velocity[i] += delta_time * force[i] / mass[i]
        position[i] += delta_time * velocity[i]
        velocity[i] *= 0.99
        
        # colision with border 
        border_collision(i)


gui = ti.GUI("Fluid Sim", res=(600, 300), background_color=0x999999)
reset_btn = gui.button("Reset")



init()
while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == reset_btn:
            init()
    update()
    gui.circles(position.to_numpy(), radius=2, color=4000)
    gui.show()