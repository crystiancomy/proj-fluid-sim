import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan, debug=True)

particles = 2000

# basic forces 
n = particles
dt = 0.002
gravity = tm.vec2(0, -9.8)
energy_loss = -0.4
radius = 0.02
stiffness = 80
cohesion_strength = 1.2
viscosity = 1.4

# basic vectors
pos = ti.Vector.field(2, dtype=float, shape=n)
vel = ti.Vector.field(2, dtype=float, shape=n)
force = ti.Vector.field(2, dtype=float, shape=n)
mass = ti.field(dtype=float, shape=n)

@ti.func
def reset_forces(i):
    force[i] = mass[i] * gravity

@ti.func
def particle_collision(i, j):
    vdiff = pos[i] - pos[j]
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
            vel_diff = vel[i] - vel[j]
            F_visc = -viscosity * vel_diff
            force[i] += F_visc
            force[j] -= F_visc

@ti.func
def border_collision(i):
    if pos[i].x < radius:
        pos[i].x = radius
        vel[i].x *= energy_loss
    if pos[i].x > 1 - radius:
        pos[i].x = 1 - radius
        vel[i].x *= energy_loss

    if pos[i].y < radius:
        pos[i].y = radius
        vel[i].y *= energy_loss
    if pos[i].y > 1 - radius:
        pos[i].y = 1 - radius
        vel[i].y *= energy_loss

# initialize particles
@ti.kernel
def init():
    for i in range(n):
        pos[i] = tm.vec2(ti.random(), ti.random())
        vel[i] = tm.vec2(0, 0)
        mass[i] = 0.1

# particle behavior
@ti.kernel
def update():
    # reset forces and apply gravity
    for i in range(n):
        reset_forces(i)

    # particle collision
    for i in range(n):
        for j in range(i + 1, n):
            particle_collision(i,j)

    #integration
    for i in range(n):
        vel[i] += dt * force[i] / mass[i]
        pos[i] += dt * vel[i]
        vel[i] *= 0.99
        
        # colision with border 
        border_collision(i)


gui = ti.GUI("Fluid Sim", res=(600, 300))

init()

while gui.running:
    update()
    gui.circles(pos.to_numpy(), radius=2)
    gui.show()