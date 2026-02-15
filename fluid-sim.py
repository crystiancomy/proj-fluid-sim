import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan, debug=True)

particles = 1000

# basic forces 
n = particles
dt = 0.005
gravity = tm.vec2(0, -9.8)
energyLoss = -0.8
radius = 0.02
stiffness = 50

# basic vectors
pos = ti.Vector.field(2, dtype=float, shape=n)
vel = ti.Vector.field(2, dtype=float, shape=n)
force = ti.Vector.field(2, dtype=float, shape=n)
mass = ti.field(dtype=float, shape=n)

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
        force[i] = mass[i] * gravity


    # particle collision
    for i in range(n):
        for j in range(i + 1, n):
            vdiff = pos[i] - pos[j]
            distance = vdiff.norm()
            
            if distance > 1e-5 and distance < radius:
                normal = vdiff / distance
                penetration = radius - distance
                F = stiffness * penetration * normal
                force[i] += F
                force[j] -= F
                    
    #integration
    for i in range(n):
        vel[i] += dt * force[i] / mass[i]
        pos[i] += dt * vel[i]
        
        
        # colision border 
        if pos[i].x < 0:
            pos[i].x = 0
            vel[i].x *= energyLoss
        if pos[i].x > 1:
            pos[i].x = 1
            vel[i].x *= energyLoss
        if pos[i].y < 0:
            pos[i].y = 0
            vel[i].y *= energyLoss
        if pos[i].y > 1:
            pos[i].y = 1
            vel[i].y *= energyLoss


gui = ti.GUI("Fluid Sim", res=(900, 600))

init()

while gui.running:
    update()
    gui.circles(pos.to_numpy(), radius=3)
    gui.show()