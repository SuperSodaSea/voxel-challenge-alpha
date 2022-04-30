from scene import Scene; import taichi as ti; from taichi.math import *

scene = Scene(0, 2)
scene.set_background_color((0.2, 0.2, 0.2))
scene.set_floor(-1, (0.75, 0.75, 0.75))
scene.set_directional_light((1, 0.5, -1), 0.3, (0.85, 0.9, 0.9))

maskData = int('0000000006183F00180000100780030600000CF80C6000000180301000000000', 16)
mask = ti.field(ti.i32, (16, 16))
for x, y in ti.ndrange(16, 16):
    mask[y, x] = (maskData >> (16 * y + x)) & 1

@ti.func
def set(idx, mat, color, noise = vec3(0)): scene.set_voxel(idx, mat, color + ti.random() * noise)

@ti.kernel
def initialize():
    for x, z in ti.ndrange((-32, 32), (-32, 32)):
        if ti.random() < 0.01:
            mat = 2 if ti.random() < 0.7 else 1
            for x1, y1, z1 in ti.ndrange(2, 2, 2):
                set(vec3(2 * x + x1, -64 + y1, 2 * z + z1), mat, vec3(0.2, 0.7, 0.6), vec3(0.1))
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        if ti.random() < 0.005:
            mat = 2 if ti.random() < 0.8 else 1
            set(vec3(x, -64, z), mat, vec3(0.2, 0.7, 0.6), vec3(0.1))
    for x, y, z in ti.ndrange((-29, 29), (-61, -3), (-29, 29)):
        set(vec3(x, y, z), 2, vec3(0.2, 0.7, 0.6))
    px, py, pz = -32, -64, -32
    for x, y in ti.ndrange(16, 16):
        if not mask[y, x]:
            color = vec3(0.2, 0.2, 0.2) + ti.random() * vec3(0.1, 0.1, 0.1)
            noise = vec3(0.05)
            for x1, y1, z1 in ti.ndrange(4, 4, 4):
                set(vec3(px + 60 - 4 * x + x1, py + 60 - 4 * y + y1, pz + z1), 1, color, noise)
                set(vec3(px + 4 * x + x1, py + 60 - 4 * y + y1, pz + 60 + z1), 1, color, noise)
                set(vec3(px + 60 - 4 * x + x1, py + y1, pz + 4 * y + z1), 1, color, noise)
                set(vec3(px + 4 * x + x1, py + 60 + y1, pz + 4 * y + z1), 1, color, noise)
                set(vec3(px + x1, py + 60 - 4 * y + y1, pz + 4 * x + z1), 1, color, noise)
                set(vec3(px + 60 + x1, py + 60 - 4 * y + y1, pz + 60 - 4 * x + z1), 1, color, noise)

initialize(); scene.finish()
