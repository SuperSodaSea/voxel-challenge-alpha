from scene import Scene; import taichi as ti; from taichi.math import *

scene = Scene(0, 1.2)
scene.set_background_color((0.6, 0.8, 1.0))
scene.set_floor(-1, (1.3, 1.25, 1.2))
scene.set_directional_light((1, 0.75, 0.5), 0.5, (1.2, 1.15, 1.1))

@ti.func
def set(idx, mat, color, noise = vec3(0)): scene.set_voxel(idx, mat, color + ti.random() * noise)

@ti.func
def fill(p0, s, mat, color, noise = vec3(0), paint = False):
    for p in ti.grouped(ti.ndrange((p0.x, p0.x + s.x), (p0.y, p0.y + s.y), (p0.z, p0.z + s.z))):
        if not paint or scene.get_voxel(p)[0] != 0: set(p, mat, color, noise)

@ti.func
def sphere(c, r, mat, color, noise = vec3(0)):
    for p in ti.grouped(ti.ndrange((-64, 64), (-64, 64), (-64, 64))):
        if (p - c).norm() <= r: set(p, mat, color, noise)

@ti.func
def coneY(p0, p1, r0, r1, mat, color, noise = vec3(0)):
    for p in ti.grouped(ti.ndrange((-64, 64), (ti.cast(p0.y, ti.i32), ti.cast(p1.y, ti.i32) + 1), (-64, 64))):
        c, r = mix(p0, p1, (p.y - p0.y) / (p1.y - p0.y)), mix(r0, r1, (p.y - p0.y) / (p1.y - p0.y))
        if (p - c).xz.norm() <= r: set(p, mat, color, noise)

@ti.kernel
def initialize():
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        p1 = vec2(28 - x, 31 - z); p1 = vec2(dot(p1, vec2(1.1, -1)), dot(p1, vec2(1, 1))) / ti.sqrt(2)
        h = 0.09 * p1.y - 0.0011 * p1.x * p1.x + 0.4 * (ti.sin(0.06 * x + 0.11 * z) + ti.sin(0.09 * x + 0.07 * z))
        if h >= 0: fill(ivec3(x, -64, z), ivec3(1, 4 + h, 1), 1, vec3(0.4, 0.9, 0.4), 0.2)
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        if scene.get_voxel(ivec3(x, -64, z))[0] == 0:
            for x1, z1 in ti.ndrange((x - 14, x + 15), (z - 14, z + 15)):
                if (x1 >= -64 and x1 < 64 and z1 >= -64 and z1 < 64 and (x1 - x) ** 2 + (z1 - z) ** 2 <= 14 ** 2 and
            scene.get_voxel(ivec3(x1, -64, z1))[0] > 0): set(ivec3(x, -63, z), 1, vec3(0.1, 0.2, 0.6), vec3(0, 0, 0.1))
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        if scene.get_voxel(ivec3(x, -64, z))[0] == 0 and scene.get_voxel(ivec3(x, -63, z))[0] > 0:
            set(ivec3(x, -64, z), 1, vec3(0.1, 0.2, 0.6), vec3(0, 0, 0.1))
        else:
            fill(ivec3(x, -64, z), ivec3(1, 3, 1), 1, vec3(0.8, 0.5, 0.3), 0.3)
            set(ivec3(x, -61, z), 1, vec3(0.4, 0.9, 0.4), 0.2)
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        for x1, z1 in ti.ndrange((x - 4, x + 5), (z - 4, z + 5)):
            if (x1 >= -64 and x1 < 64 and z1 >= -64 and z1 < 64 and (x1 - x) ** 2 + (z1 - z) ** 2 <= 4 ** 2
            and scene.get_voxel(ivec3(x1, -62, z1))[0] == 0): set(ivec3(x, -61, z), 0, vec3(0))
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        y = -64
        while y < 64 and scene.get_voxel(ivec3(x, y, z))[0] > 0: y += 1
        if y <= -62 and ti.random() < 0.02: set(ivec3(x, y - 1, z), 2, vec3(0.3), 0.7)
        elif y >= -60 and ti.random() < 0.015: set(ivec3(x, y, z), 1, vec3(1, 1 * ti.random() ** 4, 0), 0.2)
    
    c = ivec3(4, 0, 14)
    sphere(c + vec3(0, -32, 0), 11, 1, vec3(0.8, 0.75, 0.7), 0.2)
    fill(c + ivec3(-16, -35, -16), ivec3(33, 7, 33), 1, vec3(0.7, 0.15, 0.2), 0.1, True)
    fill(c + ivec3(-16, -33, -16), ivec3(33, 3, 33), 1, vec3(0.6, 0.1, 0.15), 0.1, True)
    sphere(c + vec3(0, 12, 0), 9, 1, vec3(0.8, 0.75, 0.7), 0.2)
    fill(c + ivec3(-16, 10, -16), ivec3(33, 8, 33), 1, vec3(0.7, 0.15, 0.2), 0.1, True)
    fill(c + ivec3(-16, 13, -16), ivec3(33, 3, 33), 1, vec3(0.6, 0.1, 0.15), 0.1, True)
    fill(c + ivec3(-16, 16, -16), ivec3(33, 1, 33), 1, vec3(0.7, 0.6, 0.4), 0.1, True)
    sphere(c + vec3(0, 38, 0), 4, 1, vec3(0.8, 0.75, 0.7), 0.2)
    fill(c + ivec3(-16, 38, -16), ivec3(33, 2, 33), 1, vec3(0.7, 0.15, 0.2), 0.1, True)
    for i in ti.static(range(3)):
        p = vec3(ti.cos(i * pi / 3 * 2), 0, ti.sin(i * pi / 3 * 2))
        coneY(c + vec3(0, -60, 0) - 3.8 * p, c + vec3(0, 12, 0) - 3.8 * p, 1.8, 1.8, 1, 1.1 * vec3(0.8, 0.7, 0.5), 0.1)
        coneY(c + vec3(0, -60, 0) + 18 * p, c + vec3(0, -32, 0), 3, 3, 1, 1.1 * vec3(0.9, 0.8, 0.6), 0.1)
        coneY(c + vec3(0, -60, 0), c + vec3(0, -52, 0) + 11 * p, 2, 2, 1, 0.7 * vec3(0.9, 0.8, 0.6), 0.1)
        sphere(c + vec3(0, -51, 0) + 12 * p, 3.6, 1, vec3(1), 0.1)
    for i in ti.static(range(5)):
        coneY(c + vec3(0, -18 + 4 * i, 0), c + vec3(0, -17 + 4 * i, 0), 3, 3, 1, 0.7 * vec3(0.7, 0.6, 0.4), 0.1)
    coneY(c + vec3(0, 12, 0), c + vec3(0, 38, 0), 2.5, 1.5, 1, vec3(0.9, 0.8, 0.6), 0.1)
    coneY(c + vec3(0, 38, 0), c + vec3(0, 53, 0), 2, 0, 1, vec3(0.9, 0.8, 0.6), 0.1)
    for y in range(50, 60): set(c + ivec3(0, y, 0), 2, vec3(0.6, 0.1, 0.1) if y % 2 == 1 else vec3(0.7))
    
    c = ivec3(-40, 0, 14)
    for p in ti.grouped(ti.ndrange((-16, 17), (-64, 64), (-16, 17))):
        a = ti.atan2(1.0 * p.z, p.x) - (p.y + 64) / 127 * pi / 3 * 2
        r = (12 + 4 * (0.5 * ti.cos(3 * a) + 0.5) ** 1.2) * (1 - (p.y + 64) / 127 * 0.5)
        if abs(a) < 0.2: r = 6
        if p.xz.norm() <= r and (p.y < 56 or p.xz.norm() >= r - 2):
            set(c + p, 1, (1.0 if p.y % 4 != 0 else 1.2 if p.y % 16 == 12 else 0.9) * vec3(0.4, 0.55, 0.65), 0.05)
            if abs(a) < 0.32: set(c + p, 1, (p.xz.norm() / r) ** 4 * 0.8 * vec3(0.4, 0.55, 0.65), 0.05)
    
    c = ivec3(16, 0, -40)
    for p in ti.grouped(ti.ndrange((-14, 15), (-64, 64), (-14, 15))):
        if abs(p.x) + abs(p.z) <= 14 and abs(p.z) <= ((63 - p.y) / 127) ** 0.7 * 15 + 1: set(c + p, 1,
            vec3(0.1, 0.2, 0.3) if p.z == 0 else vec3(0.4, 0.6, 0.8) if p.y % 4 == 0 else vec3(0.2, 0.4, 0.6), 0.1)
    fill(c + ivec3(-10, 46, -16), ivec3(20, 12, 32), 0, vec3(0))
    
    for p in ti.grouped(ti.ndrange((-64, 64), (-64, 64), (-64, 64))):
        mat, color = scene.get_voxel(p); set(p, mat, ((p.y + 192) / 256) ** 0.75 * color)

# scene.camera._camera_pos[:] = [3.0, -0.15, 5.0]; scene.camera._lookat_pos[:] = [0.0, -0.1, 0.0]
# scene.renderer.set_camera_pos(*scene.camera.position); scene.renderer.set_look_at(*scene.camera.look_at)
initialize(); scene.finish()
