from scene import Scene; import taichi as ti; from taichi.math import *

scene = Scene(0, 1.2)
scene.set_background_color((0.6, 0.8, 1.0))
scene.set_floor(-1, (1.3, 1.6, 2.0))
scene.set_directional_light((1, 1, 0.75), 0.7, (1.0, 0.9, 0.8))

@ti.func
def set(idx, mat, color, noise = vec3(0)): scene.set_voxel(idx, mat, color + ti.random() * noise)

@ti.func
def fill(p0, s, mat, color, noise = vec3(0), paint = False):
    for p in ti.grouped(ti.ndrange((p0.x, p0.x + s.x), (p0.y, p0.y + s.y), (p0.z, p0.z + s.z))):
        if not paint or scene.get_voxel(p)[0] != 0: set(p, mat, color, noise)

@ti.func
def sphere(c, r, mat, color, noise = vec3(0)):
    for p in ti.grouped(ti.ndrange((-64, 64), (-64, 64), (-64, 64))):
        if dot(p - c, p - c) <= r * r: set(p, mat, color, noise)

@ti.func
def coneY(p0, p1, r0, r1, mat, color, noise = vec3(0)):
    for p in ti.grouped(ti.ndrange((-64, 64), (ti.cast(p0.y, ti.i32), ti.cast(p1.y, ti.i32) + 1), (-64, 64))):
        c, r = mix(p0, p1, (p.y - p0.y) / (p1.y - p0.y)), mix(r0, r1, (p.y - p0.y) / (p1.y - p0.y))
        if dot((p - c).xz, (p - c).xz) <= r * r: set(p, mat, color, noise)

@ti.kernel
def initialize():
    c = ivec3(8, 0, 16)
    sphere(c + vec3(0, -32, 0), 11, 1, vec3(0.6), 0.2)
    fill(c + ivec3(-16, -35, -16), ivec3(33, 6, 33), 1, vec3(0.6, 0.1, 0.1), paint = True)
    fill(c + ivec3(-16, -33, -16), ivec3(33, 2, 33), 1, vec3(0.5, 0.1, 0.1), paint = True)
    sphere(c + vec3(0, 12, 0), 9, 1, vec3(0.6), 0.2)
    fill(c + ivec3(-16, 10, -16), ivec3(33, 8, 33), 1, vec3(0.6, 0.1, 0.1), paint = True)
    fill(c + ivec3(-16, 16, -16), ivec3(33, 1, 33), 1, vec3(0.7, 0.6, 0.4), paint = True)
    sphere(c + vec3(0, 38, 0), 4, 1, vec3(0.6), 0.2)
    fill(c + ivec3(-16, 37, -16), ivec3(33, 2, 33), 1, vec3(0.6, 0.1, 0.1), paint = True)
    for i in ti.static(range(3)):
        p = vec3(ti.cos(i * pi / 3 * 2), 0, ti.sin(i * pi / 3 * 2))
        coneY(c + vec3(0, -64, 0) - 3.8 * p, c + vec3(0, 12, 0) - 3.8 * p, 1.8, 1.8, 1, vec3(0.8, 0.7, 0.5), 0.1)
        coneY(c + vec3(0, -64, 0) + 18 * p, c + vec3(0, -32, 0), 3, 3, 1, vec3(0.9, 0.8, 0.6), 0.1)
        coneY(c + vec3(0, -64, 0), c + vec3(0, -52, 0) + 11 * p, 2, 2, 1, vec3(0.9, 0.8, 0.6), 0.1)
        sphere(c + vec3(0, -52, 0) + 11 * p, 3.6, 1, vec3(0.7), 0.1)
    for i in ti.static(range(5)):
        coneY(c + vec3(0, -18 + 4 * i, 0), c + vec3(0, -17 + 4 * i, 0), 3, 3, 1, vec3(0.7, 0.6, 0.4), 0.1)
    coneY(c + vec3(0, 12, 0), c + vec3(0, 38, 0), 2.5, 1.5, 1, vec3(0.9, 0.8, 0.6), 0.1)
    coneY(c + vec3(0, 38, 0), c + vec3(0, 59, 0), 2, 0, 1, vec3(0.9, 0.8, 0.6), 0.1)
    
    c = ivec3(-36, 0, 14)
    for p in ti.grouped(ti.ndrange((-16, 17), (-64, 64), (-16, 17))):
        a = ti.atan2(1.0 * p.z, p.x) - (p.y + 64) / 127 * pi / 3 * 2
        r = (12 + 4 * (0.5 * ti.cos(3 * a) + 0.5) ** 1.2) * (1 - (p.y + 64) / 127 * 0.5)
        if ti.abs(a) < 0.2: r = 6
        if dot(p.xz, p.xz) <= r * r and (p.y < 56 or dot(p.xz, p.xz) >= (r - 2) ** 2):
            set(c + p, 1, ((1.1 if p.y // 4 % 4 == 3 else 0.9) if p.y % 4 == 0 else 1) * vec3(0.4, 0.55, 0.65), 0.05)
    
    c = ivec3(16, 0, -36)
    for p in ti.grouped(ti.ndrange((-14, 15), (-64, 64), (-14, 15))):
        if ti.abs(p.x) + ti.abs(p.z) <= 14 and ti.abs(p.z) <= ((63 - p.y) / 127) ** 0.7 * 15 + 1: set(c + p, 1,
            vec3(0.1, 0.2, 0.3) if p.z == 0 else vec3(0.4, 0.6, 0.8) if p.y % 4 == 0 else vec3(0.2, 0.4, 0.6), 0.1)
    fill(c + ivec3(-10, 46, -16), ivec3(20, 12, 32), 0, vec3(0))
    
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        p1 = vec2(32 - x, 35 - z); p1 = vec2(dot(p1, vec2(1.1, -1)), dot(p1, vec2(1, 1))) / ti.sqrt(2)
        h = 0.05 * p1.y - 0.00075 * p1.x * p1.x + 0.2 * (ti.sin(0.06 * x + 0.11 * z) + ti.sin(0.09 * x + 0.07 * z))
        if h >= 0:
            for y in range(0, 3 + h):
                set(ivec3(x, -64 + y, z), 1, vec3(0.25, 0.5, 0.25), 0.1)
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        for x1, z1 in ti.ndrange((x - 14, x + 15), (z - 14, z + 15)):
            if (x1 >= -64 and x1 < 64 and z1 >= -64 and z1 < 64 and (x1 - x) ** 2 + (z1 - z) ** 2 <= 14 ** 2 and scene
            .get_voxel(ivec3(x1, -64, z1))[0] > 0): set(ivec3(x, -63, z), 1, vec3(0.05, 0.1, 0.3), vec3(0, 0, 0.1))
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        if scene.get_voxel(ivec3(x, -63, z))[0] > 0: set(ivec3(x, -64, z), 1, vec3(0.05, 0.1, 0.3), vec3(0, 0, 0.1))
        else: fill(ivec3(x, -64, z), ivec3(1, 3, 1), 1, vec3(0.25, 0.5, 0.25), 0.1)
    for x, z in ti.ndrange((-64, 64), (-64, 64)):
        for x1, z1 in ti.ndrange((x - 4, x + 5), (z - 4, z + 5)):
            if (x1 >= -64 and x1 < 64 and z1 >= -64 and z1 < 64 and (x1 - x) ** 2 + (z1 - z) ** 2 <= 4 ** 2 and scene
            .get_voxel(ivec3(x1, -62, z1))[0] == 0): fill(ivec3(x, -62, z), ivec3(1), 1, vec3(0.9, 0.7, 0.5), 0.1, True)
    fill(ivec3(-64, -64, -64), ivec3(128, 1, 128), 1, vec3(0.7, 0.5, 0.3), 0.1)

initialize(); scene.finish()
