from scene import Scene; import taichi as ti; from taichi.math import *

scene = Scene(0, 2)
scene.set_background_color((0.2, 0.2, 0.2))
scene.set_floor(-1, (0.75, 0.75, 0.75))
scene.set_directional_light((1, 0.5, -1), 0.3, (0.85, 0.9, 0.9))

@ti.func
def set(idx, mat, color, noise = vec3(0)): scene.set_voxel(idx, mat, color + ti.random() * noise)

ore = ti.field(ti.i32, (16, 16))
@ti.func
def initializeOre():
    ti.loop_config(serialize = True)
    for _ in range(256):
        d = ti.cast(4 * (ti.random() ** 2) + 2, ti.i32)
        x0, y0 = ti.cast((13 - d) * ti.random() + 2, ti.i32), ti.cast(12 * ti.random() + 2, ti.i32)
        ok = True
        for x, y in ti.ndrange((x0 - 1, x0 + d + 1), (y0 - 1, y0 + 2)):
            if ore[y, x]: ok = False
        if ok:
            for x in range(x0, x0 + d): ore[y0, x] = 1
    for x, y in ti.ndrange((2, 14), (2, 14)):
        if ti.random() < 0.1 and ore[y, x - 1] == 0 and ore[y, x + 1] == 0: ore[y, x] = 2

pickaxePaletteData = [vec3(0), vec3(0.1, 0.08, 0.05), vec3(0.15, 0.1, 0.05), vec3(0.5, 0.4, 0.2),
    vec3(0, 0.1, 0.1), vec3(0.2, 0.7, 0.6)]
pickaxePalette = vec3.field(shape = len(pickaxePaletteData))
for i in range(len(pickaxePaletteData)): pickaxePalette[i] = pickaxePaletteData[i]
pickaxe = ti.field(ti.i32, (13, 13))
@ti.func
def initializePickaxe():
    pickaxe[0, 0] = 1
    for i in range(11): pickaxe[i, i + 1] = 1; pickaxe[i + 1, i] = 2; pickaxe[i + 1, i + 1] = 3
    for i in range(20):
        a = i / 9 * 0.25 * pi
        x, y = ti.min(ti.cast(8 * ti.cos(a), ti.i32), 7), ti.min(ti.cast(8 * ti.sin(a), ti.i32), 7)
        pickaxe[4 + y, 4 + x] = 5; pickaxe[4 + x, 4 + y] = 5
    for y, x in ti.ndrange(13, 13):
        if pickaxe[y, x] == 0 and ((x > 0 and pickaxe[y, x - 1] == 5) or (x < 12 and pickaxe[y, x + 1] == 5)
            or (y > 0 and pickaxe[y - 1, x] == 5) or (y < 12 and pickaxe[y + 1, x] == 5)): pickaxe[y, x] = 4

@ti.kernel
def initialize():
    for _ in range(400):
        size, mat = 2 if ti.random() < 0.2 else 1, 2 if ti.random() < 0.5 else 1
        x, z = ti.cast(24 * ti.randn() - 8, ti.i32), ti.cast(24 * ti.randn() + 8, ti.i32)
        x, z = ti.min(ti.max(x, -64), 64 - size), ti.min(ti.max(z, -64), 64 - size)
        for p in ti.grouped(ti.ndrange(size, size, size)):
            set(ivec3(x, -64, z) + p, mat, vec3(0.2, 0.7, 0.6), vec3(0.1))
    
    initializeOre()
    c, s = ivec3(-40, -64, -24), 4
    for p in ti.grouped(ti.ndrange((c.x + s - 1, c.x + 15 * s + 1), (c.y + s - 1, c.y + 15 * s + 1),
        (c.z + s - 1, c.z + 15 * s + 1))): set(p, 2, vec3(0.2, 0.7, 0.6))
    for x, y in ti.ndrange(16, 16):
        if not ore[y, x]:
            color, noise = vec3(0.2) + ti.random() * vec3(0.1), vec3(0.05)
            for p in ti.grouped(ti.ndrange(s, s, s)):
                set(c + ivec3(s * (15 - x), s * (15 - y), 0) + p, 1, color, noise)
                set(c + ivec3(s * x, s * (15 - y), 15 * s) + p, 1, color, noise)
                set(c + ivec3(s * (15 - x), 0, s * y) + p, 1, color, noise)
                set(c + ivec3(s * x, 15 * s, s * y) + p, 1, color, noise)
                set(c + ivec3(0, s * (15 - y), s * x) + p, 1, color, noise)
                set(c + ivec3(15 * s, s * (15 - y), s * (15 - x)) + p, 1, color, noise)
    
    initializePickaxe()
    c, s = ivec3(0, -18, -56), 4
    for x, y in ti.ndrange(13, 13):
        d = pickaxe[y, x]
        if d != 0:
            color, noise = pickaxePalette[d] + ti.random() * vec3(0.05), vec3(0.05)
            mat = 2 if d == 5 else 1
            for p in ti.grouped(ti.ndrange(s, s, s)):
                set(c + ivec3(0, s * y, s * x) + p, mat, color, noise)

initialize(); scene.finish()
