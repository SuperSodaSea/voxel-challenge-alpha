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
    i = 0
    while i < 256:
        d = ti.cast(4 * (ti.random() ** 2) + 2, ti.i32)
        x0, y0 = ti.cast((13 - d) * ti.random() + 2, ti.i32), ti.cast(12 * ti.random() + 2, ti.i32)
        ok = True
        for x, y in ti.ndrange((x0 - 1, x0 + d + 1), (y0 - 1, y0 + 2)):
            if ore[y, x]: ok = False
        if ok:
            for x in range(x0, x0 + d): ore[y0, x] = 1
        i += 1
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
        for x1, y1, z1 in ti.ndrange(size, size, size):
            set(vec3(x + x1, -64 + y1, z + z1), mat, vec3(0.2, 0.7, 0.6), vec3(0.1))
    
    initializeOre()
    px, py, pz, s = -40, -64, -24, 4
    for P in ti.grouped(ti.ndrange((px + s - 1, px + 15 * s + 1), (py + s - 1, py + 15 * s + 1),
        (pz + s - 1, pz + 15 * s + 1))): set(P, 2, vec3(0.2, 0.7, 0.6))
    for x, y in ti.ndrange(16, 16):
        if not ore[y, x]:
            color, noise = vec3(0.2) + ti.random() * vec3(0.1), vec3(0.05)
            for x1, y1, z1 in ti.ndrange(s, s, s):
                set(vec3(px + s * (15 - x) + x1, py + s * (15 - y) + y1, pz + z1), 1, color, noise)
                set(vec3(px + s * x + x1, py + s * (15 - y) + y1, pz + 15 * s + z1), 1, color, noise)
                set(vec3(px + s * (15 - x) + x1, py + y1, pz + s * y + z1), 1, color, noise)
                set(vec3(px + s * x + x1, py + 15 * s + y1, pz + s * y + z1), 1, color, noise)
                set(vec3(px + x1, py + s * (15 - y) + y1, pz + s * x + z1), 1, color, noise)
                set(vec3(px + 15 * s + x1, py + s * (15 - y) + y1, pz + s * (15 - x) + z1), 1, color, noise)
    
    initializePickaxe()
    px, py, pz, s = 0, -18, -56, 4
    for x, y in ti.ndrange(13, 13):
        d = pickaxe[y, x]
        if d != 0:
            color, noise = pickaxePalette[d] + ti.random() * vec3(0.05), vec3(0.05)
            mat = 2 if d == 5 else 1
            for x1, y1, z1 in ti.ndrange(s, s, s):
                set(vec3(px + x1, py + s * y + y1, pz + s * x + z1), mat, color, noise)

initialize(); scene.finish()
