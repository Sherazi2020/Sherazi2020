#!/usr/bin/env python3
"""Little Lanterns channel artwork: profile picture, banner and video thumbnails.

Usage: python3 branding.py      -> writes out/brand/*.png
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import studio
from studio import FONT_DIR, FONTS, Lanterns, draw_lantern, fbm, lerp, ss

OUT = "out/brand"
GOLD = (255, 214, 150)
CREAM = (255, 244, 228)


def font(weight, size):
    return ImageFont.truetype(FONT_DIR + FONTS[weight], size)


def text(im, txt, xy, weight, size, fill=CREAM, anchor="mm", shadow=160, blur=None, spacing=0):
    """Text with a soft dark shadow so it reads on any background."""
    f = font(weight, size)
    if spacing:
        txt = (" " * spacing).join(txt)
    sh = Image.new("L", im.size, 0)
    ImageDraw.Draw(sh).text(xy, txt, font=f, fill=shadow, anchor=anchor)
    sh = sh.filter(ImageFilter.GaussianBlur(blur or max(3, size // 9)))
    im.paste((0, 0, 0), (0, 0), sh)
    ImageDraw.Draw(im).text(xy, txt, font=f, fill=fill, anchor=anchor)


def night_sky(W, H, seed=7, stars=900):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = yy / H
    img = lerp(np.array([0.02, 0.03, 0.09]), np.array([0.11, 0.10, 0.26]), (v ** 0.9)[..., None]).astype(np.float32)
    img += np.array([0.25, 0.10, 0.25], np.float32) * np.exp(-((v - 0.82) / 0.14) ** 2)[..., None] * 0.55
    tex = fbm(H, W, rng, 5, 3, aspect=1.8)
    img = lerp(img, np.array([0.22, 0.22, 0.38], np.float32), (ss(0.55, 0.9, tex) * 0.3 * (1 - v))[..., None])
    layer = np.zeros((H, W), np.float32)
    studio.stamp_points(layer, rng.uniform(0, W, stars), rng.uniform(0, H * 0.8, stars),
                        (rng.random(stars) ** 3 * 0.9 + 0.1).astype(np.float32))
    img += layer[..., None] * np.array([0.9, 0.93, 1.0], np.float32)
    return img, yy, xx, rng


def mountains(img, yy, xx, W, H):
    u = xx[0] / W
    r1 = 0.86 + 0.04 * np.sin(u * 6.1 + 1) + 0.02 * np.sin(u * 15 + 2) + 0.01 * np.sin(u * 39)
    r2 = 0.80 + 0.05 * np.sin(u * 4.3 + 4) + 0.02 * np.sin(u * 21 + 1)
    img[:] = lerp(img, np.array([0.05, 0.07, 0.16], np.float32), ss(r2 * H - 1, r2 * H + 1, yy)[..., None])
    img[:] = lerp(img, np.array([0.015, 0.02, 0.05], np.float32), ss(r1 * H - 1, r1 * H + 1, yy)[..., None])


def to_pil(img):
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def profile_picture(size=800):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.hypot(xx - size / 2, yy - size / 2) / (size / 2)
    img = lerp(np.array([0.16, 0.10, 0.24]), np.array([0.03, 0.03, 0.10]), np.clip(d, 0, 1)[..., None] ** 0.8)
    img = img.astype(np.float32)
    rng = np.random.default_rng(3)
    layer = np.zeros((size, size), np.float32)
    n = 140
    studio.stamp_points(layer, rng.uniform(0, size, n), rng.uniform(0, size, n),
                        (rng.random(n) ** 2 * 0.7 + 0.1).astype(np.float32))
    img += layer[..., None] * 0.8
    draw_lantern(img, size * 0.27, size * 0.30, size * 0.09, 0.9, 0.85)
    draw_lantern(img, size * 0.76, size * 0.24, size * 0.065, 0.9, 0.7)
    draw_lantern(img, size * 0.50, size * 0.50, size * 0.40, 1.0, 1.0)
    return to_pil(img)


def banner(W=2560, H=1440):
    img, yy, xx, rng = night_sky(W, H)
    lan = Lanterns(rng, W, H, 34, ymin=0.02, ymax=0.88, smin=8, smax=46, avoid=(0.30, 0.70))
    lan.draw(img, 0.0)
    for it in Lanterns(rng, W, H, 6, ymin=0.37, ymax=0.62, smin=6, smax=14).items:
        draw_lantern(img, it["x"], it["y0"], it["s"], 0.9, 0.6)
    mountains(img, yy, xx, W, H)
    im = to_pil(img)
    # glowing lantern next to the name, all inside YouTube's 1546x423 safe area
    big = np.asarray(im, np.float32) / 255
    draw_lantern(big, W / 2 - 520, H / 2 - 20, 120, 1.0, 1.0)
    im = to_pil(big)
    text(im, "Little Lanterns", (W / 2 + 80, H / 2 - 50), "xlight", 150)
    text(im, "Calm music for sleep, focus & healing", (W / 2 + 80, H / 2 + 65), "light", 50, GOLD)
    text(im, "N E W   V I D E O S   E V E R Y   W E E K", (W / 2 + 80, H / 2 + 140), "med", 26, (230, 220, 210))
    return im


THUMBS = {
    "sleep": (20.0, ["FALL ASLEEP", "in minutes"], "432 Hz"),
    "breathe": (4.0, ["BREATHE", "let go of anxiety"], "GUIDED"),
    "focus": (8.0, ["DEEP FOCUS", "rain + soft piano"], "STUDY"),
    "ocean": (16.0, ["LET IT GO", "ocean sunset calm"], "STRESS RELIEF"),
}


def thumbnail(theme):
    cls, _, W, H, dur, kw = studio.THEMES[theme]
    t, (line1, line2), badge = THUMBS[theme]
    sc = cls(W, H, dur, np.ones(int(dur * studio.FPS)) * 0.5, **kw)
    sc.notext = True
    frame = sc.frame(int(t * studio.FPS))
    if theme == "breathe":  # move the glowing circle right so the words sit on dark sky
        shift = int(W * 0.24)
        frame = np.concatenate([np.repeat(frame[:, :1], shift, 1), frame[:, :-shift]], 1)
    im = Image.fromarray(frame).resize((1280, 720), Image.LANCZOS)
    # darken the left side so the big words pop on a phone screen
    grad = np.clip(1 - np.linspace(0, 1, 1280) * 1.6, 0, 1) ** 1.5 * 150
    im.paste((0, 0, 0), (0, 0), Image.fromarray(np.tile(grad.astype(np.uint8), (720, 1))))
    x = 70
    text(im, line1, (x, 300), "xbold", 112 if len(line1) <= 10 else 96, CREAM, anchor="ls", shadow=220)
    text(im, line2, (x + 4, 380), "light", 52, GOLD, anchor="ls", shadow=220)
    f = font("bold", 30)
    bw = int(f.getlength(badge)) + 44
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([x, 425, x + bw, 475], 25, fill=(255, 176, 90))
    d.text((x + bw / 2, 451), badge, font=f, fill=(45, 20, 5), anchor="mm")
    arr = np.asarray(im, np.float32) / 255
    draw_lantern(arr, x + 18, 640, 34, 1.0, 1.0)
    im = to_pil(arr)
    text(im, "LITTLE LANTERNS", (x + 50, 652), "semi", 26, (255, 236, 210), anchor="ls", spacing=0)
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    profile_picture().save(f"{OUT}/profile_800.png")
    banner().save(f"{OUT}/banner_2560x1440.png")
    for th in THUMBS:
        thumbnail(th).save(f"{OUT}/thumb_{th}.png")
    print("done:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
