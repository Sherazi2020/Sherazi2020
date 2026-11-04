#!/usr/bin/env python3
"""Relaxing video studio.

Generates ORIGINAL music (synthesised from scratch, no samples, copyright-free)
plus matching animated visuals, and encodes a 1080p MP4.

Usage:
    python3 studio.py <theme> [seconds]
Themes:
    sleep   - Deep Sleep Music, 432 Hz, starry night with moon & shooting stars
    breathe - Guided calm breathing (inhale 4 / exhale 6) with breathing circle
    focus   - Rainy night focus: soft piano + rain, 25:00 focus timer
    ocean   - Ocean sunset: waves + pads, "breathe with the waves"
    short   - Vertical 9:16 YouTube Short of the breathing exercise (30 s)
Output goes to ./out/<theme>.mp4 (and a thumbnail frame <theme>_thumb.png).
"""
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SR = 44100
FPS = 30
FONT_DIR = "/usr/share/fonts/opentype/inter/"
FONTS = {"thin": "Inter-Thin.otf", "xlight": "Inter-ExtraLight.otf", "light": "Inter-Light.otf",
         "reg": "Inter-Regular.otf", "med": "Inter-Medium.otf", "semi": "Inter-SemiBold.otf"}
CHANNEL = "D E E P   C A L M"


# ----------------------------------------------------------------------------- audio
def midi(n, a4=440.0):
    return a4 * 2 ** ((n - 69) / 12)


def rms(x):
    return float(np.sqrt(np.mean(x * x)) + 1e-12)


def norm(x):
    return x / rms(x)


def fft_filter(x, lo=None, hi=None):
    """Zero-phase 2nd-order-ish band limiting in the frequency domain."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    H = np.ones_like(f)
    if hi:
        H /= np.sqrt(1 + (f / hi) ** 4)
    if lo:
        H /= np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** 4)
    return np.fft.irfft(X * H, len(x))


class Track:
    def __init__(self, dur):
        self.n = int(dur * SR)
        self.L = np.zeros(self.n)
        self.R = np.zeros(self.n)

    def add(self, start, sig, pan=0.5, gain=1.0):
        i = int(start * SR)
        if i < 0:
            sig, i = sig[-i:], 0
        if i >= self.n:
            return
        sig = sig[: self.n - i] * gain
        self.L[i:i + len(sig)] += sig * math.cos(pan * math.pi / 2)
        self.R[i:i + len(sig)] += sig * math.sin(pan * math.pi / 2)


def smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def pad(f, dur, rng, att=3.0, rel=4.0, harm=4):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    env = smooth01(t / att) * smooth01((dur + rel - t) / rel)
    sig = np.zeros(n)
    for det in (-0.08, 0.0, 0.08):
        ff = f * 2 ** (det / 12)
        for h in range(1, harm + 1):
            if ff * h > 9000:
                break
            sig += np.sin(2 * np.pi * ff * h * t + rng.uniform(0, 6.28)) / h ** 1.7
    sig *= 1 + 0.12 * np.sin(2 * np.pi * rng.uniform(0.08, 0.25) * t + rng.uniform(0, 6))
    return sig * env / 3


def pluck(f, dur=5.0, bright=1.0):
    """Soft felt-piano-like tone."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for h in range(1, 9):
        fh = f * h * (1 + 0.0004 * h * h)
        if fh > 12000:
            break
        sig += np.sin(2 * np.pi * fh * t) / h ** (1.9 - 0.5 * bright) * np.exp(-t * (0.9 + 1.1 * h))
    return sig * np.clip(t / 0.006, 0, 1)


def bell(f, dur=7.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for ratio, amp, rate in ((1, 1, 0.55), (2.0, 0.22, 1.1), (2.76, 0.3, 1.6), (5.4, 0.1, 3.0)):
        sig += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t * rate)
    return sig * np.clip(t / 0.003, 0, 1)


def reverb(L, R, decay=4.0, mix=0.5, seed=3):
    rng = np.random.default_rng(seed)
    n = int(SR * decay * 1.3)
    tt = np.arange(n) / SR
    env = np.exp(-tt * 6.9 / decay)
    pre = int(0.025 * SR)
    irs = []
    for _ in range(2):
        ir = fft_filter(rng.standard_normal(n), hi=4500) * env
        ir[:pre] = 0
        irs.append(ir)
    N = len(L) + n

    def conv(x, ir):
        return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[: len(x)]

    wl = conv(L, irs[0]) + 0.35 * conv(R, irs[1])
    wr = conv(R, irs[1]) + 0.35 * conv(L, irs[0])
    k = rms(L + R) / rms(wl + wr)
    return L * (1 - 0.4 * mix) + wl * k * mix, R * (1 - 0.4 * mix) + wr * k * mix


def noise_bed(n, rng, lo, hi):
    return norm(fft_filter(rng.standard_normal(n), lo, hi))


def rain_audio(dur, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = []
    for _ in range(2):
        hiss = noise_bed(n, rng, 400, 7000) * (0.9 + 0.1 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 6)))
        drops = np.zeros(n)
        bl = int(0.01 * SR)
        bt = np.arange(bl) / SR
        for _ in range(int(dur * 35)):
            i = rng.integers(0, n - bl)
            drops[i:i + bl] += rng.standard_normal(bl) * np.exp(-bt / 0.0016) * rng.uniform(0.3, 1.0)
        drops = fft_filter(drops, lo=1500, hi=9000)
        out.append(hiss + norm(drops) * 0.6)
    return norm(out[0]), norm(out[1])


def swell(t, period=9.0):
    return 0.5 - 0.5 * np.cos(2 * np.pi * t / period)


def ocean_audio(dur, rng, period=9.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    chans = []
    for ph in (0.0, 0.35):
        s = swell(t - ph, period)
        body = noise_bed(n, rng, 50, 700) * (0.15 + 0.85 * s ** 1.6)
        crest = noise_bed(n, rng, 1200, 7000) * swell(t - ph - 0.6, period) ** 5
        chans.append(norm(body + 0.5 * crest))
    return chans[0], chans[1]


def breath(t, cycle=10.0, inh=4.0):
    """Returns 0..1 size of the breath (vectorised)."""
    ph = np.mod(t, cycle)
    inhale = 0.5 - 0.5 * np.cos(np.pi * np.clip(ph / inh, 0, 1))
    exhale = 0.5 + 0.5 * np.cos(np.pi * np.clip((ph - inh) / (cycle - inh), 0, 1))
    return np.where(ph < inh, inhale, exhale)


def chord_pads(tr, chords, seg, dur, rng, a4=440.0, att=4, rel=6):
    i, t0 = 0, 0.0
    while t0 < dur:
        ch = chords[i % len(chords)]
        for k, n in enumerate(ch):
            g = 0.9 if n < 45 else 0.55
            tr.add(t0, pad(midi(n, a4), seg, rng, att=att, rel=rel, harm=3 if n < 45 else 4),
                   pan=0.2 + 0.6 * k / max(1, len(ch) - 1), gain=g)
        i += 1
        t0 += seg


def mixdown(layers, dur):
    """layers: list of (L, R, level_dB). Each layer is RMS-normalised first."""
    n = int(dur * SR)
    L, R = np.zeros(n), np.zeros(n)
    for l, r, db in layers:
        k = 10 ** (db / 20) / rms(l + r) * 1.414
        L += l[:n] * k
        R += r[:n] * k
    t = np.arange(n) / SR
    fade = smooth01(t / 2.5) * smooth01((dur - t) / 3.5)
    L, R = L * fade, R * fade
    k = 0.11 / rms(L + R) * 1.414          # about -19 dBFS RMS, gentle for relaxing music
    L, R = np.tanh(L * k * 1.3) / 1.3, np.tanh(R * k * 1.3) / 1.3
    peak = max(np.abs(L).max(), np.abs(R).max())
    if peak > 0.89:
        L, R = L * 0.89 / peak, R * 0.89 / peak
    return L, R


def music_sleep(dur):
    rng = np.random.default_rng(11)
    a4 = 432.0
    pads = Track(dur)
    chord_pads(pads, [[38, 50, 57, 64, 66], [35, 47, 54, 62, 66], [31, 43, 50, 57, 62, 66],
                      [33, 45, 52, 57, 61, 64]], 12, dur, rng, a4)
    bells = Track(dur)
    t = 5.0
    while t < dur - 6:
        bells.add(t, bell(midi(rng.choice([74, 76, 78, 81, 83, 86]), a4)), pan=rng.uniform(.2, .8))
        t += rng.uniform(3.5, 7.0)
    pl, pr = reverb(pads.L, pads.R, decay=6, mix=0.55)
    bl, br = reverb(bells.L, bells.R, decay=7, mix=0.7, seed=5)
    n = pads.n
    bed = noise_bed(n, rng, 20, 250), noise_bed(n, rng, 20, 250)
    return mixdown([(pl, pr, 0), (bl, br, -9), (bed[0], bed[1], -20)], dur)


def music_breathe(dur, cycle=10.0, inh=4.0):
    rng = np.random.default_rng(21)
    pads = Track(dur)
    chord_pads(pads, [[48, 55, 59, 64], [45, 52, 59, 60, 64], [41, 53, 57, 60, 64], [43, 50, 55, 57, 62]],
               cycle * 2, dur, rng, att=3, rel=5)
    t = np.arange(pads.n) / SR
    b = breath(t, cycle, inh)
    pads.L *= 0.55 + 0.45 * b
    pads.R *= 0.55 + 0.45 * b
    chimes = Track(dur)
    c = 0.0
    while c < dur - 2:
        chimes.add(c, bell(midi(79), 6), pan=0.45)          # inhale cue (high)
        chimes.add(c + inh, bell(midi(72), 6), pan=0.55)    # exhale cue (lower)
        c += cycle
    pl, pr = reverb(pads.L, pads.R, decay=5, mix=0.5)
    cl, cr = reverb(chimes.L, chimes.R, decay=5, mix=0.55, seed=7)
    bed = noise_bed(pads.n, rng, 20, 300), noise_bed(pads.n, rng, 20, 300)
    return mixdown([(pl, pr, 0), (cl, cr, -6), (bed[0], bed[1], -22)], dur)


def music_focus(dur):
    rng = np.random.default_rng(31)
    beat = 60 / 68
    bar = beat * 4
    chords = [[53, 57, 60, 64], [52, 55, 59, 62], [50, 53, 57, 60, 64], [48, 52, 55, 59]]
    roots = [41, 40, 38, 36]
    piano, bass, soft = Track(dur), Track(dur), Track(dur)
    melody_scale = [72, 74, 76, 79, 81, 84]
    i, t0 = 0, 1.0
    while t0 < dur:
        ch = chords[i % 4]
        for k, n in enumerate(ch):
            piano.add(t0 + k * 0.025, pluck(midi(n), 4.5), pan=0.35 + 0.3 * k / len(ch), gain=0.7)
            piano.add(t0 + 2.5 * beat + k * 0.02, pluck(midi(n), 3.0), pan=0.35 + 0.3 * k / len(ch), gain=0.35)
        bl_t = np.arange(int(bar * SR)) / SR
        bass.add(t0, np.sin(2 * np.pi * midi(roots[i % 4] - 12) * bl_t) * np.exp(-bl_t * 0.7)
                 * np.clip(bl_t / 0.02, 0, 1), pan=0.5)
        for s in range(8):
            if rng.random() < 0.28 and i > 0:
                piano.add(t0 + s * beat / 2, pluck(midi(rng.choice(melody_scale)), 3.5, bright=1.3),
                          pan=rng.uniform(0.3, 0.7), gain=0.45)
        i += 1
        t0 += bar
    chord_pads(soft, [[n - 12 for n in c] for c in chords], bar, dur, rng, att=1.5, rel=2.5)
    pl, pr = reverb(piano.L, piano.R, decay=2.8, mix=0.4)
    sl, sr_ = reverb(soft.L, soft.R, decay=4, mix=0.5, seed=8)
    rl, rr = rain_audio(dur, rng)
    return mixdown([(pl, pr, 0), (bass.L, bass.R, -8), (sl, sr_, -14), (rl, rr, -7)], dur)


def music_ocean(dur):
    rng = np.random.default_rng(41)
    pads = Track(dur)
    chord_pads(pads, [[39, 51, 58, 62, 65], [44, 56, 60, 63, 67], [36, 48, 55, 58, 62], [46, 53, 58, 60, 65]],
               14, dur, rng, att=5, rel=7)
    bells = Track(dur)
    t = 7.0
    while t < dur - 6:
        bells.add(t, bell(midi(rng.choice([75, 77, 79, 82, 84]))), pan=rng.uniform(.25, .75))
        t += rng.uniform(5, 9)
    pl, pr = reverb(pads.L, pads.R, decay=6, mix=0.55)
    bl, br = reverb(bells.L, bells.R, decay=7, mix=0.7, seed=9)
    ol, orr = ocean_audio(dur, rng)
    return mixdown([(pl, pr, 0), (bl, br, -11), (ol, orr, -3)], dur)


def write_wav(path, L, R):
    out = (np.stack([L, R], 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(out.tobytes())


def frame_envelope(L, R, dur):
    mono = np.abs(L + R)
    spf = SR // FPS
    nf = int(dur * FPS)
    env = np.array([mono[i * spf:(i + 1) * spf].mean() if i * spf < len(mono) else 0 for i in range(nf)])
    env = np.convolve(env, np.ones(9) / 9, "same")
    return env / (env.max() + 1e-9)


# ----------------------------------------------------------------------------- visuals
def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def fbm(h, w, rng, octaves=5, base=3, aspect=1.0):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        nx = base * 2 ** o
        ny = max(2, int(nx * h / w * aspect))
        small = rng.random((ny + 1, nx + 1)).astype(np.float32)
        out += np.asarray(Image.fromarray(small, "F").resize((w, h), Image.BICUBIC)) * amp
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min())


def ramp(t, t_in, t_out, f=1.2):
    return float(ss(t_in, t_in + f, t) * ss(t_out, t_out - f, t))


def lerp(a, b, k):
    return a + (b - a) * k


class Scene:
    def __init__(self, W, H, dur, env, seed=1):
        self.W, self.H, self.dur, self.env = W, H, dur, env
        self.w, self.h = W // 2, H // 2
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.xx, self.yy = xx, yy
        self.u, self.v = xx / self.w, yy / self.h
        self.m = min(self.w, self.h)
        r2 = ((self.u - 0.5) ** 2 + (self.v - 0.5) ** 2) * 2
        self.vig = (1 - 0.55 * r2 ** 1.3).astype(np.float32)[..., None]
        self.rng = np.random.default_rng(seed)
        self.dither = [self.rng.uniform(-0.6, 0.6, (self.h, self.w, 3)).astype(np.float32) for _ in range(4)]
        self.sprites = {}

    # text --------------------------------------------------------------------------
    def spr(self, txt, weight, size, color=(1, 1, 1), shadow=0.6):
        key = (txt, weight, size, color, shadow)
        if key not in self.sprites:
            f = ImageFont.truetype(FONT_DIR + FONTS[weight], size)
            x0, y0, x1, y1 = f.getbbox(txt)
            pad = max(20, size // 3)
            im = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
            ImageDraw.Draw(im).text((pad - x0, pad - y0), txt, font=f, fill=255)
            a = np.asarray(im, np.float32) / 255
            g = np.asarray(im.filter(ImageFilter.GaussianBlur(max(4, size / 7))), np.float32) / 255
            g = np.clip(g * 1.6, 0, 1) * shadow
            self.sprites[key] = (a, g, np.array(color, np.float32) * 255, pad)
        return self.sprites[key]

    def text(self, full, txt, cx, cy, alpha, weight="light", size=40, color=(1, 1, 1), shadow=0.6,
             align="center"):
        alpha *= self.fade
        if alpha <= 0.004:
            return
        a, g, col, pad = self.spr(txt, weight, size, color, shadow)
        h, w = a.shape
        x0 = int(round(cx - w / 2)) if align == "center" else int(cx) - pad
        y0 = int(round(cy - h / 2))
        xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(self.W, x0 + w), min(self.H, y0 + h)
        if xs1 <= xs0 or ys1 <= ys0:
            return
        A = a[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0][..., None] * alpha
        G = g[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0][..., None] * alpha
        reg = full[ys0:ys1, xs0:xs1].astype(np.float32)
        reg *= 1 - G
        reg += A * (col - reg)
        full[ys0:ys1, xs0:xs1] = np.clip(reg, 0, 255).astype(np.uint8)

    def rect(self, full, x0, y0, x1, y1, color, alpha):
        alpha *= self.fade
        reg = full[y0:y1, x0:x1].astype(np.float32)
        reg += alpha * (np.array(color, np.float32) * 255 - reg)
        full[y0:y1, x0:x1] = reg.astype(np.uint8)

    def watermark(self, full, t):
        self.text(full, CHANNEL, self.W * 0.035, self.H * 0.06, 0.55 * ramp(t, 0.5, 1e9), "med",
                  int(self.H * 0.019 if self.W > self.H else self.W * 0.026), align="left", shadow=0.4)

    def finish(self, img, fi):
        img = img * self.vig * self.fade
        u8 = np.clip(img * 255 + self.dither[fi % 4], 0, 255).astype(np.uint8)
        return np.array(Image.fromarray(u8).resize((self.W, self.H), Image.BILINEAR))

    def frame(self, fi):
        t = fi / FPS
        self.fade = float(ss(0, 1.5, t) * ss(self.dur, self.dur - 2.5, t))
        return self.draw(t, fi)


def stamp_points(layer, xs, ys, vals, glow=True):
    h, w = layer.shape
    ix, iy = xs.astype(int), ys.astype(int)
    ok = (ix >= 1) & (ix < w - 1) & (iy >= 1) & (iy < h - 1)
    ix, iy, vals = ix[ok], iy[ok], vals[ok]
    np.add.at(layer, (iy, ix), vals)
    if glow:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            np.add.at(layer, (iy + dy, ix + dx), vals * 0.35)


class SleepScene(Scene):
    def __init__(self, *a):
        super().__init__(*a, seed=2)
        w, h, u, v, rng = self.w, self.h, self.u, self.v, self.rng
        top, bot = np.array([0.02, 0.035, 0.10]), np.array([0.10, 0.12, 0.28])
        self.sky = lerp(top, bot, (v ** 0.9)[..., None]).astype(np.float32)
        self.sky += np.array([0.20, 0.10, 0.28], np.float32) * np.exp(-((v - 0.80) / 0.13) ** 2)[..., None] * 0.6
        n = 650
        self.sx = rng.uniform(0, w * 1.4, n)
        self.sy = rng.uniform(0, h * 0.82, n)
        self.sb = rng.random(n) ** 3 * 0.9 + 0.08
        self.sf = rng.uniform(0.15, 1.3, n)
        self.sp = rng.uniform(0, 6.28, n)
        self.mx, self.my, self.mr = w * 0.73, h * 0.25, self.m * 0.075
        d = np.hypot(self.xx - self.mx, self.yy - self.my)
        craters = fbm(h, w, rng, 4, 20)
        self.moon_mask = ss(self.mr + 1.2, self.mr - 1.2, d)[..., None]
        self.moon_col = (np.array([1.0, 0.96, 0.86], np.float32) * (0.88 + 0.12 * craters)[..., None])
        self.halo = (np.exp(-np.maximum(d - self.mr, 0) / (self.mr * 1.6)) * 0.30
                     + np.exp(-np.maximum(d - self.mr, 0) / (self.mr * 6)) * 0.10)[..., None]
        tex = fbm(h, w * 2, rng, 5, 3, aspect=1.8)
        self.clouds = np.concatenate([tex, tex[:, ::-1]], 1)
        self.moonlit = np.exp(-d / (self.m * 0.45))[..., None]
        x = u
        r1 = 0.80 + 0.045 * np.sin(x[0] * 6.1 + 1) + 0.025 * np.sin(x[0] * 15 + 2) + 0.012 * np.sin(x[0] * 39)
        r2 = 0.74 + 0.06 * np.sin(x[0] * 4.3 + 4) + 0.02 * np.sin(x[0] * 21 + 1) + 0.01 * np.sin(x[0] * 47)
        self.mtn1 = ss(r1 * h - 0.8, r1 * h + 0.8, self.yy)[..., None]
        self.mtn2 = ss(r2 * h - 0.8, r2 * h + 0.8, self.yy)[..., None]
        self.mist = (np.exp(-((v - 0.79) / 0.05) ** 2) * 0.07)[..., None]
        self.shoot = [(s, rng.uniform(0.15, 0.55), rng.uniform(0.08, 0.3)) for s in np.arange(17, self.dur - 5, 23)]
        self.lines = ["Close your eyes", "Let your shoulders soften", "Breathe slowly and deeply",
                      "There is nothing you need to do", "You are safe to rest"]

    def draw(self, t, fi):
        w, h = self.w, self.h
        img = self.sky.copy()
        stars = np.zeros((h, w), np.float32)
        xs = np.mod(self.sx - t * 1.2, w * 1.4)
        vals = (self.sb * (0.55 + 0.45 * np.sin(2 * np.pi * self.sf * t + self.sp))).astype(np.float32)
        stamp_points(stars, xs, self.sy, vals)
        lay = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(lay)
        for st, x0, y0 in self.shoot:
            a = (t - st) / 1.1
            if 0 <= a <= 1:
                hx, hy = (x0 + 0.35 * a) * w, (y0 + 0.16 * a) * h
                for k in range(14):
                    p = k / 14
                    dr.line([(hx - 0.025 * w * p * 3, hy - 0.0115 * h * p * 3),
                             (hx - 0.025 * w * (p + 1 / 14) * 3, hy - 0.0115 * h * (p + 1 / 14) * 3)],
                            fill=int(255 * (1 - p) * math.sin(math.pi * a)), width=1)
        stars += np.asarray(lay, np.float32) / 255
        img += stars[..., None] * np.array([0.9, 0.93, 1.0], np.float32)
        img += self.halo * np.array([0.55, 0.65, 1.0], np.float32) * (1 + 0.06 * math.sin(t * 0.5))
        img = lerp(img, self.moon_col, self.moon_mask)
        off = int(t * 3) % self.clouds.shape[1]
        idx = (np.arange(w) + off) % self.clouds.shape[1]
        c = ss(0.52, 0.85, self.clouds[:, idx])[..., None] * 0.42 * (1 - self.v[..., None] * 0.6)
        ccol = np.array([0.20, 0.22, 0.36], np.float32) + self.moonlit * np.array([0.5, 0.5, 0.55], np.float32)
        img = lerp(img, ccol, c)
        img += self.mist
        img = lerp(img, np.array([0.05, 0.07, 0.16], np.float32), self.mtn2)
        img = lerp(img, np.array([0.015, 0.02, 0.05], np.float32), self.mtn1)
        full = self.finish(img, fi)
        W, H = self.W, self.H
        self.text(full, "Deep Sleep Music", W / 2, H * 0.40, ramp(t, 1.5, 11), "xlight", int(H * 0.085))
        self.text(full, "432 Hz  ·  Fall asleep in minutes", W / 2, H * 0.50, ramp(t, 2.5, 11), "light",
                  int(H * 0.032), (0.85, 0.88, 1.0))
        if t > 13:
            k = int((t - 13) // 8)
            self.text(full, self.lines[k % len(self.lines)], W / 2, H * 0.90,
                      0.85 * ramp(t, 13 + 8 * k, 13 + 8 * (k + 1), 1.6), "light", int(H * 0.03))
        self.watermark(full, t)
        return full


class BreatheScene(Scene):
    CYCLE, INH = 10.0, 4.0

    def __init__(self, W, H, dur, env, short=False):
        super().__init__(W, H, dur, env, seed=3)
        self.short = short
        u, v, rng = self.u, self.v, self.rng
        self.cx, self.cy = self.w / 2, self.h * (0.47 if short else 0.52)
        self.d = np.hypot(self.xx - self.cx, self.yy - self.cy)
        k = ((u + v) / 2)[..., None]
        self.bg = lerp(np.array([0.03, 0.17, 0.22]), np.array([0.09, 0.06, 0.22]), k).astype(np.float32)
        self.r0, self.r1 = self.m * (0.17 if short else 0.17), self.m * (0.36 if short else 0.33)
        n = 170
        self.pa = rng.uniform(0, 6.28, n)
        self.pm = rng.uniform(1.15, 2.6, n)
        self.ps = rng.choice([-1, 1], n) * rng.uniform(0.02, 0.08, n)
        self.pb = rng.uniform(0.15, 0.6, n)

    def draw(self, t, fi):
        b = float(breath(t, self.CYCLE, self.INH))
        ph = t % self.CYCLE
        inhale = ph < self.INH
        r = self.r0 + (self.r1 - self.r0) * b
        d = self.d
        img = self.bg * (0.8 + 0.3 * b) + np.array([0.10, 0.02, 0.10], np.float32) * b
        glow = np.exp(-np.maximum(d - r, 0) / (r * 0.45))[..., None]
        img += glow * np.array([0.25, 0.55, 0.6], np.float32) * 0.55
        inner = ss(r + 1.5, r - 1.5, d)[..., None]
        kk = np.clip(d / r, 0, 1)[..., None]
        fill = lerp(np.array([0.80, 0.97, 0.96], np.float32), np.array([0.38, 0.78, 0.86], np.float32), kk ** 1.5)
        img = lerp(img, fill, inner * 0.92)
        img += (np.exp(-((d - r) / 1.3) ** 2) * 0.5)[..., None]
        cyc_start = t - ph
        for start in (cyc_start, cyc_start - self.CYCLE):
            age = t - start
            if 0 <= age < 3.5:
                rr = self.r0 + age * self.m * 0.12
                img += (np.exp(-((d - rr) / 1.6) ** 2) * 0.28 * (1 - age / 3.5))[..., None]
        layer = np.zeros((self.h, self.w), np.float32)
        ang = self.pa + t * self.ps
        rad = self.r1 * self.pm * (1.35 - 0.45 * b)
        tw = self.pb * (0.6 + 0.4 * np.sin(t * 1.7 + self.pa * 5))
        stamp_points(layer, self.cx + np.cos(ang) * rad, self.cy + np.sin(ang) * rad, tw.astype(np.float32))
        img += layer[..., None] * np.array([0.8, 0.95, 1.0], np.float32)
        full = self.finish(img, fi)
        W, H = self.W, self.H
        S = H if not self.short else W * 1.0
        cy = self.cy * 2
        sec = int(ph) + 1 if inhale else int(ph - self.INH) + 1
        frac = ph % 1
        self.text(full, str(sec), W / 2, cy, 0.85 * (1 - 0.5 * frac) * ramp(t, 1, 1e9, 0.6), "xlight",
                  int(S * (0.11 if not self.short else 0.14)), (0.05, 0.16, 0.22), shadow=0)
        a_in = float(ss(0, 0.5, ph) * ss(self.INH, self.INH - 0.5, ph))
        a_out = float(ss(self.INH, self.INH + 0.5, ph) * ss(self.CYCLE, self.CYCLE - 0.5, ph))
        ty = min(cy + self.r1 * 2 + S * 0.10, H * 0.89)
        ts = int(S * (0.055 if not self.short else 0.075))
        self.text(full, "Breathe in", W / 2, ty, a_in * ramp(t, 0.5, 1e9), "light", ts)
        self.text(full, "Breathe out", W / 2, ty, a_out * ramp(t, 0.5, 1e9), "light", ts)
        if not self.short:
            self.text(full, "Calm Breathing", W / 2, H * 0.085, 0.95 * ramp(t, 0.8, 1e9), "light", int(H * 0.048))
            self.text(full, "Inhale 4  ·  Exhale 6  ·  Follow the circle", W / 2, H * 0.145,
                      0.7 * ramp(t, 1.5, 1e9), "reg", int(H * 0.024), (0.85, 0.95, 1.0))
            self.watermark(full, t)
        else:
            hs = int(W * 0.075)
            self.text(full, "Feeling stressed?", W / 2, H * 0.12, ramp(t, 0.3, 4.2, 0.6), "semi", hs)
            self.text(full, "Breathe with me", W / 2, H * 0.12, ramp(t, 4.5, self.dur - 5, 0.6), "semi", hs)
            self.text(full, "for 30 seconds", W / 2, H * 0.12 + hs * 1.25, 0.85 * ramp(t, 4.8, self.dur - 5, 0.6),
                      "light", int(hs * 0.7))
            self.text(full, "Feel calmer?", W / 2, H * 0.12, ramp(t, self.dur - 4.7, 1e9, 0.6), "semi", hs)
            self.text(full, "Follow for daily calm", W / 2, H * 0.12 + hs * 1.25,
                      ramp(t, self.dur - 4.4, 1e9, 0.6), "light", int(hs * 0.7))
            self.text(full, CHANNEL, W / 2, H * 0.94, 0.6 * ramp(t, 0.5, 1e9), "med", int(W * 0.03), shadow=0.4)
        return full


class FocusScene(Scene):
    SESSION = 25 * 60

    def __init__(self, *a):
        super().__init__(*a, seed=4)
        w, h, v, rng = self.w, self.h, self.v, self.rng
        self.bg = lerp(np.array([0.035, 0.045, 0.08]), np.array([0.11, 0.07, 0.06]), v[..., None]).astype(np.float32)
        pal = np.array([[1, .62, .28], [1, .78, .42], [.35, .75, .9], [1, .45, .35], [.95, .9, .7]], np.float32)
        n = 30
        self.bk = [dict(x=rng.uniform(0, w), y=rng.uniform(h * 0.25, h * 1.0), r=rng.uniform(8, 38) * self.m / 270,
                        c=pal[rng.integers(0, len(pal))], f=rng.uniform(0.05, 0.25), p=rng.uniform(0, 6.28),
                        vx=rng.uniform(-0.6, 0.6), a=rng.uniform(0.06, 0.18)) for _ in range(n)]
        nr = 260
        self.rx = rng.uniform(-0.1 * w, w, nr)
        self.ry = rng.uniform(0, h, nr)
        self.rs = rng.uniform(260, 480, nr) * self.m / 270
        self.rl = rng.uniform(8, 22, nr) * self.m / 270
        self.rb = rng.uniform(40, 110, nr)
        nd = 140
        self.dx, self.dy = rng.uniform(0, w, nd), rng.uniform(0, h, nd)
        self.dr = rng.uniform(0.6, 2.2, nd)
        self.slide = [dict(x=rng.uniform(0.05, 0.95) * w, y0=rng.uniform(-0.2, 0.5) * h, s=rng.uniform(4, 12),
                           p=rng.uniform(0, 30)) for _ in range(10)]

    def draw(self, t, fi):
        w, h = self.w, self.h
        img = self.bg.copy()
        e = float(self.env[min(fi, len(self.env) - 1)])
        for b in self.bk:
            x = (b["x"] + b["vx"] * t) % w
            r = b["r"]
            x0, x1 = int(max(0, x - r - 2)), int(min(w, x + r + 3))
            y0, y1 = int(max(0, b["y"] - r - 2)), int(min(h, b["y"] + r + 3))
            if x1 <= x0 or y1 <= y0:
                continue
            dd = np.hypot(self.xx[y0:y1, x0:x1] - x, self.yy[y0:y1, x0:x1] - b["y"])
            disc = ss(r, r - max(3.0, r * 0.25), dd) * (0.8 + 0.35 * (dd / r) ** 3)
            inten = b["a"] * (0.75 + 0.25 * math.sin(2 * math.pi * b["f"] * t + b["p"])) * (0.85 + 0.35 * e)
            img[y0:y1, x0:x1] += (disc * inten)[..., None] * b["c"]
        lay = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(lay)
        ys = np.mod(self.ry + t * self.rs, h + 40) - 20
        for x, y, l, bb in zip(self.rx, ys, self.rl, self.rb):
            xx = x + y * 0.12
            dr.line([(xx, y), (xx - l * 0.12, y - l)], fill=int(bb), width=1)
        for x, y, r in zip(self.dx, self.dy, self.dr):
            dr.ellipse([x - r, y - r, x + r, y + r], outline=55)
            dr.point((x - r * 0.4, y - r * 0.4), fill=150)
        for s in self.slide:
            ph = (t + s["p"]) % 30
            y = s["y0"] + ph * s["s"] * (1 + 0.5 * math.sin(ph * 2.3))
            dr.line([(s["x"], y - 18), (s["x"], y)], fill=35, width=2)
            dr.ellipse([s["x"] - 2, y - 2, s["x"] + 2, y + 2], fill=110)
        img += (np.asarray(lay, np.float32) / 255)[..., None] * np.array([0.75, 0.8, 0.9], np.float32)
        full = self.finish(img, fi)
        W, H = self.W, self.H
        self.text(full, "Rainy Night Focus", W * 0.06, H * 0.14, ramp(t, 0.8, 1e9), "light", int(H * 0.055),
                  align="left")
        self.text(full, "Soft piano  ·  gentle rain  ·  deep work", W * 0.06, H * 0.205, 0.75 * ramp(t, 1.4, 1e9),
                  "reg", int(H * 0.024), (0.95, 0.88, 0.8), align="left")
        left = max(0, self.SESSION - t)
        mm, sc = int(left // 60), int(left % 60)
        a = ramp(t, 2.0, 1e9)
        self.text(full, "FOCUS SESSION", W / 2, H * 0.72, 0.7 * a, "med", int(H * 0.02), (1, 0.9, 0.8))
        self.text(full, f"{mm:02d}:{sc:02d}", W / 2, H * 0.80, 0.95 * a, "thin", int(H * 0.10))
        bx0, bx1, by = int(W * 0.38), int(W * 0.62), int(H * 0.875)
        self.rect(full, bx0, by, bx1, by + 3, (1, 1, 1), 0.18 * a)
        self.rect(full, bx0, by, bx0 + max(1, int((bx1 - bx0) * t / self.SESSION)), by + 3, (1, 0.75, 0.45), 0.9 * a)
        self.watermark(full, t)
        return full


class OceanScene(Scene):
    PERIOD = 9.0

    def __init__(self, *a):
        super().__init__(*a, seed=5)
        w, h, u, v, rng = self.w, self.h, self.u, self.v, self.rng
        self.hz = hz = 0.58
        stops_v = [0, hz * 0.55, hz * 0.85, hz]
        cols = np.array([[0.10, 0.07, 0.24], [0.45, 0.22, 0.42], [0.92, 0.48, 0.42], [1.0, 0.70, 0.45]])
        sky = np.stack([np.interp(v, stops_v, cols[:, c]) for c in range(3)], -1).astype(np.float32)
        self.sx, self.sy, self.sr = w * 0.5, hz * h - self.m * 0.025, self.m * 0.085
        d = np.hypot(self.xx - self.sx, self.yy - self.sy)
        sky += (np.exp(-d / (self.sr * 2.2)) * 0.55)[..., None] * np.array([1.0, 0.55, 0.3], np.float32)
        sky += (np.exp(-d / (self.sr * 9)) * 0.18)[..., None] * np.array([1.0, 0.5, 0.35], np.float32)
        sky = lerp(sky, np.array([1.0, 0.88, 0.65], np.float32), ss(self.sr + 1.2, self.sr - 1.2, d)[..., None])
        self.sky = sky
        tex = fbm(h, w * 2, rng, 5, 2, aspect=4.0)
        self.clouds = np.concatenate([tex, tex[:, ::-1]], 1)
        self.cloudmask = (ss(0.05, 0.25, v) * ss(hz - 0.02, hz - 0.15, v))
        hy = hz * h
        p = np.clip((self.yy - hy) / (h - hy), 0, 1)
        self.p = p
        self.water = (self.yy > hy)[..., None]
        self.wbase = lerp(np.array([0.82, 0.48, 0.42], np.float32), np.array([0.07, 0.07, 0.20], np.float32),
                          (p ** 0.45)[..., None])
        self.z = 1 / (p + 0.035)
        self.reflw = np.exp(-((self.xx - self.sx) / (3 + 75 * p)) ** 2)
        self.xn = self.xx / w
        self.rowph = rng.uniform(0, 6.28, (h, 1)).astype(np.float32)
        self.lines = ["Inhale as the wave rises", "Exhale as it falls back"]

    def draw(self, t, fi):
        w = self.w
        off = int(t * 2.0) % self.clouds.shape[1]
        idx = (np.arange(w) + off) % self.clouds.shape[1]
        c = ss(0.55, 0.85, self.clouds[:, idx]) * self.cloudmask
        ccol = lerp(np.array([0.35, 0.2, 0.4], np.float32), np.array([1.0, 0.6, 0.5], np.float32), (self.v / self.hz)[..., None] ** 2)
        sky = lerp(self.sky, ccol, (c * 0.55)[..., None])
        z, p, xn = self.z, self.p, self.xn
        sw = float(swell(t, self.PERIOD))
        band = np.sin(z * 2.4 - t * 1.25 + 0.7 * np.sin(xn * 7 + z * 0.35 + t * 0.35) + self.rowph * 0.3)
        hl = ss(0.5, 1.0, band)
        rip = np.sin(xn * (14 / (p + 0.06)) + z * 3.1 + t * 1.8 + self.rowph)
        refl = self.reflw * (0.55 + 0.45 * band)
        sparkle = self.reflw * ss(0.82, 1.0, rip * 0.5 + band * 0.5)
        foam = ss(0.72, 1.0, p) * ss(0.4, 1.0, np.sin(xn * 9 + t * 0.6) * 0.5 + 0.5) * sw ** 2
        water = self.wbase + (hl * 0.07)[..., None] * np.array([1.0, 0.6, 0.6], np.float32)
        water += (refl * 0.30)[..., None] * np.array([1.0, 0.72, 0.45], np.float32)
        water += (sparkle * 0.75)[..., None] * np.array([1.0, 0.9, 0.7], np.float32)
        water += (foam * 0.12)[..., None]
        water += (np.exp(-p / 0.012) * 0.25)[..., None] * np.array([1, 0.75, 0.55], np.float32)
        img = np.where(self.water, water, sky)
        full = self.finish(img, fi)
        W, H = self.W, self.H
        self.text(full, "Ocean Sunset", W / 2, H * 0.17, ramp(t, 1.5, 12), "xlight", int(H * 0.085))
        self.text(full, "Stress relief  ·  Let the waves carry your worries", W / 2, H * 0.265, ramp(t, 2.5, 12),
                  "light", int(H * 0.03), (1, 0.93, 0.88))
        if t > 13.5:
            ph = (t / self.PERIOD) % 1
            self.text(full, self.lines[0], W / 2, H * 0.88, 0.9 * float(ss(0.0, 0.08, ph) * ss(0.5, 0.42, ph)),
                      "light", int(H * 0.034))
            self.text(full, self.lines[1], W / 2, H * 0.88, 0.9 * float(ss(0.5, 0.58, ph) * ss(1.0, 0.92, ph)),
                      "light", int(H * 0.034))
        self.watermark(full, t)
        return full


THEMES = {
    "sleep": (SleepScene, music_sleep, 1920, 1080, 60, {}),
    "breathe": (BreatheScene, music_breathe, 1920, 1080, 60, {}),
    "focus": (FocusScene, music_focus, 1920, 1080, 60, {}),
    "ocean": (OceanScene, music_ocean, 1920, 1080, 63, {}),
    "short": (BreatheScene, music_breathe, 1080, 1920, 30, {"short": True}),
}


def main():
    theme = sys.argv[1]
    cls, music, W, H, dur, kw = THEMES[theme]
    if len(sys.argv) > 2:
        dur = float(sys.argv[2])
    os.makedirs("out", exist_ok=True)
    wav, mp4 = f"out/{theme}.wav", f"out/{theme}.mp4"
    L, R = music(dur)
    write_wav(wav, L, R)
    env = frame_envelope(L, R, dur)
    scene = cls(W, H, dur, env, **kw)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-x264-params", "aq-mode=3",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nf = int(dur * FPS)
    thumb_at = int(min(dur * 0.3, 7) * FPS)
    for fi in range(nf):
        fr = scene.frame(fi)
        if fi == thumb_at:
            Image.fromarray(fr).save(f"out/{theme}_thumb.png")
        ff.stdin.write(fr.tobytes())
        if fi % (FPS * 10) == 0:
            print(f"{theme}: {fi / FPS:.0f}s / {dur:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"{theme}: done -> {mp4}")


if __name__ == "__main__":
    main()
