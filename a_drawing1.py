#!/usr/bin/env python3
"""
Op Art Intensified Generator (Full-size, single-pass)

Generates a 10000 x 12000 px print-ready image composed of:
- Triple fine diagonal families (-21°, -20°, -19°) in red/blue/black with jitter and 1–2 px strokes
- Two complementary-color crosshatch bands (~70° and ~130°) using bold 4–6 px strokes
- Bands aligned along -20° axis (no warping in this version)

Usage:
  python opart_intensified_full.py --out opart_intensified_10000x12000.png
"""

import math, random, time, argparse
from PIL import Image, ImageDraw

def draw_parallel_lines(draw, w, h, spacing, angle_deg, colors, width_range=(1,2), jitter=0.0, alpha=255):
    """Draw evenly spaced parallel lines across the canvas at a given angle."""
    theta = math.radians(angle_deg)
    ux, uy = math.cos(theta), math.sin(theta)           
    nx, ny = -math.sin(theta), math.cos(theta)          

    corners = [(0,0),(w,0),(0,h),(w,h)]
    ds = [cx*nx + cy*ny for (cx,cy) in corners]
    dmin, dmax = min(ds), max(ds)

    d = dmin - spacing
    i = 0
    while d <= dmax + spacing:
        col_raw = colors[i % len(colors)]
        col = (*col_raw, alpha)
        jd = (random.uniform(-jitter, jitter) if jitter else 0.0)

        cx, cy = w/2.0, h/2.0
        d_center = nx*cx + ny*cy
        shift = (d + jd) - d_center
        p0x, p0y = cx + nx*shift, cy + ny*shift

        t0, t1 = -1e9, 1e9
        def clip(p, q):
            nonlocal t0, t1
            if abs(p) < 1e-9:
                if q < 0: return False
                return True
            r = q / p
            if p < 0:
                if r > t0: t0 = r
            else:
                if r < t1: t1 = r
            return t0 <= t1
        if not (clip(-ux,  p0x - 0) and
                clip( ux,  w - p0x) and
                clip(-uy,  p0y - 0) and
                clip( uy,  h - p0y)):
            d += spacing
            i += 1
            continue

        x1, y1 = p0x + t0*ux, p0y + t0*uy
        x2, y2 = p0x + t1*ux, p0y + t1*uy
        lw = random.randint(width_range[0], width_range[1])
        draw.line([(x1, y1), (x2, y2)], fill=col, width=lw)
        d += spacing
        i += 1

def draw_crosshatch_in_band(draw, w, h, spacing, angles_deg, colors, width_range, alpha,
                            band_angle_deg, band_center_offset, band_width):
    """Draw crosshatch grids inside diagonal bands (straight bands)."""
    bt = math.radians(band_angle_deg)
    bn_x, bn_y = -math.sin(bt), math.cos(bt)  

    cx0, cy0 = w/2.0 + bn_x*band_center_offset, h/2.0 + bn_y*band_center_offset
    d_center = bn_x*cx0 + bn_y*cy0
    half_w = band_width/2.0
    d1, d2 = d_center - half_w, d_center + half_w

    for angle_deg, color in zip(angles_deg, colors):
        theta = math.radians(angle_deg)
        ux, uy = math.cos(theta), math.sin(theta)
        n_x, n_y = -math.sin(theta), math.cos(theta)

        corners = [(0,0),(w,0),(0,h),(w,h)]
        ds = [cx*n_x + cy*n_y for (cx,cy) in corners]
        dmin, dmax = min(ds), max(ds)

        d = dmin - spacing
        while d <= dmax + spacing:
            if abs(n_y) > 1e-9:
                p0x = w/2.0
                p0y = (d - n_x*p0x) / n_y
            else:
                p0y = h/2.0
                p0x = (d - n_y*p0y) / n_x

            t0, t1 = -1e9, 1e9
            def clip(p, q):
                nonlocal t0, t1
                if abs(p) < 1e-9:
                    if q < 0: return False
                    return True
                r = q / p
                if p < 0:
                    if r > t0: t0 = r
                else:
                    if r < t1: t1 = r
                return t0 <= t1
            if not (clip(-ux,  p0x - 0) and
                    clip( ux,  w - p0x) and
                    clip(-uy,  p0y - 0) and
                    clip( uy,  h - p0y)):
                d += spacing
                continue

            denom = bn_x*ux + bn_y*uy
            if abs(denom) < 1e-9:
                d += spacing
                continue
            tA = (d1 - (bn_x*p0x + bn_y*p0y)) / denom
            tB = (d2 - (bn_x*p0x + bn_y*p0y)) / denom
            t_band_min, t_band_max = (tA, tB) if tA < tB else (tB, tA)

            t_start = max(t0, t_band_min)
            t_end   = min(t1, t_band_max)
            if t_start < t_end:
                x1, y1 = p0x + t_start*ux, p0y + t_start*uy
                x2, y2 = p0x + t_end*ux,   p0y + t_end*uy
                lw = random.randint(width_range[0], width_range[1])
                draw.line([(x1, y1), (x2, y2)], fill=(*color, alpha), width=lw)
            d += spacing

def build_image(W=10000, H=12000, seed=42, out="opart_intensified_10000x12000.png"):
    random.seed(seed)
    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Fine diagonals
    for ang in [-21, -20, -19]:
        draw_parallel_lines(draw, W, H, spacing=12, angle_deg=ang,
                            colors=[(220,40,40), (40,80,200), (0,0,0)],
                            width_range=(1,2), jitter=0.6, alpha=120)

    # Crosshatch bands
    band_angle = -20
    band_width = int(min(W, H) * 0.28)
    gap_width  = int(band_width * 0.60)
    offset0    = int(-band_width * 0.20)
    offset1    = offset0 + band_width + gap_width

    for offset in [offset0, offset1]:
        draw_crosshatch_in_band(draw, W, H, spacing=242,
                                angles_deg=[70,130],
                                colors=[(20,160,120),(220,100,40)],
                                width_range=(4,6), alpha=200,
                                band_angle_deg=band_angle,
                                band_center_offset=offset,
                                band_width=band_width)

    img.convert("RGB").save(out, "PNG", optimize=True)
    print(f"Saved {out}")

if __name__ == "__main__":
    build_image()
