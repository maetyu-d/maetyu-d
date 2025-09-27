#!/usr/bin/env python3
"""
Op Art Intensified Generator (Full-size, single-pass)

Changes requested:
- Fine diagonal spacing: 36 px (was 12)
- Fine diagonal stroke: 2–5 px (was 1–2)
- Crosshatch spacing: 300 px (was 242)
- Crosshatch stroke: 6–8 px (was 4–6)
"""

import math, random
from PIL import Image, ImageDraw

# ---------- Core geometry utilities ----------

def draw_parallel_lines(draw, w, h, spacing, angle_deg, colors,
                        width_range=(2,5), jitter=0.6, alpha=150):
    """
    Draw evenly spaced, infinite parallel lines at angle_deg, clipped to the WxH canvas.
    Efficient math (no rotations). Slight jitter along the line normal to induce vibrato.
    """
    theta = math.radians(angle_deg)
    ux, uy = math.cos(theta), math.sin(theta)           # line direction
    nx, ny = -math.sin(theta), math.cos(theta)          # normal to line

    # Project canvas corners on the normal to find min/max distance (coverage range)
    corners = [(0,0), (w,0), (0,h), (w,h)]
    distances = [x*nx + y*ny for (x,y) in corners]
    dmin, dmax = min(distances), max(distances)

    d = dmin - spacing
    idx = 0
    while d <= dmax + spacing:
        col = (*colors[idx % len(colors)], alpha)
        idx += 1

        # tiny normal jitter for optical shimmer
        jd = (random.uniform(-jitter, jitter) if jitter else 0.0)

        # point p0 on this line: n·p0 = d + jd, pick around center for stability
        cx, cy = w/2.0, h/2.0
        d_center = nx*cx + ny*cy
        shift = (d + jd) - d_center
        p0x, p0y = cx + nx*shift, cy + ny*shift

        # Liang–Barsky clip of infinite line p(t) = p0 + t*u to [0,w]×[0,h]
        t0, t1 = -1e12, 1e12
        def clip(p, q):
            nonlocal t0, t1
            if abs(p) < 1e-9:
                return q >= 0  # parallel: keep if inside
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

        x1, y1 = p0x + t0*ux, p0y + t0*uy
        x2, y2 = p0x + t1*ux, p0y + t1*uy
        lw = random.randint(width_range[0], width_range[1])
        draw.line([(x1, y1), (x2, y2)], fill=col, width=lw)

        d += spacing

def draw_crosshatch_in_band(draw, w, h, spacing, angles_deg, colors,
                            width_range=(6,8), alpha=210,
                            band_angle_deg=-20, band_center_offset=0, band_width=1000):
    """
    Draw line families at angles_deg, but only within a diagonal band defined by band_angle_deg,
    band_center_offset (signed distance along band normal), and band_width.
    """
    bt = math.radians(band_angle_deg)
    bn_x, bn_y = -math.sin(bt), math.cos(bt)  # band normal

    # Band center in canvas coordinates
    cx0, cy0 = w/2.0 + bn_x*band_center_offset, h/2.0 + bn_y*band_center_offset
    d_center = bn_x*cx0 + bn_y*cy0
    half_w = band_width / 2.0
    d1, d2 = d_center - half_w, d_center + half_w

    for angle_deg, color in zip(angles_deg, colors):
        theta = math.radians(angle_deg)
        ux, uy = math.cos(theta), math.sin(theta)
        n_x, n_y = -math.sin(theta), math.cos(theta)

        # distance range across canvas for this line orientation
        corners = [(0,0), (w,0), (0,h), (w,h)]
        ds = [x*n_x + y*n_y for (x,y) in corners]
        dmin, dmax = min(ds), max(ds)

        d = dmin - spacing
        while d <= dmax + spacing:
            # point p0 with n·p0 = d
            if abs(n_y) > 1e-9:
                p0x = w/2.0
                p0y = (d - n_x*p0x) / n_y
            else:
                p0y = h/2.0
                p0x = (d - n_y*p0y) / n_x

            # clip to canvas
            t0, t1 = -1e12, 1e12
            def clip(p, q):
                nonlocal t0, t1
                if abs(p) < 1e-9:
                    return q >= 0
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

            # intersect with band edges (two parallels at distances d1, d2 along band normal)
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

# ---------- Main builder ----------

def build_image(W=10000, H=12000, seed=42, out="opart_intensified_10000x12000.png"):
    random.seed(seed)
    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # --- Fine diagonals (triple), UPDATED spacing & stroke ---
    fine_angles = [-21, -20, -19]
    fine_spacing = 36               # (was 12)
    fine_colors  = [(220,40,40), (40,80,200), (0,0,0)]
    for ang in fine_angles:
        draw_parallel_lines(draw, W, H,
                            spacing=fine_spacing,
                            angle_deg=ang,
                            colors=fine_colors,
                            width_range=(2,5),   # (was 1–2)
                            jitter=0.6,
                            alpha=150)

    # --- Crosshatch bands (complementary), UPDATED spacing & stroke ---
    grid_spacing = 300              # (was 242)
    grid_angles  = [70, 130]
    grid_colors  = [(20,160,120), (220,100,40)]  # green/cyan + red/orange
    band_angle   = -20
    band_width   = int(min(W, H) * 0.28)
    gap_width    = int(band_width * 0.60)
    offset0      = int(-band_width * 0.20)
    offset1      = offset0 + band_width + gap_width

    draw_crosshatch_in_band(draw, W, H,
                            spacing=grid_spacing,
                            angles_deg=grid_angles,
                            colors=grid_colors,
                            width_range=(6,8),   # (was 4–6)
                            alpha=210,
                            band_angle_deg=band_angle,
                            band_center_offset=offset0,
                            band_width=band_width)

    draw_crosshatch_in_band(draw, W, H,
                            spacing=grid_spacing,
                            angles_deg=grid_angles,
                            colors=grid_colors,
                            width_range=(6,8),
                            alpha=210,
                            band_angle_deg=band_angle,
                            band_center_offset=offset1,
                            band_width=band_width)

    img.convert("RGB").save(out, "PNG", optimize=True)
    print(f"Saved {out}")

# ---------- Entry point ----------

if __name__ == "__main__":
    build_image()

