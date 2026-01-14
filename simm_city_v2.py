import pygame
import sys
import random
from enum import Enum
from collections import deque

pygame.init()

# ───────────────────────────────────────────────
#  SETTINGS
# ───────────────────────────────────────────────

TILE_SIZE   = 24
GRID_W      = 60
GRID_H      = 60
SCREEN_W    = 1280
SCREEN_H    = 720 + 110
VIEW_W      = SCREEN_W // TILE_SIZE
VIEW_H      = (SCREEN_H - 110) // TILE_SIZE
FPS         = 60

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("16-bit Mini SimCity – Bayer Dither")
clock = pygame.time.Clock()

font_ui   = pygame.font.SysFont("consolas", 22, bold=True)
font_tool = pygame.font.SysFont("consolas", 16)

# ───────────────────────────────────────────────
#  16-BIT PALETTE + DITHER PAIRS
# ───────────────────────────────────────────────

C = {
    "grass_light":   (60,  120,  90),
    "grass_dark":    (36,  88,   64),
    "water_light":   (40,  110,  190),
    "water_dark":    (20,  70,   130),
    "road_light":    (100, 100,  100),
    "road_dark":     (72,  72,   72),
    "pipe_w_light":  (60,  140,  240),
    "pipe_w_dark":   (32,  100,  190),
    "pipe_s_light":  (180, 110,  60),
    "pipe_s_dark":   (140,  80,  40),
    "res_light":     (100, 220,  100),
    "res_dark":      (60,  180,  60),
    "com_light":     (80,  160,  255),
    "com_dark":      (50,  120,  210),
    "ind_light":     (240, 180,  60),
    "ind_dark":      (190, 140,  40),
    "power_light":   (220, 60,   60),
    "power_dark":    (170, 40,   40),
    "pump_light":    (60,  100,  240),
    "pump_dark":     (30,  70,   190),
    "sewage_light":  (160, 120,  80),
    "sewage_dark":   (120, 90,   50),
    "park_light":    (60,  160,  80),
    "park_dark":     (30,  120,  50),
    "ui_bg":         (24,  24,   44),
    "ui_border":     (120, 120,  160),
    "text":          (240, 240,  255),
    "selected":      (0,   255,  255),
    "unselected":    (140, 140,  180),
}

# ───────────────────────────────────────────────
#  PRECOMPUTED DITHERED TILE SURFACES
# ───────────────────────────────────────────────

PRE_DITHERED = {}

BAYER_4X4 = [
    [ 0/16,  8/16,  2/16, 10/16],
    [12/16,  4/16, 14/16,  6/16],
    [ 3/16, 11/16,  1/16,  9/16],
    [15/16,  7/16, 13/16,  5/16],
]

def bayer_dither(surface, rect, color_a, color_b):
    for y in range(rect.height):
        for x in range(rect.width):
            bx = x % 4
            by = y % 4
            level = BAYER_4X4[by][bx]
            c = color_a if level < 0.5 else color_b
            surface.set_at((rect.x + x, rect.y + y), c)

def make_dithered_tile(light, dark):
    s = pygame.Surface((TILE_SIZE, TILE_SIZE))
    s.fill(light)
    bayer_dither(s, s.get_rect(), light, dark)
    return s

# Precompute once
def precompute_tiles():
    global PRE_DITHERED
    PRE_DITHERED = {
        Tile.EMPTY:        make_dithered_tile(C["grass_light"], C["grass_dark"]),
        Tile.WATER:        make_dithered_tile(C["water_light"], C["water_dark"]),
        Tile.ROAD:         make_dithered_tile(C["road_light"],  C["road_dark"]),
        Tile.PIPE_WATER:   make_dithered_tile(C["pipe_w_light"], C["pipe_w_dark"]),
        Tile.PIPE_SEWAGE:  make_dithered_tile(C["pipe_s_light"], C["pipe_s_dark"]),
        Tile.POWER_PLANT:  make_dithered_tile(C["power_light"], C["power_dark"]),
        Tile.WATER_PUMP:   make_dithered_tile(C["pump_light"],  C["pump_dark"]),
        Tile.SEWAGE_PLANT: make_dithered_tile(C["sewage_light"], C["sewage_dark"]),
        Tile.PARK:         make_dithered_tile(C["park_light"],  C["park_dark"]),
        Tile.ZONE_R:       make_dithered_tile(C["res_light"],   C["res_dark"]),
        Tile.ZONE_C:       make_dithered_tile(C["com_light"],   C["com_dark"]),
        Tile.ZONE_I:       make_dithered_tile(C["ind_light"],   C["ind_dark"]),
        Tile.RESIDENTIAL:  make_dithered_tile(C["res_light"],   C["res_dark"]),
        Tile.COMMERCIAL:   make_dithered_tile(C["com_light"],   C["com_dark"]),
        Tile.INDUSTRIAL:   make_dithered_tile(C["ind_light"],   C["ind_dark"]),
    }

precompute_tiles()  # Run once at startup

# ───────────────────────────────────────────────
#  TILE TYPES & COSTS
# ───────────────────────────────────────────────

class Tile(Enum):
    EMPTY           = 0
    WATER           = 1
    ROAD            = 2
    ZONE_R          = 3
    ZONE_C          = 4
    ZONE_I          = 5
    RESIDENTIAL     = 6
    COMMERCIAL      = 7
    INDUSTRIAL      = 8
    POWER_PLANT     = 9
    WATER_PUMP      = 10
    SEWAGE_PLANT    = 11
    PIPE_WATER      = 12
    PIPE_SEWAGE     = 13
    PARK            = 14

COST = {
    Tile.ZONE_R:        20,
    Tile.ZONE_C:        20,
    Tile.ZONE_I:        20,
    Tile.ROAD:          50,
    Tile.POWER_PLANT:   1200,
    Tile.WATER_PUMP:    1600,
    Tile.SEWAGE_PLANT:  2200,
    Tile.PIPE_WATER:    35,
    Tile.PIPE_SEWAGE:   40,
    Tile.PARK:          120,
}

TOOL_NAME = {
    Tile.ZONE_R:        "Residential Zone",
    Tile.ZONE_C:        "Commercial Zone",
    Tile.ZONE_I:        "Industrial Zone",
    Tile.ROAD:          "Road",
    Tile.POWER_PLANT:   "Power Plant",
    Tile.WATER_PUMP:    "Water Pump",
    Tile.SEWAGE_PLANT:  "Sewage Plant",
    Tile.PIPE_WATER:    "Water Pipe",
    Tile.PIPE_SEWAGE:   "Sewage Pipe",
    Tile.PARK:          "Park",
}

TOOL_SHORT = {
    Tile.ZONE_R:        "RZn",
    Tile.ZONE_C:        "CZn",
    Tile.ZONE_I:        "IZn",
    Tile.ROAD:          "Rd",
    Tile.POWER_PLANT:   "Pwr",
    Tile.WATER_PUMP:    "Wtr",
    Tile.SEWAGE_PLANT:  "Swg",
    Tile.PIPE_WATER:    "WP",
    Tile.PIPE_SEWAGE:   "SP",
    Tile.PARK:          "Pk",
}

TOOL_KEY = {
    pygame.K_1: Tile.ZONE_R,
    pygame.K_2: Tile.ZONE_C,
    pygame.K_3: Tile.ZONE_I,
    pygame.K_4: Tile.ROAD,
    pygame.K_5: Tile.POWER_PLANT,
    pygame.K_6: Tile.WATER_PUMP,
    pygame.K_7: Tile.SEWAGE_PLANT,
    pygame.K_8: Tile.PIPE_WATER,
    pygame.K_9: Tile.PIPE_SEWAGE,
    pygame.K_0: Tile.PARK,
}

TOOLS_ORDER = list(TOOL_NAME.keys())

# ───────────────────────────────────────────────
#  CITY CLASS
# ───────────────────────────────────────────────

class City:
    def __init__(self):
        self.grid = [[Tile.EMPTY for _ in range(GRID_W)] for _ in range(GRID_H)]
        self.power   = [[False]*GRID_W for _ in range(GRID_H)]
        self.water   = [[False]*GRID_W for _ in range(GRID_H)]
        self.sewage  = [[False]*GRID_W for _ in range(GRID_H)]
        self.air_p   = [[0.0]*GRID_W for _ in range(GRID_H)]
        self.water_p = [[0.0]*GRID_W for _ in range(GRID_H)]
        self.traffic = [[0.0]*GRID_W for _ in range(GRID_H)]
        self.cars = []

        # Lakes
        for _ in range(6):
            cx = random.randint(8, GRID_W-9)
            cy = random.randint(8, GRID_H-9)
            r = random.randint(5, 11)
            for y in range(max(0, cy-r), min(GRID_H, cy+r)):
                for x in range(max(0, cx-r), min(GRID_W, cx+r)):
                    if (x-cx)**2 + (y-cy)**2 < r**2 * random.uniform(0.6, 1.1):
                        self.grid[y][x] = Tile.WATER

    def count(self, t: Tile):
        return sum(row.count(t) for row in self.grid)

    def update_networks(self):
        self.power = [[False]*GRID_W for _ in range(GRID_H)]
        self.water = [[False]*GRID_W for _ in range(GRID_H)]
        self.sewage = [[False]*GRID_W for _ in range(GRID_H)]

        dirs = [(-1,0),(1,0),(0,-1),(0,1)]

        # Power
        q = deque()
        vis = [[False]*GRID_W for _ in range(GRID_H)]
        for y in range(GRID_H):
            for x in range(GRID_W):
                if self.grid[y][x] == Tile.POWER_PLANT:
                    q.append((x,y))
                    vis[y][x] = True
                    self.power[y][x] = True

        while q:
            x,y = q.popleft()
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0<=nx<GRID_W and 0<=ny<GRID_H and not vis[ny][nx]:
                    if self.grid[ny][nx] in (Tile.ROAD, Tile.POWER_PLANT):
                        vis[ny][nx] = True
                        q.append((nx,ny))
                        self.power[ny][nx] = True

        # Water
        q = deque()
        vis = [[False]*GRID_W for _ in range(GRID_H)]
        for y in range(GRID_H):
            for x in range(GRID_W):
                if self.grid[y][x] == Tile.WATER_PUMP:
                    q.append((x,y))
                    vis[y][x] = True
                    self.water[y][x] = True

        while q:
            x,y = q.popleft()
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0<=nx<GRID_W and 0<=ny<GRID_H and not vis[ny][nx]:
                    if self.grid[ny][nx] in (Tile.PIPE_WATER, Tile.WATER_PUMP):
                        vis[ny][nx] = True
                        q.append((nx,ny))
                        self.water[ny][nx] = True

        # Sewage
        q = deque()
        vis = [[False]*GRID_W for _ in range(GRID_H)]
        for y in range(GRID_H):
            for x in range(GRID_W):
                if self.grid[y][x] == Tile.SEWAGE_PLANT:
                    q.append((x,y))
                    vis[y][x] = True
                    self.sewage[y][x] = True

        while q:
            x,y = q.popleft()
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0<=nx<GRID_W and 0<=ny<GRID_H and not vis[ny][nx]:
                    if self.grid[ny][nx] in (Tile.PIPE_SEWAGE, Tile.SEWAGE_PLANT,
                                             Tile.ROAD, Tile.ZONE_R, Tile.ZONE_C, Tile.ZONE_I,
                                             Tile.RESIDENTIAL, Tile.COMMERCIAL, Tile.INDUSTRIAL):
                        vis[ny][nx] = True
                        q.append((nx,ny))
                        self.sewage[ny][nx] = True

    def can_develop(self, x, y):
        t = self.grid[y][x]
        if t not in (Tile.ZONE_R, Tile.ZONE_C, Tile.ZONE_I):
            return False
        return (self.power[y][x] and
                self.water[y][x] and
                self.sewage[y][x] and
                any(self.grid[y+dy][x+dx] == Tile.ROAD
                    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]
                    if 0<=x+dx<GRID_W and 0<=y+dy<GRID_H))

# ───────────────────────────────────────────────
#  DRAW TILE – using precomputed dither
# ───────────────────────────────────────────────

def draw_tile(gx, gy, cam_x, cam_y, city):
    sx = (gx - cam_x) * TILE_SIZE
    sy = (gy - cam_y) * TILE_SIZE
    t = city.grid[gy][gx]

    if t in PRE_DITHERED:
        screen.blit(PRE_DITHERED[t], (sx, sy))
    else:
        pygame.draw.rect(screen, (80,80,80), (sx, sy, TILE_SIZE, TILE_SIZE))

    # Overlays / details
    if t == Tile.WATER:
        for i in range(0, TILE_SIZE, 8):
            pygame.draw.line(screen, C["water_ripple"], (sx+i, sy), (sx+i+4, sy+TILE_SIZE), 2)

    elif t == Tile.ROAD:
        pygame.draw.rect(screen, (255,240,100), (sx+10, sy, 4, TILE_SIZE))
        pygame.draw.rect(screen, (255,240,100), (sx, sy+10, TILE_SIZE, 4))

    elif t == Tile.PIPE_WATER:
        pygame.draw.line(screen, C["pipe_w_light"], (sx+2, sy+12), (sx+TILE_SIZE-2, sy+12), 6)

    elif t == Tile.PIPE_SEWAGE:
        pygame.draw.line(screen, C["pipe_s_light"], (sx+3, sy+11), (sx+TILE_SIZE-3, sy+11), 7)

    elif t == Tile.POWER_PLANT:
        pygame.draw.rect(screen, (255,220,100), (sx+8, sy+8, TILE_SIZE-16, 6))

    elif t == Tile.WATER_PUMP:
        pygame.draw.circle(screen, (140,200,255), (sx + TILE_SIZE//2, sy + TILE_SIZE//2), 9)

    elif t == Tile.PARK:
        pygame.draw.circle(screen, (40,180,80), (sx+12, sy+10), 9)

    # Pollution
    ap = city.air_p[gy][gx]
    if ap > 0.4:
        alpha = min(170, int(ap * 100))
        col = (255,140,60,alpha) if ap > 1.8 else (255,255,100,alpha) if ap > 0.9 else (140,255,140,alpha)
        s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        s.fill(col)
        screen.blit(s, (sx, sy))

    wp = city.water_p[gy][gx]
    if wp > 0.4:
        alpha = min(170, int(wp * 100))
        col = (120,80,40,alpha) if wp > 1.8 else (180,140,60,alpha) if wp > 0.9 else (200,180,120,alpha)
        s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        s.fill(col)
        screen.blit(s, (sx, sy))

    pygame.draw.rect(screen, (0,0,0), (sx, sy, TILE_SIZE, TILE_SIZE), 2)

# ───────────────────────────────────────────────
#  MAIN LOOP
# ───────────────────────────────────────────────

city = City()
cam_x = cam_y = GRID_W // 2 - VIEW_W // 2
money = 60000
population = 0
month_timer = 0
MONTH_MS = 3500
current_tool = Tile.ZONE_R

running = True
while running:
    dt = clock.tick(FPS)
    month_timer += dt

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            mx, my = pygame.mouse.get_pos()
            if my >= SCREEN_H - 110:
                idx = (mx - 20) // 115
                if 0 <= idx < len(TOOLS_ORDER):
                    current_tool = TOOLS_ORDER[idx]
            else:
                gx = cam_x + mx // TILE_SIZE
                gy = cam_y + my // TILE_SIZE
                if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                    cost = COST.get(current_tool, 0)
                    if money >= cost and city.grid[gy][gx] == Tile.EMPTY:
                        city.grid[gy][gx] = current_tool
                        money -= cost

        elif e.type == pygame.KEYDOWN:
            if e.key in TOOL_KEY:
                current_tool = TOOL_KEY[e.key]

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:   cam_x = max(0, cam_x - 1)
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:  cam_x = min(GRID_W - VIEW_W, cam_x + 1)
    if keys[pygame.K_w] or keys[pygame.K_UP]:     cam_y = max(0, cam_y - 1)
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:   cam_y = min(GRID_H - VIEW_H, cam_y + 1)

    if month_timer >= MONTH_MS:
        month_timer -= MONTH_MS
        city.update_networks()

        # Growth
        for y in range(GRID_H):
            for x in range(GRID_W):
                if city.can_develop(x, y):
                    t = city.grid[y][x]
                    chance = 0.035
                    if t == Tile.ZONE_R: chance *= 1.35
                    if t == Tile.ZONE_I: chance *= 0.85
                    if random.random() < chance:
                        new_tile = {Tile.ZONE_R: Tile.RESIDENTIAL,
                                    Tile.ZONE_C: Tile.COMMERCIAL,
                                    Tile.ZONE_I: Tile.INDUSTRIAL}[t]
                        city.grid[y][x] = new_tile
                        population += random.randint(8, 35) if new_tile == Tile.RESIDENTIAL else random.randint(4, 18)

        # Pollution
        for y in range(GRID_H):
            for x in range(GRID_W):
                t = city.grid[y][x]
                if t == Tile.INDUSTRIAL:      city.air_p[y][x] += 0.38
                if t == Tile.POWER_PLANT:     city.air_p[y][x] += 0.28
                if t == Tile.RESIDENTIAL and not city.sewage[y][x]:
                    city.water_p[y][x] += 0.22
                if t == Tile.COMMERCIAL and not city.sewage[y][x]:
                    city.water_p[y][x] += 0.14

        for y in range(GRID_H):
            for x in range(GRID_W):
                city.air_p[y][x] *= 0.94
                city.water_p[y][x] *= 0.92

        money += population // 12

    screen.fill((12, 25, 50))

    for gy in range(cam_y, min(cam_y + VIEW_H + 2, GRID_H)):
        for gx in range(cam_x, min(cam_x + VIEW_W + 2, GRID_W)):
            draw_tile(gx, gy, cam_x, cam_y, city)

    # Cars
    if random.random() < 0.09 and len(city.cars) < 90:
        roads = [(x,y) for y in range(GRID_H) for x in range(GRID_W) if city.grid[y][x] == Tile.ROAD]
        if roads:
            rx, ry = random.choice(roads)
            city.cars.append([rx + 0.5, ry + 0.5, random.choice([(1,0),(-1,0),(0,1),(0,-1)])])

    for car in city.cars[:]:
        dx, dy = car[2]
        nx = int(car[0] + dx * 0.14)
        ny = int(car[1] + dy * 0.14)
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H and city.grid[ny][nx] == Tile.ROAD:
            car[0] += dx * 0.14
            car[1] += dy * 0.14
        else:
            car[2] = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        sx = int((car[0] - cam_x) * TILE_SIZE)
        sy = int((car[1] - cam_y) * TILE_SIZE)
        pygame.draw.circle(screen, (255,220,80), (sx + TILE_SIZE//2, sy + TILE_SIZE//2), 5)

    # UI
    pygame.draw.rect(screen, C["ui_bg"], (0, SCREEN_H-110, SCREEN_W, 110))
    pygame.draw.rect(screen, C["ui_border"], (0, SCREEN_H-110, SCREEN_W, 110), 3)

    status = f"Money ${money:,}     Population {population:,}     Tool: {TOOL_NAME.get(current_tool, '—')}"
    screen.blit(font_ui.render(status, True, C["text"]), (24, SCREEN_H - 85))

    btn_w = 100
    start_x = 24
    gap = 8
    for i, t in enumerate(TOOLS_ORDER):
        x = start_x + i * (btn_w + gap)
        col = C["selected"] if t == current_tool else C["unselected"]
        pygame.draw.rect(screen, col, (x, SCREEN_H-75, btn_w, 50), 4, border_radius=6)
        label = font_tool.render(TOOL_SHORT[t], True, C["text"])
        tx = x + (btn_w - label.get_width()) // 2
        ty = SCREEN_H - 75 + (50 - label.get_height()) // 2 + 2
        screen.blit(label, (tx, ty))

    pygame.display.flip()

pygame.quit()
sys.exit()
