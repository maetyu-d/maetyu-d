import pygame
import sys
import random
from enum import Enum
from collections import deque

# ───────────────────────────────────────────────
#  SETTINGS
# ───────────────────────────────────────────────

TILE_SIZE = 24
GRID_W = 60
GRID_H = 60
SCREEN_W = 1280
SCREEN_H = 720 + 80
VIEW_W = SCREEN_W // TILE_SIZE
VIEW_H = (SCREEN_H - 80) // TILE_SIZE
FPS = 60

pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Mini SimCity – Power / Water / Sewage")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20, bold=True)
font_small = pygame.font.SysFont("consolas", 14)

# Colors
C = {
    "grass": (34, 139, 34),
    "grass_dark": (25, 100, 25),
    "water": (0, 70, 150),
    "water_ripple": (100, 180, 255),
    "road": (70, 70, 70),
    "road_line": (255, 255, 100),
    "pipe_water": (40, 120, 220),
    "pipe_sewage": (160, 90, 40),
    "residential": (60, 220, 60),
    "commercial": (60, 160, 255),
    "industrial": (220, 180, 40),
    "power_plant": (220, 40, 40),
    "water_pump": (40, 80, 220),
    "sewage_plant": (140, 100, 60),
    "park": (20, 140, 20),
    "pollution_air_low": (140, 255, 140, 100),
    "pollution_air_med": (255, 255, 100, 140),
    "pollution_air_high": (255, 140, 60, 180),
    "pollution_water_low": (200, 180, 120, 90),
    "pollution_water_med": (180, 140, 60, 140),
    "pollution_water_high": (120, 80, 40, 180),
    "ui_bg": (30, 30, 50),
    "text": (240, 240, 255),
    "selected": (0, 255, 255),
    "unselected": (140, 140, 160),
}

# ───────────────────────────────────────────────
#  TILE TYPES
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
    Tile.ZONE_R:        "R Zone",
    Tile.ZONE_C:        "C Zone",
    Tile.ZONE_I:        "I Zone",
    Tile.ROAD:          "Road",
    Tile.POWER_PLANT:   "Power Plant",
    Tile.WATER_PUMP:    "Water Pump",
    Tile.SEWAGE_PLANT:  "Sewage Plant",
    Tile.PIPE_WATER:    "Water Pipe",
    Tile.PIPE_SEWAGE:   "Sewage Pipe",
    Tile.PARK:          "Park",
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

# ───────────────────────────────────────────────
#  CITY GRID & STATE
# ───────────────────────────────────────────────

class City:
    def __init__(self):
        self.grid = [[Tile.EMPTY for _ in range(GRID_W)] for _ in range(GRID_H)]

        # Utility networks
        self.power = [[False]*GRID_W for _ in range(GRID_H)]
        self.water = [[False]*GRID_W for _ in range(GRID_H)]
        self.sewage = [[False]*GRID_W for _ in range(GRID_H)]

        # Pollution (0–3 range roughly)
        self.air_pollution = [[0.0]*GRID_W for _ in range(GRID_H)]
        self.water_pollution = [[0.0]*GRID_W for _ in range(GRID_H)]

        # Traffic density
        self.traffic = [[0.0]*GRID_W for _ in range(GRID_H)]

        # Cars for visual effect
        self.cars = []

        # Generate some lakes
        for _ in range(6):
            cx, cy = random.randint(8, GRID_W-9), random.randint(8, GRID_H-9)
            r = random.randint(5, 11)
            for y in range(max(0, cy-r), min(GRID_H, cy+r)):
                for x in range(max(0, cx-r), min(GRID_W, cx+r)):
                    if (x-cx)**2 + (y-cy)**2 < r**2 * random.uniform(0.6, 1.1):
                        self.grid[y][x] = Tile.WATER

    def count(self, t: Tile):
        return sum(row.count(t) for row in self.grid)

    def update_networks(self):
        # Reset
        self.power = [[False]*GRID_W for _ in range(GRID_H)]
        self.water = [[False]*GRID_W for _ in range(GRID_H)]
        self.sewage = [[False]*GRID_W for _ in range(GRID_H)]

        # Power (conducts through roads + power plants)
        visited = [[False]*GRID_W for _ in range(GRID_H)]
        q = deque()
        for y in range(GRID_H):
            for x in range(GRID_W):
                if self.grid[y][x] == Tile.POWER_PLANT:
                    q.append((x,y))
                    visited[y][x] = True
                    self.power[y][x] = True

        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            x,y = q.popleft()
            self.power[y][x] = True
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0<=nx<GRID_W and 0<=ny<GRID_H and not visited[ny][nx]:
                    if self.grid[ny][nx] in (Tile.ROAD, Tile.POWER_PLANT):
                        visited[ny][nx] = True
                        q.append((nx,ny))

        # Water (only through water pipes + pumps)
        visited = [[False]*GRID_W for _ in range(GRID_H)]
        q = deque()
        for y in range(GRID_H):
            for x in range(GRID_W):
                if self.grid[y][x] == Tile.WATER_PUMP:
                    q.append((x,y))
                    visited[y][x] = True
                    self.water[y][x] = True

        while q:
            x,y = q.popleft()
            self.water[y][x] = True
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0<=nx<GRID_W and 0<=ny<GRID_H and not visited[ny][nx]:
                    if self.grid[ny][nx] in (Tile.PIPE_WATER, Tile.WATER_PUMP):
                        visited[ny][nx] = True
                        q.append((nx,ny))

        # Sewage (reverse direction – from buildings to treatment plants)
        visited = [[False]*GRID_W for _ in range(GRID_H)]
        q = deque()
        for y in range(GRID_H):
            for x in range(GRID_W):
                if self.grid[y][x] == Tile.SEWAGE_PLANT:
                    q.append((x,y))
                    visited[y][x] = True
                    self.sewage[y][x] = True

        while q:
            x,y = q.popleft()
            self.sewage[y][x] = True
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0<=nx<GRID_W and 0<=ny<GRID_H and not visited[ny][nx]:
                    if self.grid[ny][nx] in (Tile.PIPE_SEWAGE, Tile.SEWAGE_PLANT,
                                             Tile.ROAD, Tile.ZONE_R, Tile.ZONE_C, Tile.ZONE_I,
                                             Tile.RESIDENTIAL, Tile.COMMERCIAL, Tile.INDUSTRIAL):
                        visited[ny][nx] = True
                        q.append((nx,ny))

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
#  RENDERING HELPERS
# ───────────────────────────────────────────────

def draw_tile(x, y, cam_x, cam_y, city):
    sx = (x - cam_x) * TILE_SIZE
    sy = (y - cam_y) * TILE_SIZE
    rect = pygame.Rect(sx, sy, TILE_SIZE, TILE_SIZE)
    t = city.grid[y][x]

    if t == Tile.EMPTY:
        pygame.draw.rect(screen, C["grass"], rect)
        for i in range(0, TILE_SIZE, 5):
            pygame.draw.line(screen, C["grass_dark"], (sx+i, sy), (sx+i, sy+TILE_SIZE), 1)

    elif t == Tile.WATER:
        pygame.draw.rect(screen, C["water"], rect)
        for i in range(0, TILE_SIZE, 8):
            pygame.draw.line(screen, C["water_ripple"], (sx+i, sy), (sx+i+4, sy+TILE_SIZE), 2)

    elif t == Tile.ROAD:
        pygame.draw.rect(screen, C["road"], rect)
        pygame.draw.rect(screen, C["road_line"], (sx+TILE_SIZE//2-1, sy, 2, TILE_SIZE))
        pygame.draw.rect(screen, C["road_line"], (sx, sy+TILE_SIZE//2-1, TILE_SIZE, 2))

    elif t == Tile.PIPE_WATER:
        pygame.draw.rect(screen, (40,40,40), rect)
        pygame.draw.line(screen, C["pipe_water"], (sx, sy+TILE_SIZE//2), (sx+TILE_SIZE, sy+TILE_SIZE//2), 5)
        pygame.draw.line(screen, C["pipe_water"], (sx+TILE_SIZE//2, sy), (sx+TILE_SIZE//2, sy+TILE_SIZE), 5)

    elif t == Tile.PIPE_SEWAGE:
        pygame.draw.rect(screen, (50,50,50), rect)
        pygame.draw.line(screen, C["pipe_sewage"], (sx+3, sy+TILE_SIZE//2), (sx+TILE_SIZE-3, sy+TILE_SIZE//2), 6)
        pygame.draw.line(screen, C["pipe_sewage"], (sx+TILE_SIZE//2, sy+3), (sx+TILE_SIZE//2, sy+TILE_SIZE-3), 6)

    elif t == Tile.POWER_PLANT:
        pygame.draw.rect(screen, C["power_plant"], rect)
        pygame.draw.rect(screen, (180,60,60), (sx+4, sy+4, TILE_SIZE-8, TILE_SIZE-8))

    elif t == Tile.WATER_PUMP:
        pygame.draw.rect(screen, C["water_pump"], rect)
        pygame.draw.circle(screen, (100,180,255), rect.center, TILE_SIZE//3)

    elif t == Tile.SEWAGE_PLANT:
        pygame.draw.rect(screen, C["sewage_plant"], rect)
        pygame.draw.rect(screen, (120,80,50), (sx+6, sy+6, TILE_SIZE-12, TILE_SIZE-12))

    elif t == Tile.PARK:
        pygame.draw.rect(screen, C["park"], rect)
        pygame.draw.circle(screen, (0,180,0), (sx+TILE_SIZE//2, sy+TILE_SIZE//2-4), 9)

    elif t in (Tile.ZONE_R, Tile.ZONE_C, Tile.ZONE_I):
        base = (100,220,100) if t==Tile.ZONE_R else (80,160,255) if t==Tile.ZONE_C else (220,200,80)
        pygame.draw.rect(screen, base, rect)

    elif t in (Tile.RESIDENTIAL, Tile.COMMERCIAL, Tile.INDUSTRIAL):
        base = C["residential"] if t==Tile.RESIDENTIAL else C["commercial"] if t==Tile.COMMERCIAL else C["industrial"]
        pygame.draw.rect(screen, base, rect)

    # Pollution overlays
    ap = city.air_pollution[y][x]
    if ap > 0.4:
        col = C["pollution_air_high"] if ap > 1.8 else C["pollution_air_med"] if ap > 0.9 else C["pollution_air_low"]
        s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        s.fill(col)
        screen.blit(s, (sx, sy))

    wp = city.water_pollution[y][x]
    if wp > 0.4:
        col = C["pollution_water_high"] if wp > 1.8 else C["pollution_water_med"] if wp > 0.9 else C["pollution_water_low"]
        s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        s.fill(col)
        screen.blit(s, (sx, sy))

    # Border
    pygame.draw.rect(screen, (10,10,10), rect, 1)

# ───────────────────────────────────────────────
#  MAIN LOOP
# ───────────────────────────────────────────────

city = City()
cam_x = cam_y = 15
money = 50000
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
            if my >= SCREEN_H - 80:
                # UI click
                idx = (mx - 20) // 100
                if 0 <= idx < len(TOOL_NAME):
                    keys = list(TOOL_NAME.keys())
                    current_tool = keys[idx]
            else:
                gx = cam_x + mx // TILE_SIZE
                gy = cam_y + my // TILE_SIZE
                if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                    t = city.grid[gy][gx]
                    if t == Tile.EMPTY or current_tool in (Tile.PIPE_WATER, Tile.PIPE_SEWAGE, Tile.ROAD):
                        cost = COST.get(current_tool, 0)
                        if money >= cost:
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

    # Monthly update
    if month_timer >= MONTH_MS:
        month_timer = 0

        city.update_networks()

        # Growth
        for y in range(GRID_H):
            for x in range(GRID_W):
                t = city.grid[y][x]
                if t in (Tile.ZONE_R, Tile.ZONE_C, Tile.ZONE_I) and city.can_develop(x, y):
                    chance = 0.04
                    if t == Tile.ZONE_R: chance *= 1.3
                    if t == Tile.ZONE_I: chance *= 0.9
                    if random.random() < chance:
                        new_t = {Tile.ZONE_R: Tile.RESIDENTIAL,
                                 Tile.ZONE_C: Tile.COMMERCIAL,
                                 Tile.ZONE_I: Tile.INDUSTRIAL}[t]
                        city.grid[y][x] = new_t
                        population += random.randint(8, 35) if new_t == Tile.RESIDENTIAL else random.randint(4, 18)

        # Pollution generation
        for y in range(GRID_H):
            for x in range(GRID_W):
                t = city.grid[y][x]
                if t == Tile.INDUSTRIAL:
                    city.air_pollution[y][x] += 0.35
                if t == Tile.POWER_PLANT:
                    city.air_pollution[y][x] += 0.25
                if t == Tile.RESIDENTIAL and not city.sewage[y][x]:
                    city.water_pollution[y][x] += 0.20
                if t == Tile.COMMERCIAL and not city.sewage[y][x]:
                    city.water_pollution[y][x] += 0.12

        # Decay pollution
        for y in range(GRID_H):
            for x in range(GRID_W):
                city.air_pollution[y][x] *= 0.94
                city.water_pollution[y][x] *= 0.92

        # Simple tax income
        money += population // 12

    # Draw
    screen.fill((12, 25, 50))

    for gy in range(cam_y, min(cam_y + VIEW_H + 1, GRID_H)):
        for gx in range(cam_x, min(cam_x + VIEW_W + 1, GRID_W)):
            draw_tile(gx, gy, cam_x, cam_y, city)

    # Cars (very simple)
    if random.random() < 0.08 and len(city.cars) < 80:
        roads = [(x,y) for y in range(GRID_H) for x in range(GRID_W) if city.grid[y][x] == Tile.ROAD]
        if roads:
            rx, ry = random.choice(roads)
            city.cars.append([rx + 0.5, ry + 0.5, random.choice([(1,0),(-1,0),(0,1),(0,-1)])])

    for car in city.cars[:]:
        dx, dy = car[2]
        nx = int(car[0] + dx * 0.12)
        ny = int(car[1] + dy * 0.12)
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H and city.grid[ny][nx] == Tile.ROAD:
            car[0] += dx * 0.12
            car[1] += dy * 0.12
        else:
            car[2] = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        # Draw
        sx = int((car[0] - cam_x) * TILE_SIZE)
        sy = int((car[1] - cam_y) * TILE_SIZE)
        pygame.draw.circle(screen, (255,220,80), (sx + TILE_SIZE//2, sy + TILE_SIZE//2), 5)

    # UI
    pygame.draw.rect(screen, C["ui_bg"], (0, SCREEN_H-80, SCREEN_W, 80))

    status = f"Money ${money:,}   Pop {population:,}   Tool: {TOOL_NAME.get(current_tool, '—')}"
    screen.blit(font.render(status, True, C["text"]), (20, SCREEN_H-55))

    # Tool buttons
    tools_list = list(TOOL_NAME.keys())
    for i, t in enumerate(tools_list):
        x = 20 + i * 110
        col = C["selected"] if t == current_tool else C["unselected"]
        pygame.draw.rect(screen, col, (x, SCREEN_H-65, 100, 50), 3, border_radius=6)
        txt = font_small.render(TOOL_NAME[t], True, C["text"])
        screen.blit(txt, (x + 8, SCREEN_H-48))

    pygame.display.flip()

pygame.quit()
sys.exit()
