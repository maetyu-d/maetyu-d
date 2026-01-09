import pygame
import sys
import random
from enum import Enum
from typing import List

pygame.init()

# ==================== SETTINGS ====================
TILE_SIZE = 24
GRID_WIDTH = 60
GRID_HEIGHT = 60
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720 + 80
VIEW_WIDTH = SCREEN_WIDTH // TILE_SIZE
VIEW_HEIGHT = (SCREEN_HEIGHT - 80) // TILE_SIZE
FPS = 60

# ==================== COLORS ====================
COLORS = {
    'grass': (34, 139, 34),
    'grass_dark': (25, 100, 25),
    'residential': (50, 205, 50),
    'res_roof': (255, 69, 0),
    'commercial': (30, 144, 255),
    'com_roof': (100, 100, 255),
    'industrial': (255, 215, 0),
    'ind_roof': (169, 169, 169),
    'road_base': (70, 70, 70),
    'road_line': (255, 255, 0),
    'power_plant': (220, 20, 60),
    'pp_smoke': (105, 105, 105),
    'park': (0, 128, 0),
    'park_tree': (0, 100, 0),
    'ui_bg': (40, 40, 60),
    'text': (255, 255, 255),
    'selected_tool': (0, 255, 255),
    'unselected_tool': (150, 150, 150),
    # Traffic overlays
    'traffic_low': (0, 200, 0),
    'traffic_med': (255, 255, 0),
    'traffic_high': (255, 100, 0),
    'traffic_jam': (200, 0, 0),
    # Pollution haze
    'pollution_low': (100, 255, 100),
    'pollution_med': (255, 255, 100),
    'pollution_high': (255, 150, 50),
    'pollution_severe': (150, 50, 50),
}

# ==================== TILE TYPES ====================
class TileType(Enum):
    EMPTY = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    ROAD = 4
    POWER_PLANT = 5
    ZONE_RESIDENTIAL = 6
    ZONE_COMMERCIAL = 7
    ZONE_INDUSTRIAL = 8
    PARK = 9

COSTS = {
    TileType.ZONE_RESIDENTIAL: 20,
    TileType.ZONE_COMMERCIAL: 20,
    TileType.ZONE_INDUSTRIAL: 20,
    TileType.ROAD: 50,
    TileType.POWER_PLANT: 1000,
    TileType.PARK: 100,
}

TOOL_NAMES = {
    TileType.ZONE_RESIDENTIAL: "R-Zone",
    TileType.ZONE_COMMERCIAL: "C-Zone",
    TileType.ZONE_INDUSTRIAL: "I-Zone",
    TileType.ROAD: "Road",
    TileType.POWER_PLANT: "Power",
    TileType.PARK: "Park",
}

TOOL_LABELS = {
    TileType.ZONE_RESIDENTIAL: "RZ",
    TileType.ZONE_COMMERCIAL: "CZ",
    TileType.ZONE_INDUSTRIAL: "IZ",
    TileType.ROAD: "RD",
    TileType.POWER_PLANT: "PP",
    TileType.PARK: "PK",
}

# ==================== CITY GRID ====================
class CityGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[TileType.EMPTY for _ in range(width)] for _ in range(height)]
        self.traffic = [[0.0 for _ in range(width)] for _ in range(height)]
        self.pollution = [[0.0 for _ in range(width)] for _ in range(height)]

    def get_tile(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def set_tile(self, x: int, y: int, tile_type: TileType):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = tile_type

    def get_traffic(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.traffic[y][x]
        return 0.0

    def add_traffic(self, x: int, y: int, amount: float):
        if 0 <= x < self.width and 0 <= y < self.height and self.grid[y][x] == TileType.ROAD:
            self.traffic[y][x] += amount

    def decay_traffic(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == TileType.ROAD:
                    self.traffic[y][x] = max(0.0, self.traffic[y][x] * 0.95)

    def get_pollution(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pollution[y][x]
        return 0.0

    def add_pollution(self, x: int, y: int, amount: float):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pollution[y][x] += amount

    def decay_pollution(self):
        for y in range(self.height):
            for x in range(self.width):
                self.pollution[y][x] = max(0.0, self.pollution[y][x] * 0.9)

    def spread_pollution(self):
        new_pollution = [[0.0] * self.width for _ in range(self.height)]
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        for y in range(self.height):
            for x in range(self.width):
                total = self.pollution[y][x]
                count = 1
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        total += self.pollution[ny][nx] * 0.1
                        count += 0.1
                new_pollution[y][x] = total / count
        self.pollution = new_pollution

    def reduce_pollution_by_parks(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == TileType.PARK:
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                self.pollution[ny][nx] = max(0.0, self.pollution[ny][nx] - 0.05)

    def is_connected_to_road(self, x: int, y: int) -> bool:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            if self.get_tile(x + dx, y + dy) == TileType.ROAD:
                return True
        return False

    def has_power_plant(self) -> bool:
        return any(TileType.POWER_PLANT in row for row in self.grid)

    def count_building(self, tile_type: TileType) -> int:
        return sum(row.count(tile_type) for row in self.grid)

# ==================== CAR ====================
class Car:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        self.speed = 0.15
        self.color = random.choice([(255,0,0), (0,0,255), (255,255,255), (0,255,0)])

    def update(self, city_grid: CityGrid):
        dx, dy = self.direction
        nx, ny = int(self.x + dx), int(self.y + dy)
        if city_grid.get_tile(nx, ny) == TileType.ROAD:
            traffic = city_grid.get_traffic(nx, ny)
            speed = self.speed * max(0.1, 1.0 - traffic * 0.8)
            self.x += dx * speed
            self.y += dy * speed
        else:
            self.direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])

    def draw(self, screen, camera_x, camera_y):
        sx = int((self.x - camera_x) * TILE_SIZE + TILE_SIZE // 2)
        sy = int((self.y - camera_y) * TILE_SIZE + TILE_SIZE // 2)
        pygame.draw.circle(screen, self.color, (sx, sy), 4)

# ==================== GAME STATE ====================
class GameState:
    def __init__(self):
        self.money = 25000
        self.population = 0
        self.res_demand = 20
        self.com_demand = 10
        self.ind_demand = 10
        self.current_tool = TileType.ZONE_RESIDENTIAL
        self.tools = [
            TileType.ZONE_RESIDENTIAL,
            TileType.ZONE_COMMERCIAL,
            TileType.ZONE_INDUSTRIAL,
            TileType.ROAD,
            TileType.POWER_PLANT,
            TileType.PARK
        ]
        self.camera_x = 20
        self.camera_y = 20
        self.cars: List[Car] = []
        self.city_grid = None  # Will be set in main

# ==================== RENDERER ====================
class Renderer:
    def __init__(self, screen, font, font_small):
        self.screen = screen
        self.font = font
        self.font_small = font_small

    def draw_tile(self, tile_type: TileType, traffic: float, pollution: float, sx: int, sy: int):
        rect = pygame.Rect(sx, sy, TILE_SIZE, TILE_SIZE)

        if tile_type == TileType.EMPTY:
            pygame.draw.rect(self.screen, COLORS['grass'], rect)
            for i in range(0, TILE_SIZE, 4):
                pygame.draw.line(self.screen, COLORS['grass_dark'], (sx + i, sy), (sx + i, sy + TILE_SIZE), 1)

        elif tile_type == TileType.ZONE_RESIDENTIAL:
            pygame.draw.rect(self.screen, (144, 238, 144), rect)
            for i in range(3, TILE_SIZE, 6):
                pygame.draw.line(self.screen, (100, 200, 100), (sx, sy + i), (sx + TILE_SIZE, sy + i), 2)

        elif tile_type == TileType.ZONE_COMMERCIAL:
            pygame.draw.rect(self.screen, (173, 216, 230), rect)
            for i in range(3, TILE_SIZE, 6):
                pygame.draw.line(self.screen, (100, 150, 200), (sx + i, sy), (sx + i, sy + TILE_SIZE), 2)

        elif tile_type == TileType.ZONE_INDUSTRIAL:
            pygame.draw.rect(self.screen, (255, 248, 220), rect)
            for i in range(0, TILE_SIZE, 5):
                pygame.draw.line(self.screen, (200, 200, 150), (sx + i, sy + i), (sx + i + 5, sy + i + 5), 2)

        elif tile_type == TileType.RESIDENTIAL:
            pygame.draw.rect(self.screen, COLORS['residential'], rect)
            roof_h = TILE_SIZE // 3
            pygame.draw.polygon(self.screen, COLORS['res_roof'],
                                [(sx, sy + roof_h), (sx + TILE_SIZE//2, sy), (sx + TILE_SIZE, sy + roof_h)])
            pygame.draw.rect(self.screen, (255, 255, 200), (sx + 4, sy + roof_h + 4, 7, 7))
            pygame.draw.rect(self.screen, (255, 255, 200), (sx + TILE_SIZE - 11, sy + roof_h + 4, 7, 7))
            pygame.draw.rect(self.screen, (139, 69, 19), (sx + TILE_SIZE//2 - 5, sy + TILE_SIZE - 12, 10, 12))

        elif tile_type == TileType.COMMERCIAL:
            pygame.draw.rect(self.screen, COLORS['commercial'], rect)
            pygame.draw.rect(self.screen, COLORS['com_roof'], (sx + 2, sy + 2, TILE_SIZE - 4, TILE_SIZE // 4))
            win = pygame.Rect(sx + 4, sy + TILE_SIZE//4 + 2, TILE_SIZE - 8, TILE_SIZE//2 - 4)
            pygame.draw.rect(self.screen, (220, 240, 255), win)
            pygame.draw.rect(self.screen, (0, 0, 0), win, 1)
            pygame.draw.rect(self.screen, (139, 69, 19), (sx + TILE_SIZE//2 - 6, sy + TILE_SIZE - 14, 12, 14))

        elif tile_type == TileType.INDUSTRIAL:
            pygame.draw.rect(self.screen, COLORS['industrial'], rect)
            pygame.draw.rect(self.screen, COLORS['ind_roof'], (sx, sy, TILE_SIZE, TILE_SIZE // 4))
            pygame.draw.rect(self.screen, (100, 100, 100), (sx + TILE_SIZE - 8, sy + 5, 7, TILE_SIZE // 2))

        elif tile_type == TileType.ROAD:
            pygame.draw.rect(self.screen, COLORS['road_base'], rect)
            pygame.draw.rect(self.screen, COLORS['road_line'], (sx + TILE_SIZE//2 - 1, sy, 2, TILE_SIZE))
            pygame.draw.rect(self.screen, COLORS['road_line'], (sx, sy + TILE_SIZE//2 - 1, TILE_SIZE, 2))

        elif tile_type == TileType.POWER_PLANT:
            pygame.draw.rect(self.screen, COLORS['power_plant'], rect)
            pygame.draw.rect(self.screen, (80, 80, 80), (sx + 4, sy + 4, 7, TILE_SIZE - 8))
            pygame.draw.rect(self.screen, (80, 80, 80), (sx + TILE_SIZE - 11, sy + 4, 7, TILE_SIZE - 8))
            smoke_y = sy + 8 + (pygame.time.get_ticks() // 100 % 4)
            pygame.draw.circle(self.screen, COLORS['pp_smoke'], (sx + TILE_SIZE//2, smoke_y), 5)

        elif tile_type == TileType.PARK:
            pygame.draw.rect(self.screen, COLORS['park'], rect)
            trunk = pygame.Rect(sx + TILE_SIZE//2 - 2, sy + TILE_SIZE//2, 4, TILE_SIZE//2)
            pygame.draw.rect(self.screen, (139, 69, 19), trunk)
            pygame.draw.circle(self.screen, COLORS['park_tree'], (sx + TILE_SIZE//2, sy + TILE_SIZE//4), TILE_SIZE//3)
            pygame.draw.circle(self.screen, COLORS['park_tree'], (sx + TILE_SIZE//3, sy + TILE_SIZE//3), TILE_SIZE//4)
            pygame.draw.circle(self.screen, COLORS['park_tree'], (sx + TILE_SIZE*2//3, sy + TILE_SIZE//3), TILE_SIZE//4)

        # Traffic overlay
        if tile_type == TileType.ROAD and traffic > 0.1:
            if traffic > 1.2: col = COLORS['traffic_jam']
            elif traffic > 0.8: col = COLORS['traffic_high']
            elif traffic > 0.4: col = COLORS['traffic_med']
            else: col = COLORS['traffic_low']
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            surf.fill((*col, min(180, int(traffic * 150))))
            self.screen.blit(surf, (sx, sy))

        # Pollution overlay
        if pollution > 0.1:
            if pollution > 1.5: col = COLORS['pollution_severe']
            elif pollution > 1.0: col = COLORS['pollution_high']
            elif pollution > 0.5: col = COLORS['pollution_med']
            else: col = COLORS['pollution_low']
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            surf.fill((*col, min(150, int(pollution * 100))))
            self.screen.blit(surf, (sx, sy))

        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

    def draw_grid(self, city_grid: CityGrid, cam_x: int, cam_y: int, cars: List[Car]):
        start_x = max(0, cam_x)
        end_x = min(city_grid.width, cam_x + VIEW_WIDTH + 1)
        start_y = max(0, cam_y)
        end_y = min(city_grid.height, cam_y + VIEW_HEIGHT + 1)

        for gy in range(start_y, end_y):
            for gx in range(start_x, end_x):
                tile = city_grid.get_tile(gx, gy)
                traffic = city_grid.get_traffic(gx, gy)
                pollution = city_grid.get_pollution(gx, gy)
                sx = (gx - cam_x) * TILE_SIZE
                sy = (gy - cam_y) * TILE_SIZE
                self.draw_tile(tile, traffic, pollution, sx, sy)

        for car in cars:
            if start_x <= car.x < end_x and start_y <= car.y < end_y:
                car.draw(self.screen, cam_x, cam_y)

    def draw_ui(self, game_state: GameState):
        ui_y = SCREEN_HEIGHT - 80
        pygame.draw.rect(self.screen, COLORS['ui_bg'], (0, ui_y, SCREEN_WIDTH, 80))

        road_count = game_state.city_grid.count_building(TileType.ROAD)
        avg_traffic = sum(sum(row) for row in game_state.city_grid.traffic) / max(1, road_count)
        traffic_status = "Low" if avg_traffic < 0.4 else "Med" if avg_traffic < 0.8 else "High" if avg_traffic < 1.2 else "Jam!"

        total_cells = GRID_WIDTH * GRID_HEIGHT
        avg_pollution = sum(sum(row) for row in game_state.city_grid.pollution) / total_cells
        pollution_status = "Clean" if avg_pollution < 0.3 else "Moderate" if avg_pollution < 0.7 else "High" if avg_pollution < 1.2 else "Severe"

        status = (f"Money: ${game_state.money:,}  |  Pop: {game_state.population:,}  |  "
                  f"Traffic: {traffic_status}  |  Pollution: {pollution_status}  |  "
                  f"R:{game_state.res_demand} C:{game_state.com_demand} I:{game_state.ind_demand}  |  "
                  f"Tool: {TOOL_NAMES[game_state.current_tool]}")
        text_surf = self.font.render(status, True, COLORS['text'])
        self.screen.blit(text_surf, (20, ui_y + 25))

        for i, tool in enumerate(game_state.tools):
            x = SCREEN_WIDTH - 360 + i * 60
            color = COLORS['selected_tool'] if tool == game_state.current_tool else COLORS['unselected_tool']
            pygame.draw.rect(self.screen, color, (x, ui_y + 20, 50, 40), 4)
            label_surf = self.font_small.render(TOOL_LABELS[tool], True, COLORS['text'])
            self.screen.blit(label_surf, (x + 10, ui_y + 30))

# ==================== SIMULATION ====================
class Simulation:
    @staticmethod
    def place_item(city_grid: CityGrid, game_state: GameState, x: int, y: int) -> bool:
        if city_grid.get_tile(x, y) != TileType.EMPTY:
            return False
        cost = COSTS.get(game_state.current_tool, 0)
        if game_state.money < cost:
            return False
        city_grid.set_tile(x, y, game_state.current_tool)
        game_state.money -= cost
        return True

    @staticmethod
    def generate_traffic(city_grid: CityGrid, game_state: GameState):
        for y in range(city_grid.height):
            for x in range(city_grid.width):
                tile = city_grid.get_tile(x, y)
                if tile in (TileType.RESIDENTIAL, TileType.COMMERCIAL, TileType.INDUSTRIAL):
                    if city_grid.is_connected_to_road(x, y):
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                            city_grid.add_traffic(x + dx, y + dy, 0.02)

        if random.random() < 0.15 and len(game_state.cars) < 150:
            roads = [(x,y) for y in range(city_grid.height) for x in range(city_grid.width) if city_grid.get_tile(x,y) == TileType.ROAD]
            if roads:
                rx, ry = random.choice(roads)
                game_state.cars.append(Car(rx + 0.5, ry + 0.5))

    @staticmethod
    def generate_pollution(city_grid: CityGrid):
        for y in range(city_grid.height):
            for x in range(city_grid.width):
                tile = city_grid.get_tile(x, y)
                if tile == TileType.INDUSTRIAL:
                    city_grid.add_pollution(x, y, 0.5)
                elif tile == TileType.POWER_PLANT:
                    city_grid.add_pollution(x, y, 0.3)
                elif tile == TileType.ROAD:
                    city_grid.add_pollution(x, y, city_grid.get_traffic(x, y) * 0.1)

    @staticmethod
    def develop_zones(city_grid: CityGrid, game_state: GameState):
        game_state.res_demand = max(0, min(100, city_grid.count_building(TileType.INDUSTRIAL) * 4 + random.randint(0, 30)))
        game_state.com_demand = max(0, min(100, city_grid.count_building(TileType.RESIDENTIAL) * 2 + random.randint(0, 25)))
        game_state.ind_demand = max(0, min(100, city_grid.count_building(TileType.COMMERCIAL) * 2 + random.randint(0, 20)))

        has_power = city_grid.has_power_plant()

        for y in range(city_grid.height):
            for x in range(city_grid.width):
                tile = city_grid.get_tile(x, y)
                pollution = city_grid.get_pollution(x, y)
                pollution_factor = max(0.1, 1.0 - pollution * 0.5)

                if tile == TileType.ZONE_RESIDENTIAL and has_power and city_grid.is_connected_to_road(x, y):
                    if random.random() < (game_state.res_demand / 120.0) * pollution_factor:
                        city_grid.set_tile(x, y, TileType.RESIDENTIAL)
                        game_state.population += random.randint(30, 60)
                elif tile == TileType.ZONE_COMMERCIAL and has_power and city_grid.is_connected_to_road(x, y):
                    if random.random() < (game_state.com_demand / 120.0) * pollution_factor:
                        city_grid.set_tile(x, y, TileType.COMMERCIAL)
                        game_state.population += random.randint(10, 30)
                elif tile == TileType.ZONE_INDUSTRIAL and has_power and city_grid.is_connected_to_road(x, y):
                    if random.random() < (game_state.ind_demand / 120.0) * pollution_factor:
                        city_grid.set_tile(x, y, TileType.INDUSTRIAL)
                        game_state.population += random.randint(5, 20)

    @staticmethod
    def simulate_monthly(city_grid: CityGrid, game_state: GameState):
        Simulation.develop_zones(city_grid, game_state)
        Simulation.generate_traffic(city_grid, game_state)
        Simulation.generate_pollution(city_grid)
        city_grid.spread_pollution()
        city_grid.decay_pollution()
        city_grid.reduce_pollution_by_parks()
        city_grid.decay_traffic()
        game_state.money += game_state.population // 10  # Taxes

# ==================== MAIN ====================
def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Advanced SimCity Clone - Zoning, Traffic & Pollution")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22, bold=True)
    font_small = pygame.font.SysFont("consolas", 16)

    city_grid = CityGrid(GRID_WIDTH, GRID_HEIGHT)
    game_state = GameState()
    game_state.city_grid = city_grid
    renderer = Renderer(screen, font, font_small)

    month_timer = 0
    MONTH_INTERVAL = 3000

    running = True
    while running:
        dt = clock.tick(FPS)
        month_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                if my < SCREEN_HEIGHT - 80:
                    gx = game_state.camera_x + mx // TILE_SIZE
                    gy = game_state.camera_y + my // TILE_SIZE
                    Simulation.place_item(city_grid, game_state, gx, gy)
                else:
                    if SCREEN_WIDTH - 360 <= mx < SCREEN_WIDTH:
                        idx = (mx - (SCREEN_WIDTH - 360)) // 60
                        if 0 <= idx < len(game_state.tools):
                            game_state.current_tool = game_state.tools[idx]
            elif event.type == pygame.KEYDOWN:
                key_map = {
                    pygame.K_1: TileType.ZONE_RESIDENTIAL,
                    pygame.K_2: TileType.ZONE_COMMERCIAL,
                    pygame.K_3: TileType.ZONE_INDUSTRIAL,
                    pygame.K_4: TileType.ROAD,
                    pygame.K_5: TileType.POWER_PLANT,
                    pygame.K_6: TileType.PARK,
                }
                if event.key in key_map:
                    game_state.current_tool = key_map[event.key]

        # Camera movement
        keys = pygame.key.get_pressed()
        speed = 5
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:   game_state.camera_x = max(0, game_state.camera_x - speed)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:  game_state.camera_x = min(GRID_WIDTH - VIEW_WIDTH, game_state.camera_x + speed)
        if keys[pygame.K_w] or keys[pygame.K_UP]:     game_state.camera_y = max(0, game_state.camera_y - speed)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:   game_state.camera_y = min(GRID_HEIGHT - VIEW_HEIGHT, game_state.camera_y + speed)

        # Update cars
        for car in game_state.cars[:]:
            car.update(city_grid)
            if random.random() < 0.005:
                game_state.cars.remove(car)

        # Monthly simulation
        if month_timer >= MONTH_INTERVAL:
            Simulation.simulate_monthly(city_grid, game_state)
            month_timer = 0

        # Render
        screen.fill((10, 20, 40))
        renderer.draw_grid(city_grid, game_state.camera_x, game_state.camera_y, game_state.cars)
        renderer.draw_ui(game_state)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
