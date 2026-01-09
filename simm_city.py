import pygame
import sys
import random
from enum import Enum

# Initialize Pygame
pygame.init()

# Constants
TILE_SIZE = 16
GRID_WIDTH = 30
GRID_HEIGHT = 30
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE + 60
FPS = 60

# Colors (8-bit retro style)
COLORS = {
    'grass': (0, 100, 0),
    'residential': (0, 255, 0),
    'commercial': (0, 0, 255),
    'industrial': (255, 255, 0),
    'road': (100, 100, 100),
    'power_plant': (255, 0, 0),
    'grid_line': (0, 0, 0),
    'ui_bg': (50, 50, 50),
    'text': (255, 255, 255),
    'selected_tool': (0, 255, 0),
    'unselected_tool': (200, 200, 200)
}

class TileType(Enum):
    EMPTY = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    ROAD = 4
    POWER_PLANT = 5

BUILDING_COSTS = {
    TileType.RESIDENTIAL: 100,
    TileType.COMMERCIAL: 100,
    TileType.INDUSTRIAL: 100,
    TileType.ROAD: 50,
    TileType.POWER_PLANT: 500
}

INITIAL_POPULATION_PER_ZONE = {
    TileType.RESIDENTIAL: 10,
    TileType.COMMERCIAL: 5,
    TileType.INDUSTRIAL: 5
}

TOOL_NAMES = {
    TileType.RESIDENTIAL: "Res",
    TileType.COMMERCIAL: "Com",
    TileType.INDUSTRIAL: "Ind",
    TileType.ROAD: "Road",
    TileType.POWER_PLANT: "Power"
}

class CityGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[TileType.EMPTY for _ in range(width)] for _ in range(height)]

    def get_tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def set_tile(self, x, y, tile_type):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = tile_type

    def is_connected_to_road(self, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            if self.get_tile(x + dx, y + dy) == TileType.ROAD:
                return True
        return False

    def has_power_plant(self):
        for row in self.grid:
            if TileType.POWER_PLANT in row:
                return True
        return False

class GameState:
    def __init__(self):
        self.money = 10000
        self.population = 0
        self.current_tool = TileType.RESIDENTIAL
        self.tools = [TileType.RESIDENTIAL, TileType.COMMERCIAL, TileType.INDUSTRIAL,
                      TileType.ROAD, TileType.POWER_PLANT]

class Renderer:
    def __init__(self, screen, font):
        self.screen = screen
        self.font = font

    def draw_grid(self, city_grid):
        for y in range(city_grid.height):
            for x in range(city_grid.width):
                tile = city_grid.get_tile(x, y)
                color = COLORS['grass']
                if tile == TileType.RESIDENTIAL:
                    color = COLORS['residential']
                elif tile == TileType.COMMERCIAL:
                    color = COLORS['commercial']
                elif tile == TileType.INDUSTRIAL:
                    color = COLORS['industrial']
                elif tile == TileType.ROAD:
                    color = COLORS['road']
                elif tile == TileType.POWER_PLANT:
                    color = COLORS['power_plant']

                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLORS['grid_line'], rect, 1)

    def draw_ui(self, game_state):
        ui_y = GRID_HEIGHT * TILE_SIZE
        pygame.draw.rect(self.screen, COLORS['ui_bg'], (0, ui_y, SCREEN_WIDTH, 60))

        status_text = f"Money: ${game_state.money}   Pop: {game_state.population}   Tool: {TOOL_NAMES[game_state.current_tool]}"
        text_surf = self.font.render(status_text, True, COLORS['text'])
        self.screen.blit(text_surf, (10, ui_y + 20))

        # Tool selector
        for i, tool in enumerate(game_state.tools):
            color = COLORS['selected_tool'] if tool == game_state.current_tool else COLORS['unselected_tool']
            pygame.draw.rect(self.screen, color,
                             (SCREEN_WIDTH - 220 + i * 45, ui_y + 15, 35, 35), 3)

class Simulation:
    @staticmethod
    def place_building(city_grid, game_state, x, y):
        if city_grid.get_tile(x, y) != TileType.EMPTY:
            return False

        cost = BUILDING_COSTS.get(game_state.current_tool, 0)
        if game_state.money < cost:
            return False

        city_grid.set_tile(x, y, game_state.current_tool)
        game_state.money -= cost

        if game_state.current_tool in INITIAL_POPULATION_PER_ZONE:
            game_state.population += INITIAL_POPULATION_PER_ZONE[game_state.current_tool]

        return True

    @staticmethod
    def monthly_update(city_grid, game_state):
        has_power = city_grid.has_power_plant()
        growth = 0
        decline = 0

        for y in range(city_grid.height):
            for x in range(city_grid.width):
                tile = city_grid.get_tile(x, y)
                if tile in (TileType.RESIDENTIAL, TileType.COMMERCIAL, TileType.INDUSTRIAL):
                    if has_power and city_grid.is_connected_to_road(x, y):
                        growth += random.randint(1, 5)
                    else:
                        decline += random.randint(0, 2)

        game_state.population += growth - decline
        game_state.population = max(0, game_state.population)

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Modular 8-bit SimCity")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)

    city_grid = CityGrid(GRID_WIDTH, GRID_HEIGHT)
    game_state = GameState()
    renderer = Renderer(screen, font)

    month_timer = 0
    MONTH_INTERVAL = 5000  # ms

    running = True
    while running:
        dt = clock.tick(FPS)
        month_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                grid_x, grid_y = mx // TILE_SIZE, my // TILE_SIZE

                if my < GRID_HEIGHT * TILE_SIZE:
                    Simulation.place_building(city_grid, game_state, grid_x, grid_y)
                else:
                    # Tool selection
                    if SCREEN_WIDTH - 220 <= mx < SCREEN_WIDTH:
                        idx = (mx - (SCREEN_WIDTH - 220)) // 45
                        if idx < len(game_state.tools):
                            game_state.current_tool = game_state.tools[idx]

            elif event.type == pygame.KEYDOWN:
                key_to_tool = {
                    pygame.K_1: TileType.RESIDENTIAL,
                    pygame.K_2: TileType.COMMERCIAL,
                    pygame.K_3: TileType.INDUSTRIAL,
                    pygame.K_4: TileType.ROAD,
                    pygame.K_5: TileType.POWER_PLANT
                }
                if event.key in key_to_tool:
                    game_state.current_tool = key_to_tool[event.key]

        if month_timer >= MONTH_INTERVAL:
            Simulation.monthly_update(city_grid, game_state)
            month_timer = 0

        screen.fill((0, 0, 0))
        renderer.draw_grid(city_grid)
        renderer.draw_ui(game_state)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
