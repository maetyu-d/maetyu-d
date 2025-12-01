import math
import random
import pygame

# ============================================
# 10k x 10k WORLD, 4-COLUMN SPLIT VIEW
# --------------------------------------------
# OPTION 1: TWO INDEPENDENT TREES
#
# COL 0: ORANGE (side)  — baseline bots, own tree
# COL 1: ORANGE (iso)   — baseline bots, own tree
# COL 2: GREEN (side)   — disabled-gait bots, own tree
# COL 3: GREEN (iso)    — disabled-gait bots, own tree
#
# Each colour group:
# - Has its own node tree, targets, pruning, links, and vertical structure.
# - Picks a random node per level, independently of the other group.
#
# This version:
# - Labels the columns with: orange (side), orange (iso), green (side), green (iso)
# - Shows per-column "active: <title>" labels
# - Displays a full-width banner with:
#       ORANGE: <title>    |    GREEN: <title>
#   whenever either target changes, for at least 1 second.
# ============================================

WORLD_SIZE = 10000
SCREEN_WIDTH = 1000   # single window, 4 views
SCREEN_HEIGHT = 1000

VIEW_COLS = 4
VIEW_WIDTH = SCREEN_WIDTH // VIEW_COLS  # 250 px per view

# Plane extents in world units
PLANE_HALF = WORLD_SIZE * 0.4  # nodes in roughly [-PLANE_HALF, PLANE_HALF]

NUM_BOTS_PER_GROUP = 100
BOT_SPEED = 600             # base units/s for orange bots
BOT_RADIUS = 200            # "reached" radius

MAX_LEVELS = 100
MAX_LINKS_PER_LEVEL = 12

# Vertical separation between levels (height axis)
TOTAL_TOWER_HEIGHT = WORLD_SIZE * 0.4
LEVEL_HEIGHT_3D = TOTAL_TOWER_HEIGHT / MAX_LEVELS

# --- Projection scales ---
SIDE_SCALE_X = (VIEW_WIDTH * 0.45) / PLANE_HALF
SIDE_SCALE_Z = (SCREEN_HEIGHT * 0.7) / TOTAL_TOWER_HEIGHT

MARGIN_FACTOR = 0.8
ISO_MARGIN = PLANE_HALF * MARGIN_FACTOR
ISO_SCALE = (VIEW_WIDTH * 0.45) / (2.0 * ISO_MARGIN * 2.0)

# Colors
COLOR_BG = (10, 10, 20)
COLOR_NODE = (80, 160, 255)
COLOR_NODE_ACTIVE = (255, 255, 255)
COLOR_BOT_ORANGE = (255, 140, 0)
COLOR_BOT_LIME = (0, 255, 100)
COLOR_STRUCTURE = (220, 220, 220)
COLOR_LINK = (180, 120, 255)
COLOR_TEXT = (230, 230, 230)
COLOR_LEVEL_OUTLINE = (40, 40, 70)
COLOR_BOT_FOCUS = (255, 60, 60)
COLOR_COL_BG = [
    (15, 15, 30),
    (20, 10, 30),
    (10, 20, 30),
    (20, 20, 10),
]

# Banner timing globals
banner_until_time = 0.0
last_orange_title = ""
last_green_title = ""


# ==============================
# DATA STRUCTURES
# ==============================

class Node:
    def __init__(self, title, x, y, level):
        self.title = title
        self.x = x
        self.y = y
        self.level = level
        self.children_titles = []
        self.expanded = False

    @property
    def pos(self):
        return (self.x, self.y)


class Bot:
    """Baseline bot with symmetric gait."""
    def __init__(self, x, y, level):
        self.x = x
        self.y = y
        self.level = level
        self.target_node = None
        self.reached = False

    @property
    def pos(self):
        return (self.x, self.y)

    def set_target(self, node):
        self.target_node = node
        self.reached = False
        self.level = node.level

    def update(self, dt):
        if self.target_node is None or self.reached:
            return

        tx, ty = self.target_node.pos
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            self.reached = True
            return

        if dist <= BOT_RADIUS:
            self.x, self.y = tx, ty
            self.reached = True
            return

        step = BOT_SPEED * dt
        if step >= dist:
            self.x, self.y = tx, ty
            self.reached = True
        else:
            self.x += dx / dist * step
            self.y += dy / dist * step


class DisabledBot(Bot):
    """
    Bot with a leg length discrepancy and reduced stability/efficiency.
    """
    def __init__(self, x, y, level):
        super().__init__(x, y, level)
        self.gait_phase = random.uniform(0, 2 * math.pi)
        self.gait_freq = random.uniform(1.2, 2.0)
        self.leg_diff = random.uniform(0.12, 0.3)
        self.fatigue = 0.0
        self.base_speed_factor = random.uniform(0.55, 0.75)
        self.stumble_chance = 0.003

    def update(self, dt):
        if self.target_node is None or self.reached:
            return

        tx, ty = self.target_node.pos
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            self.reached = True
            return

        if dist <= BOT_RADIUS:
            self.x, self.y = tx, ty
            self.reached = True
            return

        # Advance gait
        self.gait_phase += self.gait_freq * dt * 2 * math.pi

        # Base direction
        dir_x = dx / dist
        dir_y = dy / dist

        # Lateral direction
        lat_x = -dir_y
        lat_y = dir_x

        # Lateral wobble
        wobble_amp = self.leg_diff * (0.6 + 0.6 * self.fatigue)
        wobble = math.sin(self.gait_phase) * wobble_amp

        move_x = dir_x + lat_x * wobble
        move_y = dir_y + lat_y * wobble
        move_len = math.hypot(move_x, move_y)
        if move_len < 1e-6:
            move_x, move_y = dir_x, dir_y
            move_len = 1.0
        move_x /= move_len
        move_y /= move_len

        fatigue_factor = max(0.3, 1.0 - self.fatigue * 0.7)
        speed = BOT_SPEED * self.base_speed_factor * fatigue_factor
        step = speed * dt

        # Occasional stumble
        if random.random() < self.stumble_chance:
            stumble_mag = step * 0.6
            move_x += lat_x * self.leg_diff * stumble_mag
            move_y += lat_y * self.leg_diff * stumble_mag
            mlen2 = math.hypot(move_x, move_y)
            if mlen2 > 1e-6:
                move_x /= mlen2
                move_y /= mlen2

        if step >= dist:
            self.x, self.y = tx, ty
            self.reached = True
        else:
            self.x += move_x * step
            self.y += move_y * step

        # Fatigue accumulation
        effort = step * (1.0 + abs(wobble) * 3.0 + self.leg_diff * 2.0)
        self.fatigue += effort * 0.00002
        if self.fatigue > 1.0:
            self.fatigue = 1.0


class StructureSegment:
    def __init__(self, x, y, base_level, top_level):
        self.x = x
        self.y = y
        self.base_level = base_level
        self.top_level = top_level


class LinkSegment:
    def __init__(self, x1, y1, level1, x2, y2, level2):
        self.x1 = x1
        self.y1 = y1
        self.level1 = level1
        self.x2 = x2
        self.y2 = y2
        self.level2 = level2


# ==============================
# SYNTHETIC GRAPH
# ==============================

def generate_synthetic_titles(level_index, count, tag):
    return [f"{tag}L{level_index}-N{i}" for i in range(count)]


# ==============================
# LAYOUT
# ==============================

def layout_level(node_titles, level_index):
    nodes = []
    if not node_titles:
        return nodes
    margin = PLANE_HALF * MARGIN_FACTOR
    for t in node_titles:
        x = random.uniform(-margin, margin)
        y = random.uniform(-margin, margin)
        nodes.append(Node(t, x, y, level_index))
    return nodes


# ==============================
# BOT & TREE LOGIC
# ==============================

def initialize_bots(num_bots, start_level, disabled=False):
    bots = []
    jitter = PLANE_HALF * 0.1
    for _ in range(num_bots):
        bx = random.uniform(-jitter, jitter)
        by = random.uniform(-jitter, jitter)
        if disabled:
            bots.append(DisabledBot(bx, by, start_level))
        else:
            bots.append(Bot(bx, by, start_level))
    return bots


def assign_bots_to_node(bots, node):
    for b in bots:
        b.set_target(node)


def all_bots_reached(bots):
    return all(b.reached for b in bots)


def any_bot_reached(bots):
    return any(b.reached for b in bots)


def choose_target(nodes_per_level, level_index):
    if level_index >= len(nodes_per_level):
        return None
    level_nodes = nodes_per_level[level_index]
    if not level_nodes:
        return None
    return random.choice(level_nodes)


# ==============================
# PROJECTIONS
# ==============================

def level_to_height(level_index):
    return level_index * LEVEL_HEIGHT_3D


def iso_project(x, y, level_index, col_index):
    view_offset_x = col_index * VIEW_WIDTH
    z = level_to_height(level_index)
    iso_x = (x - y)
    iso_y = (x + y) * 0.5 - z * 1.2
    ix = iso_x * ISO_SCALE
    iy = iso_y * ISO_SCALE
    sx = int(view_offset_x + VIEW_WIDTH / 2 + ix)
    sy = int(SCREEN_HEIGHT * 0.75 + iy)
    return sx, sy


def side_project(x, y, level_index, col_index):
    view_offset_x = col_index * VIEW_WIDTH
    z = level_to_height(level_index)
    sx = int(view_offset_x + VIEW_WIDTH / 2 + x * SIDE_SCALE_X)
    sy = int(SCREEN_HEIGHT * 0.8 - z * SIDE_SCALE_Z)
    return sx, sy


def draw_text(surface, text, pos, font, color=COLOR_TEXT):
    img = font.render(text, True, color)
    surface.blit(img, pos)


# ==============================
# DRAWING HELPERS
# ==============================

def draw_level_outline_side(screen, level_index, col_index):
    y_dummy = 0.0
    p1 = side_project(-PLANE_HALF * MARGIN_FACTOR, y_dummy, level_index, col_index)
    p2 = side_project(PLANE_HALF * MARGIN_FACTOR, y_dummy, level_index, col_index)
    pygame.draw.line(screen, COLOR_LEVEL_OUTLINE, p1, p2, 1)


def draw_level_outline_iso(screen, level_index, col_index):
    corners = [
        (0, -PLANE_HALF * MARGIN_FACTOR),
        (PLANE_HALF * MARGIN_FACTOR, 0),
        (0, PLANE_HALF * MARGIN_FACTOR),
        (-PLANE_HALF * MARGIN_FACTOR, 0),
    ]
    pts = [iso_project(cx, cy, level_index, col_index) for (cx, cy) in corners]
    pygame.draw.polygon(screen, COLOR_LEVEL_OUTLINE, pts, 1)


# ==============================
# DRAW SCENE
# ==============================

def draw_scene(screen, font,
               nodes_orange, nodes_green,
               bots_orange, bots_green,
               struct_orange, struct_green,
               links_orange, links_green,
               cur_lvl_orange, cur_lvl_green,
               tgt_orange, tgt_green,
               level_complete_orange, level_complete_green,
               built_orange, built_green):
    global banner_until_time, last_orange_title, last_green_title

    # Column backgrounds
    for c in range(VIEW_COLS):
        x0 = c * VIEW_WIDTH
        rect = pygame.Rect(x0, 0, VIEW_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, COLOR_COL_BG[c], rect)

    # Vertical split lines
    for c in range(1, VIEW_COLS):
        x = c * VIEW_WIDTH
        pygame.draw.line(screen, (80, 80, 80), (x, 0), (x, SCREEN_HEIGHT), 2)

    orange_title = tgt_orange.title if tgt_orange is not None else "None"
    green_title = tgt_green.title if tgt_green is not None else "None"

    # ---------- ORANGE TREE ----------
    # COL 0: ORANGE (side)
    col = 0
    for lvl_idx, lvl_nodes in enumerate(nodes_orange):
        if not lvl_nodes:
            continue
        draw_level_outline_side(screen, lvl_idx, col)

    for seg in struct_orange:
        p1 = side_project(seg.x, seg.y, seg.base_level, col)
        p2 = side_project(seg.x, seg.y, seg.top_level, col)
        pygame.draw.line(screen, COLOR_STRUCTURE, p1, p2, 4)

    for link in links_orange:
        p1 = side_project(link.x1, link.y1, link.level1, col)
        p2 = side_project(link.x2, link.y2, link.level2, col)
        pygame.draw.line(screen, COLOR_LINK, p1, p2, 2)

    for lvl_nodes in nodes_orange:
        for node in lvl_nodes:
            sx, sy = side_project(node.x, node.y, node.level, col)
            if not (col * VIEW_WIDTH <= sx <= (col + 1) * VIEW_WIDTH):
                continue
            pygame.draw.circle(
                screen,
                COLOR_NODE_ACTIVE if node is tgt_orange else COLOR_NODE,
                (sx, sy),
                4
            )

    for b in bots_orange:
        sx, sy = side_project(b.x, b.y, b.level, col)
        if not (col * VIEW_WIDTH <= sx <= (col + 1) * VIEW_WIDTH):
            continue
        pygame.draw.circle(screen, COLOR_BOT_ORANGE, (sx, sy), 3)

    if bots_orange:
        avg_x = sum(b.x for b in bots_orange) / len(bots_orange)
        avg_y = sum(b.y for b in bots_orange) / len(bots_orange)
        ax, ay = side_project(avg_x, avg_y, bots_orange[0].level, col)
        if col * VIEW_WIDTH <= ax <= (col + 1) * VIEW_WIDTH:
            pygame.draw.circle(screen, COLOR_BOT_FOCUS, (ax, ay), 6, 1)

    draw_text(screen, "orange (side)", (5, 5), font, COLOR_BOT_ORANGE)
    draw_text(screen, f"active: {orange_title}", (5, 25), font)

    # COL 1: ORANGE (iso)
    col = 1
    for lvl_idx, lvl_nodes in enumerate(nodes_orange):
        if not lvl_nodes:
            continue
        draw_level_outline_iso(screen, lvl_idx, col)

    for seg in struct_orange:
        p1 = iso_project(seg.x, seg.y, seg.base_level, col)
        p2 = iso_project(seg.x, seg.y, seg.top_level, col)
        pygame.draw.line(screen, COLOR_STRUCTURE, p1, p2, 4)

    for link in links_orange:
        p1 = iso_project(link.x1, link.y1, link.level1, col)
        p2 = iso_project(link.x2, link.y2, link.level2, col)
        pygame.draw.line(screen, COLOR_LINK, p1, p2, 2)

    for lvl_nodes in nodes_orange:
        for node in lvl_nodes:
            sx, sy = iso_project(node.x, node.y, node.level, col)
            if not (col * VIEW_WIDTH <= sx <= (col + 1) * VIEW_WIDTH):
                continue
            pygame.draw.circle(
                screen,
                COLOR_NODE_ACTIVE if node is tgt_orange else COLOR_NODE,
                (sx, sy),
                5
            )

    for b in bots_orange:
        sx, sy = iso_project(b.x, b.y, b.level, col)
        if not (col * VIEW_WIDTH <= sx <= (col + 1) * VIEW_WIDTH):
            continue
        pygame.draw.circle(screen, COLOR_BOT_ORANGE, (sx, sy), 3)

    if bots_orange:
        avg_x = sum(b.x for b in bots_orange) / len(bots_orange)
        avg_y = sum(b.y for b in bots_orange) / len(bots_orange)
        ax, ay = iso_project(avg_x, avg_y, bots_orange[0].level, col)
        if col * VIEW_WIDTH <= ax <= (col + 1) * VIEW_WIDTH:
            pygame.draw.circle(screen, COLOR_BOT_FOCUS, (ax, ay), 6, 1)

    draw_text(screen, "orange (iso)", (VIEW_WIDTH + 5, 5), font, COLOR_BOT_ORANGE)
    draw_text(screen, f"active: {orange_title}", (VIEW_WIDTH + 5, 25), font)

    # ---------- GREEN TREE ----------
    # COL 2: GREEN (side)
    col = 2
    for lvl_idx, lvl_nodes in enumerate(nodes_green):
        if not lvl_nodes:
            continue
        draw_level_outline_side(screen, lvl_idx, col)

    for seg in struct_green:
        p1 = side_project(seg.x, seg.y, seg.base_level, col)
        p2 = side_project(seg.x, seg.y, seg.top_level, col)
        pygame.draw.line(screen, COLOR_STRUCTURE, p1, p2, 4)

    for link in links_green:
        p1 = side_project(link.x1, link.y1, link.level1, col)
        p2 = side_project(link.x2, link.y2, link.level2, col)
        pygame.draw.line(screen, COLOR_LINK, p1, p2, 2)

    for lvl_nodes in nodes_green:
        for node in lvl_nodes:
            sx, sy = side_project(node.x, node.y, node.level, col)
            if not (col * VIEW_WIDTH <= sx <= (col + 1) * VIEW_WIDTH):
                continue
            pygame.draw.circle(
                screen,
                COLOR_NODE_ACTIVE if node is tgt_green else COLOR_NODE,
                (sx, sy),
                4
            )

    for b in bots_green:
        sx, sy = side_project(b.x, b.y, b.level, col)
        if not (col * VIEW_WIDTH <= sx <= (col + 1) * VIEW_WIDTH):
            continue
        pygame.draw.circle(screen, COLOR_BOT_LIME, (sx, sy), 3)

    if bots_green:
        avg_x = sum(b.x for b in bots_green) / len(bots_green)
        avg_y = sum(b.y for b in bots_green) / len(bots_green)
        ax, ay = side_project(avg_x, avg_y, bots_green[0].level, col)
        if col * VIEW_WIDTH <= ax <= (col + 1) * VIEW_WIDTH:
            pygame.draw.circle(screen, COLOR_BOT_FOCUS, (ax, ay), 6, 1)

    draw_text(screen, "green (side)", (2 * VIEW_WIDTH + 5, 5), font, COLOR_BOT_LIME)
    draw_text(screen, f"active: {green_title}", (2 * VIEW_WIDTH + 5, 25), font)

    # COL 3: GREEN (iso)
    col = 3
    for lvl_idx, lvl_nodes in enumerate(nodes_green):
        if not lvl_nodes:
            continue
        draw_level_outline_iso(screen, lvl_idx, col)

    for seg in struct_green:
        p1 = iso_project(seg.x, seg.y, seg.base_level, col)
        p2 = iso_project(seg.x, seg.y, seg.top_level, col)
        pygame.draw.line(screen, COLOR_STRUCTURE, p1, p2, 4)

    for link in links_green:
        p1 = iso_project(link.x1, link.y1, link.level1, col)
        p2 = iso_project(link.x2, link.y2, link.level2, col)
        pygame.draw.line(screen, COLOR_LINK, p1, p2, 2)

    for lvl_nodes in nodes_green:
        for node in lvl_nodes:
            sx, sy = iso_project(node.x, node.y, node.level, col)
            if not (col * VIEW_WIDTH <= sx <= (col + 1) * VIEW_WIDTH):
                continue
            pygame.draw.circle(
                screen,
                COLOR_NODE_ACTIVE if node is tgt_green else COLOR_NODE,
                (sx, sy),
                5
            )

    for b in bots_green:
        sx, sy = iso_project(b.x, b.y, b.level, col)
        if not (col * VIEW_WIDTH <= sx <= (col + 1) * VIEW_WIDTH):
            continue
        pygame.draw.circle(screen, COLOR_BOT_LIME, (sx, sy), 3)

    if bots_green:
        avg_x = sum(b.x for b in bots_green) / len(bots_green)
        avg_y = sum(b.y for b in bots_green) / len(bots_green)
        ax, ay = iso_project(avg_x, avg_y, bots_green[0].level, col)
        if col * VIEW_WIDTH <= ax <= (col + 1) * VIEW_WIDTH:
            pygame.draw.circle(screen, COLOR_BOT_FOCUS, (ax, ay), 6, 1)

    draw_text(screen, "green (iso)", (3 * VIEW_WIDTH + 5, 5), font, COLOR_BOT_LIME)
    draw_text(screen, f"active: {green_title}", (3 * VIEW_WIDTH + 5, 25), font)

    # ----- TEXT OVERLAY -----
    bots_at_target_orange = sum(1 for b in bots_orange if b.reached)
    bots_at_target_green = sum(1 for b in bots_green if b.reached)

    info_lines = [
        "4-COLUMN VIEW — TWO INDEPENDENT TREES",
        f"ORANGE:   Level {cur_lvl_orange + 1} / {MAX_LEVELS},   Levels built: {built_orange},   Bots at target: {bots_at_target_orange}/{len(bots_orange)}",
        f"GREEN(DISABLED): Level {cur_lvl_green + 1} / {MAX_LEVELS},   Levels built: {built_green},   Bots at target: {bots_at_target_green}/{len(bots_green)}",
        "SPACE: toggle pause   ESC: quit"
    ]
    y = SCREEN_HEIGHT - 22 * len(info_lines) - 30
    for line in info_lines:
        draw_text(screen, line, (10, y), font)
        y += 22

    if level_complete_orange:
        draw_text(screen, "ORANGE LEVEL COMPLETE", (10, 60), font, (255, 200, 150))
    if level_complete_green:
        draw_text(screen, "GREEN LEVEL COMPLETE", (10, 80), font, (150, 255, 150))

    # ----- FULL-WIDTH BANNER -----
    now = pygame.time.get_ticks() / 1000.0
    if orange_title != last_orange_title or green_title != last_green_title:
        last_orange_title = orange_title
        last_green_title = green_title
        banner_until_time = now + 1.0

    if now <= banner_until_time:
        banner = f"ORANGE: {orange_title}    |    GREEN: {green_title}"
        banner_surface = font.render(banner, True, COLOR_TEXT)
        bx = (SCREEN_WIDTH - banner_surface.get_width()) // 2
        by = SCREEN_HEIGHT - banner_surface.get_height() - 5
        screen.blit(banner_surface, (bx, by))

    pygame.display.flip()


# ==============================
# MAIN LOOP
# ==============================

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("10k Tree Bots — Dual Trees (Orange + Disabled Green, banner titles)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    # ORANGE ROOT
    root_links_orange = generate_synthetic_titles(0, MAX_LINKS_PER_LEVEL, tag="O")
    nodes_orange = []
    level0_orange = layout_level(root_links_orange, level_index=0)
    for n in level0_orange:
        n.children_titles = []
    nodes_orange.append(level0_orange)

    # GREEN ROOT
    root_links_green = generate_synthetic_titles(0, MAX_LINKS_PER_LEVEL, tag="G")
    nodes_green = []
    level0_green = layout_level(root_links_green, level_index=0)
    for n in level0_green:
        n.children_titles = []
    nodes_green.append(level0_green)

    # Bot groups
    bots_orange = initialize_bots(NUM_BOTS_PER_GROUP, start_level=0, disabled=False)
    bots_green = initialize_bots(NUM_BOTS_PER_GROUP, start_level=0, disabled=True)

    # Structures & links
    struct_orange = []
    struct_green = []
    links_orange = []
    links_green = []

    # State
    cur_lvl_orange = 0
    cur_lvl_green = 0
    tgt_orange = choose_target(nodes_orange, cur_lvl_orange)
    tgt_green = choose_target(nodes_green, cur_lvl_green)
    assign_bots_to_node(bots_orange, tgt_orange)
    assign_bots_to_node(bots_green, tgt_green)

    built_orange = 0
    built_green = 0
    pruned_orange = False
    pruned_green = False

    paused = False
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not running:
            break

        if paused:
            draw_scene(screen, font,
                       nodes_orange, nodes_green,
                       bots_orange, bots_green,
                       struct_orange, struct_green,
                       links_orange, links_green,
                       cur_lvl_orange, cur_lvl_green,
                       tgt_orange, tgt_green,
                       False, False,
                       built_orange, built_green)
            continue

        # Update bots
        for b in bots_orange:
            b.update(dt)
        for b in bots_green:
            b.update(dt)

        # ----- ORANGE TREE LOGIC -----
        level_complete_orange = False
        if not pruned_orange and any_bot_reached(bots_orange) and tgt_orange is not None:
            if cur_lvl_orange > 0 and len(nodes_orange[cur_lvl_orange - 1]) == 1:
                prev_node = nodes_orange[cur_lvl_orange - 1][0]
                links_orange.append(LinkSegment(
                    prev_node.x, prev_node.y, prev_node.level,
                    tgt_orange.x, tgt_orange.y, tgt_orange.level
                ))
            nodes_orange[cur_lvl_orange] = [tgt_orange]
            pruned_orange = True

        if all_bots_reached(bots_orange):
            level_complete_orange = True

            if built_orange < MAX_LEVELS - 1:
                base_level = cur_lvl_orange
                top_level = cur_lvl_orange + 1
                struct_orange.append(StructureSegment(
                    x=tgt_orange.x,
                    y=tgt_orange.y,
                    base_level=base_level,
                    top_level=top_level
                ))
                built_orange += 1

                if not tgt_orange.children_titles:
                    children = generate_synthetic_titles(cur_lvl_orange + 1, MAX_LINKS_PER_LEVEL, tag="O")
                    tgt_orange.children_titles = children
                    tgt_orange.expanded = True
                else:
                    children = tgt_orange.children_titles

                next_level_index = cur_lvl_orange + 1
                if next_level_index >= len(nodes_orange):
                    unique_titles = list(dict.fromkeys(children))[:MAX_LINKS_PER_LEVEL]
                    next_nodes = layout_level(unique_titles, next_level_index)
                    for n in next_nodes:
                        n.children_titles = []
                    nodes_orange.append(next_nodes)

                cur_lvl_orange = next_level_index

                jitter = PLANE_HALF * 0.1
                for b in bots_orange:
                    b.x = tgt_orange.x + random.uniform(-jitter, jitter)
                    b.y = tgt_orange.y + random.uniform(-jitter, jitter)
                    b.level = cur_lvl_orange
                    b.reached = False

                tgt_orange = choose_target(nodes_orange, cur_lvl_orange)
                assign_bots_to_node(bots_orange, tgt_orange)
                pruned_orange = False

        # ----- GREEN TREE LOGIC -----
        level_complete_green = False
        if not pruned_green and any_bot_reached(bots_green) and tgt_green is not None:
            if cur_lvl_green > 0 and len(nodes_green[cur_lvl_green - 1]) == 1:
                prev_node = nodes_green[cur_lvl_green - 1][0]
                links_green.append(LinkSegment(
                    prev_node.x, prev_node.y, prev_node.level,
                    tgt_green.x, tgt_green.y, tgt_green.level
                ))
            nodes_green[cur_lvl_green] = [tgt_green]
            pruned_green = True

        if all_bots_reached(bots_green):
            level_complete_green = True

            if built_green < MAX_LEVELS - 1:
                base_level = cur_lvl_green
                top_level = cur_lvl_green + 1
                struct_green.append(StructureSegment(
                    x=tgt_green.x,
                    y=tgt_green.y,
                    base_level=base_level,
                    top_level=top_level
                ))
                built_green += 1

                if not tgt_green.children_titles:
                    children = generate_synthetic_titles(cur_lvl_green + 1, MAX_LINKS_PER_LEVEL, tag="G")
                    tgt_green.children_titles = children
                    tgt_green.expanded = True
                else:
                    children = tgt_green.children_titles

                next_level_index = cur_lvl_green + 1
                if next_level_index >= len(nodes_green):
                    unique_titles = list(dict.fromkeys(children))[:MAX_LINKS_PER_LEVEL]
                    next_nodes = layout_level(unique_titles, next_level_index)
                    for n in next_nodes:
                        n.children_titles = []
                    nodes_green.append(next_nodes)

                cur_lvl_green = next_level_index

                jitter = PLANE_HALF * 0.1
                for b in bots_green:
                    b.x = tgt_green.x + random.uniform(-jitter, jitter)
                    b.y = tgt_green.y + random.uniform(-jitter, jitter)
                    b.level = cur_lvl_green
                    b.reached = False

                tgt_green = choose_target(nodes_green, cur_lvl_green)
                assign_bots_to_node(bots_green, tgt_green)
                pruned_green = False

        draw_scene(screen, font,
                   nodes_orange, nodes_green,
                   bots_orange, bots_green,
                   struct_orange, struct_green,
                   links_orange, links_green,
                   cur_lvl_orange, cur_lvl_green,
                   tgt_orange, tgt_green,
                   level_complete_orange, level_complete_green,
                   built_orange, built_green)

    pygame.quit()


if __name__ == "__main__":
    main()
