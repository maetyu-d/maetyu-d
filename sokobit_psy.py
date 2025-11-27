
"""
Sokubit
----------------------------------------------
"""

import math
import random
from dataclasses import dataclass, field

import pygame

# -------- Optional audio backend (sounddevice) --------
try:
    import sounddevice as sd
    HAS_SD = True
except Exception:
    sd = None
    HAS_SD = False


# -------- Utility functions --------

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, v))


def blend(c1, c2, a):
    """Blend two RGB colours, a in [0,1] toward c2."""
    return tuple(int((1 - a) * c1[i] + a * c2[i]) for i in range(3))


def cycle_color(rgb, t, strength=0.4):
    """Temporal neon wobble around a base colour."""
    r, g, b = rgb
    r = clamp(int(r + strength * 80 * math.sin(t * 0.9 + 0.0)))
    g = clamp(int(g + strength * 80 * math.sin(t * 0.9 + 2.1)))
    b = clamp(int(b + strength * 80 * math.sin(t * 0.9 + 4.2)))
    return (r, g, b)


# -------- Opcodes & bit helpers --------

OPCODES = {
    "NOP": 0, "XOR": 1, "AND": 2, "OR": 3, "NOT": 4,
    "SHL": 5, "SHR": 6, "ADD": 7, "SUB": 8, "ROL": 9,
    "ROR": 10, "MOV": 11, "LOADC": 12, "NOISE": 13,
    "STEP": 14, "MUL": 15,
}

OPCODE_COLORS = {
    0:  (30, 30, 40),   # NOP
    1:  (255, 40, 255), # XOR
    2:  (60, 255, 150), # AND
    3:  (60, 220, 255), # OR
    4:  (250, 250, 250),# NOT
    5:  (255, 180, 60), # SHL
    6:  (255, 230, 80), # SHR
    7:  (100, 140, 255),# ADD
    8:  (60, 170, 230), # SUB
    9:  (200, 80, 255), # ROL
    10: (255, 120, 220),# ROR
    11: (130, 230, 255),# MOV
    12: (255, 210, 140),# LOADC
    13: (255, 70, 70),  # NOISE
    14: (170, 255, 90), # STEP
    15: (255, 90, 140), # MUL
}


def rol16(x, n=1):
    x &= 0xFFFF
    n &= 15
    return ((x << n) | (x >> (16 - n))) & 0xFFFF


def ror16(x, n=1):
    x &= 0xFFFF
    n &= 15
    return ((x >> n) | (x << (16 - n))) & 0xFFFF


# -------- BitGrid program --------

@dataclass
class BitGridProgram:
    grid: list = field(default_factory=lambda: [[0] * 16 for _ in range(16)])

    @classmethod
    def empty(cls):
        return cls()


# -------- Pentatonic for "beautiful" engine --------

PENT = [0, 3, 5, 7, 10]
ROOT_MIDI = 52  # E3-ish


def midi_to_inc(midi, sr):
    freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    return freq / sr  # cycles per sample


# -------- Hybrid audio VM (nasty <-> beautiful) --------

@dataclass
class BitGridVM:
    program: BitGridProgram
    hardware_sample_rate: int = 44100

    def __post_init__(self):
        self.registers = [0] * 16
        # base pitch/freq
        self.registers[1] = 180
        self.registers[2] = 230
        self.registers[10] = 90
        # timbre seeds
        self.registers[3] = 0xAAAA
        self.registers[4] = 0x0F0F
        self.registers[5] = 0x3333
        # meta
        self.registers[6] = 32
        self.registers[7] = 128
        self.registers[11] = 0x4000
        self.registers[12] = 0x8000
        self.registers[14] = 0x8000  # morph mid

        self.registers[0] = 0

        self.lfsr = 0xACE1

        self.seq_interval = 256
        self.mod_interval = 64
        self.chaos_interval = 512

        self._seq = 0
        self._mod = 0
        self._chaos = 0

        self.phase_scale = 1.0
        self._filter_state = 0.0
        self._filter_alpha = 0.2
        self._am_depth = 0.3
        self.morph = 0.5

        # engine phases
        self.n1 = self.n2 = self.n3 = 0.0
        self.b1 = self.b2 = self.b3 = 0.0

    def _noise16(self):
        bit = ((self.lfsr >> 0) ^ (self.lfsr >> 2) ^ (self.lfsr >> 3) ^ (self.lfsr >> 5)) & 1
        self.lfsr = (self.lfsr >> 1) | (bit << 15)
        return self.lfsr & 0xFFFF

    def run_row(self, r):
        g = self.program.grid
        regs = self.registers
        acc = regs[r] & 0xFFFF
        for c in range(16):
            op = g[r][c]
            src = regs[c] & 0xFFFF
            if op == 0:
                continue
            elif op == 1:
                acc ^= src
            elif op == 2:
                acc &= src
            elif op == 3:
                acc |= src
            elif op == 4:
                acc = (~acc) & 0xFFFF
            elif op == 5:
                acc = (acc << 1) & 0xFFFF
            elif op == 6:
                acc = (acc >> 1) & 0xFFFF
            elif op == 7:
                acc = (acc + src) & 0xFFFF
            elif op == 8:
                acc = (acc - src) & 0xFFFF
            elif op == 9:
                acc = rol16(acc, src & 3)
            elif op == 10:
                acc = ror16(acc, src & 3)
            elif op == 11:
                acc = src
            elif op == 12:
                acc = ((r << 4) | c) & 0xFFFF
            elif op == 13:
                acc ^= self._noise16()
            elif op == 14:
                regs[0] = (regs[0] + (src & 0xF) + 1) & 0xF
            elif op == 15:
                acc = ((acc * (src & 0xFF)) >> 4) & 0xFFFF
        regs[r] = acc

    def update_meta(self):
        r = self.registers
        raw = r[6] & 0xFF
        self.seq_interval = max(4, min(8192, 4 + raw * 32))
        rawm = r[7] & 0xFF
        self.mod_interval = max(1, min(2048, 1 + rawm * 8))
        rawc = r[11] & 0xFF
        self.chaos_interval = max(8, min(4096, 8 + rawc * 16))
        self.phase_scale = 0.2 + (rawm / 255.0) * 3.0
        rawf = r[12] & 0xFF
        self._filter_alpha = 0.02 + (rawf / 255.0) * 0.45
        self._am_depth = (rawc / 255.0) * 0.6
        self.morph = (r[14] & 0xFF) / 255.0

    def nasty(self):
        r = self.registers
        inc1 = (r[1] & 0xFF) * self.phase_scale / 65536.0
        inc2 = (r[2] & 0xFF) * self.phase_scale * 0.75 / 65536.0
        inc3 = (r[10] & 0xFF) * self.phase_scale * 0.5 / 65536.0
        self.n1 = (self.n1 + inc1) % 1.0
        self.n2 = (self.n2 + inc2) % 1.0
        self.n3 = (self.n3 + inc3) % 1.0
        pha = int(self.n1 * 256) & 0xFF
        phb = int(self.n2 * 256) & 0xFF
        phc = int(self.n3 * 256) & 0xFF
        w1 = pha ^ ((r[3] >> 8) & 0xFF)
        w2 = phb & ((r[4] >> 8) & 0xFF)
        tri = phc if phc < 128 else 255 - phc
        w3 = (tri + ((r[5] >> 8) & 0x3F)) & 0xFF
        mix = (w1 + w2 + w3) & 0xFF
        if (r[3] & 0x8000) and (self.lfsr & 0xF) == 0:
            mix ^= (self._noise16() >> 8) & 0xFF
        amsrc = (r[2] >> 8) & 0xFF
        am = 1.0 + self._am_depth * ((amsrc / 127.5) - 1.0)
        val = (mix - 128) / 128.0
        val *= am
        fm = ((r[10] >> 8) & 0xFF) - 128
        val += fm / 1024.0
        return math.tanh(val * 1.2)

    def beaut(self):
        r = self.registers
        sr = self.hardware_sample_rate

        def q(val):
            deg = val & 0xF
            st = PENT[deg % len(PENT)]
            octv = (val >> 4) & 0x3
            return ROOT_MIDI + st + 12 * octv

        m1 = q(r[1])
        m2 = q(r[2])
        m3 = q(r[10]) - 12
        inc1 = midi_to_inc(m1, sr) * self.phase_scale
        inc2 = midi_to_inc(m2, sr) * self.phase_scale * 0.99
        inc3 = midi_to_inc(m3, sr) * self.phase_scale * 0.5
        self.b1 = (self.b1 + inc1) % 1.0
        self.b2 = (self.b2 + inc2) % 1.0
        self.b3 = (self.b3 + inc3) % 1.0
        ph1 = self.b1
        ph2 = self.b2
        ph3 = self.b3
        s1 = math.sin(2 * math.pi * ph1)
        s2 = math.sin(2 * math.pi * (ph2 + 0.01))
        tri = 2 * abs(2 * (ph3 - math.floor(ph3 + 0.5))) - 1.0
        fm_amt = ((r[3] >> 8) & 0x3F) / 4096.0
        s1f = math.sin(2 * math.pi * (ph1 + fm_amt * tri))
        mix = 0.55 * s1f + 0.3 * s2 + 0.25 * tri
        amsrc = ((r[2] >> 8) & 0xFF) / 255.0
        mix *= 1.0 + self._am_depth * 0.5 * (amsrc - 0.5) * 2.0
        noise = ((self._noise16() & 0xFF) / 127.5 - 1.0) * 0.02
        mix += noise
        return math.tanh(mix * 1.1)

    def tick(self):
        r = self.registers
        if self._seq <= 0:
            self.run_row(r[0] & 0xF)
            self._seq = self.seq_interval
        self._seq -= 1

        if self._mod <= 0:
            self.run_row((r[0] + 5) & 0xF)
            self._mod = self.mod_interval
        self._mod -= 1

        if self._chaos <= 0:
            row = (self._noise16() >> 12) & 0xF
            self.run_row(row)
            jitter = (self._noise16() & 0xFF) - 128
            fac = 1.0 + jitter / 256.0
            self._chaos = max(4, int(self.chaos_interval * max(0.25, min(2.0, fac))))
        self._chaos -= 1

    def next_sample(self):
        self.tick()
        self.update_meta()
        n = self.nasty()
        b = self.beaut()
        m = self.morph
        out = (1.0 - m) * n + m * b
        out = math.tanh(out * 1.5) / 1.2
        return max(-1.0, min(1.0, out))


# -------- Live audio wrapper --------

class LiveAudio:
    def __init__(self, vm: BitGridVM):
        self.vm = vm
        self.stream = None

    def start(self):
        if not HAS_SD or self.stream is not None:
            return

        def cb(outdata, frames, time, status):
            if status:
                print("Audio status:", status)
            for i in range(frames):
                outdata[i, 0] = self.vm.next_sample()

        self.stream = sd.OutputStream(
            samplerate=self.vm.hardware_sample_rate,
            channels=1,
            callback=cb,
            blocksize=256,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# -------- Procedural Sokoban (full rules) --------

# Tiles:
# ' ' floor
# '#' wall
# '.' goal
# '$' box
# '*' box on goal
# '@' player
# '+' player on goal

def make_sokoban_level():
    """
    Procedurally generate a moderately complex 16x16 Sokoban board.
    - Border walls around edges.
    - Several random rectangular wall blocks in the interior.
    - 3–5 goals, 3–5 boxes, 1 player, all on floor.
    """
    rows, cols = 16, 16
    board = [["#"] * cols for _ in range(rows)]

    # Empty interior
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            board[r][c] = " "

    # Random wall blocks
    num_blocks = random.randint(4, 7)
    for _ in range(num_blocks):
        h = random.randint(2, 4)
        w = random.randint(3, 6)
        r0 = random.randint(2, rows - h - 2)
        c0 = random.randint(2, cols - w - 2)
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                # leave some central-ish space sometimes
                if 6 <= r <= 9 and 6 <= c <= 9 and random.random() < 0.4:
                    continue
                board[r][c] = "#"

    floor_cells = [
        (r, c)
        for r in range(1, rows - 1)
        for c in range(1, cols - 1)
        if board[r][c] == " "
    ]

    if len(floor_cells) < 20:
        return make_sokoban_level()

    # Goals
    num_goals = random.randint(3, 5)
    goals = random.sample(floor_cells, num_goals)
    for (r, c) in goals:
        board[r][c] = "."
    floor_cells = [(r, c) for (r, c) in floor_cells if (r, c) not in goals]

    # Boxes
    num_boxes = num_goals
    boxes = random.sample(floor_cells, num_boxes)
    for (r, c) in boxes:
        board[r][c] = "$"
    floor_cells = [(r, c) for (r, c) in floor_cells if (r, c) not in boxes]

    # Player
    def is_reasonable_player_spot(rr, cc):
        if board[rr][cc] != " ":
            return False
        free_nbrs = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r2, c2 = rr + dr, cc + dc
            if 0 <= r2 < rows and 0 <= c2 < cols and board[r2][c2] in (" ", ".", "$"):
                free_nbrs += 1
        return free_nbrs >= 2

    random.shuffle(floor_cells)
    player_pos = None
    for (rr, cc) in floor_cells:
        if is_reasonable_player_spot(rr, cc):
            player_pos = (rr, cc)
            break
    if player_pos is None:
        player_pos = floor_cells[0]
    pr, pc = player_pos
    board[pr][pc] = "@"

    return board


def find_player(board):
    for r in range(16):
        for c in range(16):
            if board[r][c] in ("@", "+"):
                return r, c
    return 1, 1


def move_player(board, pr, pc, dr, dc):
    """
    Full Sokoban move rules (walls, boxes, goals, box-on-goal).
    Returns (moved, new_pr, new_pc, changed_cells)
    """
    rows, cols = 16, 16
    nr, nc = pr + dr, pc + dc
    if not (0 <= nr < rows and 0 <= nc < cols):
        return False, pr, pc, []

    dest = board[nr][nc]
    changed = []

    def set_tile(rr, cc, ch):
        if board[rr][cc] != ch:
            board[rr][cc] = ch
            changed.append((rr, cc))

    player_on_goal = (board[pr][pc] == "+")

    if dest == "#":
        return False, pr, pc, []

    if dest in (" ", "."):
        set_tile(pr, pc, "." if player_on_goal else " ")
        if dest == ".":
            set_tile(nr, nc, "+")
        else:
            set_tile(nr, nc, "@")
        return True, nr, nc, changed

    if dest in ("$", "*"):
        br, bc = nr + dr, nc + dc
        if not (0 <= br < rows and 0 <= bc < cols):
            return False, pr, pc, []
        beyond = board[br][bc]
        if beyond in (" ", "."):
            # Move box
            if dest == "$":
                set_tile(nr, nc, "+" if beyond == "." else "@")
            else:  # '*'
                set_tile(nr, nc, "+" if beyond == "." else "@")
            if beyond == ".":
                set_tile(br, bc, "*")
            else:
                set_tile(br, bc, "$")
            set_tile(pr, pc, "." if player_on_goal else " ")
            return True, nr, nc, changed

    return False, pr, pc, []


# -------- Coupling: Sokoban → BitGrid mutations + pulses --------

def randomise_area_from(program: BitGridProgram, pulses, r, c, max_pulse=40):
    rows, cols = 16, 16
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    r_end = min(rows, r + h)
    c_end = min(cols, c + w)
    vals = list(OPCODES.values())
    for rr in range(r, r_end):
        for cc in range(c, c_end):
            program.grid[rr][cc] = random.choice(vals)
            pulses[rr][cc] = max_pulse


def apply_sokoban_changes(program: BitGridProgram, pulses, changed_cells):
    for (r, c) in changed_cells:
        randomise_area_from(program, pulses, r, c)


# -------- Example musical program --------

def example_program():
    p = BitGridProgram.empty()
    g = p.grid

    # Row 0: drive STEP + NOISE
    for c in range(16):
        g[0][c] = OPCODES["STEP"] if c % 2 == 0 else OPCODES["NOISE"]

    # Rows 1-2: arithmetic and bit ops
    for c in range(16):
        if c % 4 == 0:
            g[1][c] = OPCODES["ADD"]
            g[2][c] = OPCODES["ADD"]
        elif c % 4 == 1:
            g[1][c] = OPCODES["SUB"]
            g[2][c] = OPCODES["XOR"]
        elif c % 4 == 2:
            g[1][c] = OPCODES["ROL"]
            g[2][c] = OPCODES["AND"]
        else:
            g[1][c] = OPCODES["NOP"]
            g[2][c] = OPCODES["NOP"]

    # Rows 3-5: timbral noise & rotations
    for c in range(16):
        g[3][c] = OPCODES["ROL"] if c % 3 == 0 else OPCODES["XOR"]
        g[4][c] = OPCODES["NOISE"] if c in (0, 5, 9, 13) else OPCODES["AND"]
        g[5][c] = OPCODES["MUL"] if c % 2 else OPCODES["ADD"]

    # Meta rows affecting intervals / morph
    for c in range(16):
        g[6][c] = OPCODES["ADD"] if c % 3 else OPCODES["NOISE"]
        g[7][c] = OPCODES["ADD"] if c % 2 else OPCODES["SUB"]
        g[10][c] = OPCODES["XOR"] if c % 2 else OPCODES["ADD"]
        g[11][c] = OPCODES["ADD"] if c % 5 else OPCODES["NOISE"]
        g[12][c] = OPCODES["ADD"] if c % 2 else OPCODES["SUB"]
        g[14][c] = OPCODES["ADD"] if c % 3 else OPCODES["SUB"]

    return p


# -------- Pygame fullscreen app (fixed camera, pseudo-shader background) --------

def run_fullscreen_app():
    pygame.init()

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    info = pygame.display.Info()
    width, height = info.current_w, info.current_h

    clock = pygame.time.Clock()
    t = 0.0  # global time

    program = example_program()
    vm = BitGridVM(program)
    audio = LiveAudio(vm)
    audio.start()

    board = make_sokoban_level()
    pr, pc = find_player(board)

    pulses = [[0] * 16 for _ in range(16)]
    max_pulse = 40

    grid_size = int(min(width, height) * 0.9)
    cell = grid_size / 16.0
    offset_x = (width - grid_size) / 2.0
    offset_y = (height - grid_size) / 2.0

    # surfaces
    trail_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    # pseudo-shader background: low-res field scaled up
    bg_w = max(160, width // 8)
    bg_h = max(90, height // 8)
    bg_surface = pygame.Surface((bg_w, bg_h))

    BG_COLOR      = (0, 0, 10)
    WALL_COLOR    = (10, 10, 40)
    FLOOR_COLOR   = (3, 3, 20)
    GOAL_COLOR    = (80, 90, 200)
    BOX_COLOR     = (200, 130, 80)
    BOX_GOAL_COLOR= (230, 200, 120)
    PLAYER_COLOR  = (180, 240, 255)
    PLAYER_GOAL   = (210, 255, 255)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        t += dt

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    dr, dc = 0, 0
                    if event.key == pygame.K_UP:
                        dr, dc = -1, 0
                    elif event.key == pygame.K_DOWN:
                        dr, dc = 1, 0
                    elif event.key == pygame.K_LEFT:
                        dr, dc = 0, -1
                    elif event.key == pygame.K_RIGHT:
                        dr, dc = 0, 1
                    if dr != 0 or dc != 0:
                        moved, pr_new, pc_new, changed = move_player(board, pr, pc, dr, dc)
                        if moved:
                            pr, pc = pr_new, pc_new
                            apply_sokoban_changes(program, pulses, changed)

        # Decay pulses
        for r in range(16):
            for c in range(16):
                if pulses[r][c] > 0:
                    pulses[r][c] -= 1

        # Fade trails slightly
        trail_surface.fill((0, 0, 0, 40), special_flags=pygame.BLEND_RGBA_SUB)

        # ---- Pseudo-shader background (plasma-style, domain-warped) ----
        for y in range(bg_h):
            v = (y / bg_h) * 2.0 - 1.0
            for x in range(bg_w):
                u = (x / bg_w) * 2.0 - 1.0

                dx = u
                dy = v
                r = math.sqrt(dx * dx + dy * dy) + 1e-6
                ang = math.atan2(dy, dx)

                # domain-warped multi-sine field
                w1 = math.sin(3.0 * r - t * 1.2 + 2.0 * math.sin(ang * 2.0 + t * 0.7))
                w2 = math.sin(4.0 * (u + v) + t * 1.5)
                w3 = math.sin(6.0 * (u - v) - t * 1.1)
                p = (w1 + w2 + w3) / 3.0

                cr = 0.5 + 0.5 * math.sin(p * 3.0 + t * 0.9)
                cg = 0.5 + 0.5 * math.sin(p * 3.0 + t * 1.1 + 2.1 + 0.7 * math.sin(ang * 3.0 + t * 0.5))
                cb = 0.5 + 0.5 * math.sin(p * 3.0 + t * 0.8 + 4.2 + 0.9 * math.cos(r * 5.0 - t * 0.4))

                # radial vignette
                vig = math.exp(-r * 1.4)
                cr *= vig
                cg *= vig
                cb *= vig

                r_i = clamp(int(cr * 255), 0, 255)
                g_i = clamp(int(cg * 255), 0, 255)
                b_i = clamp(int(cb * 255), 0, 255)
                bg_surface.set_at((x, y), (r_i, g_i, b_i))

        bg_scaled = pygame.transform.smoothscale(bg_surface, (width, height))
        screen.blit(bg_scaled, (0, 0))

        global_glow = 0.5 * (math.sin(t * 0.4) + 1.0)  # 0..1

        # ---- Draw fixed grid ----
        for r in range(16):
            for c in range(16):
                ch = board[r][c]
                x = int(offset_x + c * cell)
                y = int(offset_y + r * cell)
                rect = pygame.Rect(x, y, int(cell) + 1, int(cell) + 1)

                # Sokoban base colour
                if ch == "#":
                    base_color = WALL_COLOR
                elif ch in (" ", "@", "+", "$", "*"):
                    base_color = FLOOR_COLOR
                elif ch == ".":
                    base_color = GOAL_COLOR
                else:
                    base_color = FLOOR_COLOR

                opcode_val = program.grid[r][c]
                op_color = OPCODE_COLORS.get(opcode_val, (80, 80, 80))
                op_color_dyn = cycle_color(op_color, t + 0.15 * r + 0.23 * c, strength=0.6)

                pf = pulses[r][c] / float(max_pulse) if pulses[r][c] > 0 else 0.0

                wobble_phase = r * 0.5 + c * 0.9
                wobble = 1.0 + 0.15 * math.sin(t * 1.8 + wobble_phase)

                local_glow = pf
                glow_mix = 0.4 * global_glow + 0.6 * local_glow

                blend_a = 0.25 + 0.4 * pf + 0.25 * global_glow
                blend_a = max(0.0, min(1.0, blend_a))
                base_blended = blend(base_color, op_color_dyn, blend_a)

                highlight = (255, 255, 255)
                final_color = blend(base_blended, highlight, glow_mix * 0.35)

                screen.fill(final_color, rect)

                cx = x + cell / 2.0
                cy = y + cell / 2.0
                radius = int(cell * 0.28 * wobble * (1.0 + 0.3 * pf))

                # soft additive aura on trail surface
                aura_alpha = clamp(int(80 + 100 * glow_mix), 0, 255)
                aura_color = (*final_color, aura_alpha)
                pygame.draw.circle(
                    trail_surface,
                    aura_color,
                    (int(cx), int(cy)),
                    int(radius * 1.2),
                )

                # Goal halo
                if ch in (".", "+", "*"):
                    ring_radius = int(radius * 1.05)
                    ring_color = blend(GOAL_COLOR, (255, 255, 255), 0.4 + 0.3 * pf)
                    pygame.draw.circle(
                        screen, ring_color, (int(cx), int(cy)), ring_radius, width=2
                    )

                # Box
                if ch in ("$", "*"):
                    inset = cell * 0.18 * (1.0 - 0.2 * (pf + global_glow))
                    inset = max(0.0, min(cell * 0.4, inset))
                    box_rect = pygame.Rect(
                        int(x + inset),
                        int(y + inset),
                        int(cell - 2 * inset),
                        int(cell - 2 * inset),
                    )
                    box_col = BOX_COLOR if ch == "$" else BOX_GOAL_COLOR
                    pygame.draw.rect(screen, box_col, box_rect, border_radius=int(cell * 0.18))
                    inner_inset = inset + cell * 0.04
                    inner_rect = pygame.Rect(
                        int(x + inner_inset),
                        int(y + inner_inset),
                        int(cell - 2 * inner_inset),
                        int(cell - 2 * inner_inset),
                    )
                    inner_col = blend(box_col, (255, 255, 255), 0.35 + 0.25 * pf)
                    pygame.draw.rect(screen, inner_col, inner_rect, border_radius=int(cell * 0.15))

                # Player orb
                if ch in ("@", "+"):
                    player_col = PLAYER_COLOR if ch == "@" else PLAYER_GOAL
                    outer_r = int(radius * 1.05)
                    mid_r = int(radius * 0.75)
                    inner_r = int(radius * 0.45)

                    pygame.draw.circle(screen, player_col, (int(cx), int(cy)), outer_r)
                    mid_col = blend(player_col, (255, 255, 255), 0.4 + 0.2 * global_glow)
                    pygame.draw.circle(screen, mid_col, (int(cx), int(cy)), mid_r)
                    core_color = blend(player_col, (255, 255, 255), 0.6 + 0.3 * pf)
                    pygame.draw.circle(screen, core_color, (int(cx), int(cy)), inner_r)

        # Composite trails on top with additive blending
        screen.blit(trail_surface, (0, 0), special_flags=pygame.BLEND_ADD)

        pygame.display.flip()

    audio.stop()
    pygame.quit()


if __name__ == "__main__":
    run_fullscreen_app()
