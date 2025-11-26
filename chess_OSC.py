import pygame
import sys
import math
import random
import array
import socket
import struct

# =====================================================
# Chess + Bitwise Audio Grid
# Mega Drive – Slow, Dreamlike FM + DSL
# - Audio buffer size 4096
# - Sends OSC messages on chess moves to UDP port 9001
# - Improved OSC packing + console debug
# - WINDOWED by default (1280x720, resizable)
# =====================================================

BOARD_W = 8
BOARD_H = 8

TILE = 70
FPS = 60

BG_BASE = (8, 10, 25)
GRID = (80, 90, 140)
WHITE = (230, 230, 240)
COL_BLUE = (120, 200, 255)
COL_RED = (255, 130, 150)
HIGHLIGHT = (255, 240, 120)
VALID = (150, 255, 190)
AUDIO_STEP_GLOW = (90, 140, 220)

FONT_NAME = "arial"

BOARD_ORIGIN = (0, 0)

BIT_WIDTH = 8
BIT_MASK = (1 << BIT_WIDTH) - 1
BITWISE_OPS = ["AND", "OR", "XOR", "SHL", "SHR", "NOT"]

audio_grid = []
audio_registers = [0] * BOARD_H
audio_step_index = 0
audio_step_accum = 0.0
current_step_ms = 0
step_counter = 0
bar_index = 0

# --- OSC -------------------------------------------------

OSC_HOST = "127.0.0.1"
OSC_PORT = 9001
osc_sock = None

def osc_pad(b: bytes) -> bytes:
    while len(b) % 4 != 0:
        b += b'\0'
    return b

def send_osc_message(address: str, int_args):
    """Minimal OSC encoder: address + int arguments only."""
    global osc_sock
    if osc_sock is None:
        return
    try:
        # address
        addr_bin = address.encode("ascii") + b'\0'
        addr_bin = osc_pad(addr_bin)

        # type tags: ,iii...
        typetags = "," + ("i" * len(int_args))
        tt_bin = typetags.encode("ascii") + b'\0'
        tt_bin = osc_pad(tt_bin)

        # args
        args_bin = b"".join(struct.pack(">i", int(a)) for a in int_args)
        args_bin = osc_pad(args_bin)

        packet = addr_bin + tt_bin + args_bin
        osc_sock.sendto(packet, (OSC_HOST, OSC_PORT))

        # debug
        print(f"OSC -> {OSC_HOST}:{OSC_PORT} {address} {int_args}")
    except Exception as e:
        print("OSC send error:", e)

def init_osc():
    global osc_sock
    try:
        osc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"OSC socket initialised to {OSC_HOST}:{OSC_PORT}")
    except Exception as e:
        osc_sock = None
        print("Failed to init OSC socket:", e)

# --- Global musical / system params (DSL-tweakable) ---

tempo_base = 420
tempo_jitter = 40
audio_mutation_rate = 0.04
audio_trigger_prob = 0.75

BASE_MIDI = 50
SCALE_STEPS = [0, 2, 3, 5, 7, 8, 10, 12]
SCALE_MIDIS = []

CHORD_ROOT_INDEXES = [0, 5, 2, 3]

composition_sections = {}
composition_form = []
composition_timeline = []

bass_sounds = []
chord_sounds = []
lead_sounds = []

op_not_sounds = []
op_xor_sounds = []

voice_density = {
    "BASS": 0.9,
    "CHORD": 0.9,
    "LEAD": 0.8,
    "OP_CHIME": 0.8,
    "OP_GLITCH": 0.7,
}

soldier_move_mode = "ORTH"
commander_move_mode = "KING"

synth_params = {
    "BASS":       {"mod_index": 1.0, "square_mix": 0.10, "brightness": 1.0},
    "LEAD":       {"mod_index": 1.1, "square_mix": 0.18, "brightness": 1.0},
    "CHORD":      {"mod_index": 0.9, "square_mix": 0.14, "brightness": 1.0},
    "OP_CHIME":   {"mod_index": 1.3, "square_mix": 0.20, "brightness": 1.2},
    "OP_GLITCH":  {"mod_index": 1.6, "square_mix": 0.25, "brightness": 1.3},
}

dsl_initial_text = """# DSL – edit and click APPLY (button) to reprogram game+music
# Lines starting with # are comments.
#
# --- Music & timing ---
TEMPO base=420 jitter=40
AUDIO mutation=0.04 trigger=0.75
SCALE base=50 steps=0,2,3,5,7,8,10,12
#
# Basic chord roots (used if no sections/form defined)
CHORDS roots=0,5,2,3
#
# --- Higher-level composition ---
SECTION A roots=0,5,2,3
SECTION B roots=0,3,5,1
FORM seq=A,A,B,A
#
# --- Movement rules ---
MOVES SOLDIER=ORTH COMMANDER=KING
#
# --- Synthesis params per voice ---
SYN BASS mod=1.0 sq=0.10 bright=1.0
SYN LEAD mod=1.1 sq=0.18 bright=1.0
SYN CHORD mod=0.9 sq=0.14 bright=1.0
SYN OP_CHIME mod=1.3 sq=0.20 bright=1.2
SYN OP_GLITCH mod=1.6 sq=0.25 bright=1.3
#
# --- Voice densities ---
DENSITY BASS=0.9 CHORD=0.9 LEAD=0.8 OP_CHIME=0.8 OP_GLITCH=0.7
"""

dsl_lines = dsl_initial_text.splitlines()
dsl_status = "Click in the DSL pane to focus, then type. Click APPLY DSL to recompile."
dsl_button_rect = None
dsl_focus = True

caret_line = len(dsl_lines) - 1
caret_col = len(dsl_lines[caret_line]) if dsl_lines else 0
caret_visible = True
caret_timer = 0.0
CARET_BLINK_MS = 550

# -------------------- CHESS --------------------------

class Piece:
    def __init__(self, team, kind):
        self.team = team
        self.kind = kind

def empty_board():
    return [[None for _ in range(BOARD_W)] for _ in range(BOARD_H)]

def initial_board():
    b = empty_board()
    b[0][3] = Piece("BLUE", "COMMANDER")
    for x in range(BOARD_W):
        b[1][x] = Piece("BLUE", "SOLDIER")
    b[BOARD_H - 1][4] = Piece("RED", "COMMANDER")
    for x in range(BOARD_W):
        b[BOARD_H - 2][x] = Piece("RED", "SOLDIER")
    return b

def in_bounds(x, y):
    return 0 <= x < BOARD_W and 0 <= y < BOARD_H

def get_piece(board, pos):
    x, y = pos
    return board[y][x]

def set_piece(board, pos, p):
    x, y = pos
    board[y][x] = p

def directions_for_mode(mode):
    mode = mode.upper()
    if mode == "ORTH":
        return [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if mode == "DIAG":
        return [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    if mode in ("ORTH_DIAG", "KING"):
        return [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1)
        ]
    if mode == "KNIGHT":
        return [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
    return [(1, 0), (-1, 0), (0, 1), (0, -1)]

def valid_moves(board, pos):
    x, y = pos
    p = get_piece(board, pos)
    if not p:
        return []
    if p.kind == "SOLDIER":
        dirs = directions_for_mode(soldier_move_mode)
    else:
        dirs = directions_for_mode(commander_move_mode)

    moves = []
    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if not in_bounds(nx, ny):
            continue
        tgt = get_piece(board, (nx, ny))
        if tgt is None or tgt.team != p.team:
            moves.append((nx, ny))
    return moves

# -------------------- AUDIO GRID ---------------------

def random_mask():
    bits = random.randint(1, 3)
    v = 0
    for _ in range(bits):
        v |= 1 << random.randint(0, BIT_WIDTH - 1)
    return v & BIT_MASK

def init_audio_grid():
    global audio_grid, audio_registers, audio_step_index, audio_step_accum
    global current_step_ms, step_counter, bar_index
    audio_grid = []
    for y in range(BOARD_H):
        row = []
        for x in range(BOARD_W):
            if y < BOARD_H // 2:
                op = random.choice(["AND", "OR", "XOR"])
            else:
                op = random.choice(["SHL", "SHR", "NOT", "XOR"])
            row.append({"op": op, "mask": random_mask()})
        audio_grid.append(row)
    audio_registers = [random.randint(0, BIT_MASK) for _ in range(BOARD_H)]
    audio_step_index = 0
    audio_step_accum = 0.0
    current_step_ms = tempo_base
    step_counter = 0
    bar_index = 0

def rerandomise_audio_area(sx, sy):
    w = random.randint(2, 4)
    h = random.randint(2, 4)
    for y in range(sy, min(BOARD_H, sy + h)):
        for x in range(sx, min(BOARD_W, sx + w)):
            op = random.choice(BITWISE_OPS)
            audio_grid[y][x] = {"op": op, "mask": random_mask()}

def mutate_random_cell():
    if random.random() < audio_mutation_rate:
        x = random.randint(0, BOARD_W - 1)
        y = random.randint(0, BOARD_H - 1)
        audio_grid[y][x] = {"op": random.choice(BITWISE_OPS), "mask": random_mask()}

def apply_bit_op(v, op, m):
    v &= BIT_MASK
    if op == "AND":
        v &= m
    elif op == "OR":
        v |= m
    elif op == "XOR":
        v ^= m
    elif op == "SHL":
        v = (v << 1) & BIT_MASK
    elif op == "SHR":
        v = (v >> 1) & BIT_MASK
    elif op == "NOT":
        v = (~v) & BIT_MASK
    return v & BIT_MASK

def get_current_root_index():
    if composition_timeline:
        idx = composition_timeline[bar_index % len(composition_timeline)]
        return idx
    return CHORD_ROOT_INDEXES[bar_index % len(CHORD_ROOT_INDEXES)]

def advance_audio_step():
    global audio_step_index, audio_registers, current_step_ms
    global step_counter, bar_index
    x = audio_step_index
    current_root_idx = get_current_root_index()

    for y in range(BOARD_H):
        cell = audio_grid[y][x]
        v = apply_bit_op(audio_registers[y], cell["op"], cell["mask"])
        audio_registers[y] = v
        bit = (y + 2 * x) % BIT_WIDTH
        if (v & (1 << bit)) and random.random() < audio_trigger_prob:
            trigger_sound_for_row_and_op(y, v, x, current_root_idx, cell["op"])

    jitter = random.randint(-tempo_jitter, tempo_jitter)
    current_step_ms = max(150, tempo_base + jitter)
    mutate_random_cell()

    audio_step_index = (audio_step_index + 1) % BOARD_W
    step_counter += 1
    if step_counter % BOARD_W == 0:
        base_len = len(composition_timeline) if composition_timeline else len(CHORD_ROOT_INDEXES)
        base_len = max(1, base_len)
        bar_index = (bar_index + 1) % base_len

# -------------------- SOUND --------------------------

def midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def make_md_tone(freq, ms=650, volume=0.32, rate=22050,
                 voice_type="CHORD", brightness_override=None):
    total_time = ms / 1000.0
    count = int(rate * total_time)
    buf = array.array("h")

    vt = voice_type.upper()
    params = synth_params.get(vt, {"mod_index": 1.0, "square_mix": 0.15, "brightness": 1.0})
    base_mod_index = params["mod_index"]
    base_square_mix = params["square_mix"]
    base_brightness = params["brightness"]

    brightness = base_brightness if brightness_override is None else brightness_override
    mod_index = base_mod_index * brightness
    square_mix = base_square_mix * brightness

    if vt == "BASS":
        mod_ratio = 2.0
        chorus_detune = 0.003
    elif vt == "LEAD":
        mod_ratio = 2.7
        chorus_detune = 0.006
    elif vt == "OP_CHIME":
        mod_ratio = 3.0
        chorus_detune = 0.004
    elif vt == "OP_GLITCH":
        mod_ratio = 1.5
        chorus_detune = 0.008
    else:
        mod_ratio = 2.0
        chorus_detune = 0.004

    attack = 0.02
    release = 0.40

    for i in range(count):
        t = i / rate
        if t < attack:
            env = t / attack
        elif t > total_time - release:
            env = max(0.0, (total_time - t) / release)
        else:
            env = 1.0

        vib = 0.0018 * math.sin(2 * math.pi * 5.0 * t)
        f_base = freq * (1.0 + vib)

        mod = math.sin(2 * math.pi * f_base * mod_ratio * t) * mod_index
        fm1 = math.sin(2 * math.pi * f_base * t + mod)

        f_ch = f_base * (1.0 + chorus_detune)
        mod2 = math.sin(2 * math.pi * f_ch * mod_ratio * t) * (mod_index * 0.7)
        fm2 = math.sin(2 * math.pi * f_ch * t + mod2)

        combined = 0.65 * fm1 + 0.35 * fm2

        sq = 1.0 if math.sin(2 * math.pi * f_base * t) >= 0 else -1.0
        combined = (1.0 - square_mix) * combined + square_mix * sq

        shaped = math.tanh(0.75 * combined)

        raw_val = volume * env * 30000 * shaped
        crushed = int(raw_val / 4) * 4

        buf.append(int(crushed))

    return pygame.mixer.Sound(buffer=buf.tobytes())

def rebuild_scale_and_chords():
    global SCALE_MIDIS
    SCALE_MIDIS = [BASE_MIDI + s for s in SCALE_STEPS] + [
        BASE_MIDI + 12 + s for s in [0, 2, 3, 5, 7]
    ]

def rebuild_composition_timeline():
    global composition_timeline
    if not composition_sections or not composition_form:
        composition_timeline = []
        return
    timeline = []
    for sec_name in composition_form:
        roots = composition_sections.get(sec_name, [])
        timeline.extend(roots)
    composition_timeline = timeline

def init_sound():
    global bass_sounds, chord_sounds, lead_sounds, op_not_sounds, op_xor_sounds
    if not SCALE_MIDIS:
        rebuild_scale_and_chords()

    bass_midis = [m - 12 for m in SCALE_MIDIS]
    chord_midis = SCALE_MIDIS[:]
    lead_midis = SCALE_MIDIS[2:] + [SCALE_MIDIS[-1] + 2]

    bass_sounds = [make_md_tone(midi_to_freq(m), ms=900, volume=0.26, voice_type="BASS") for m in bass_midis]
    chord_sounds = [make_md_tone(midi_to_freq(m), ms=780, volume=0.24, voice_type="CHORD") for m in chord_midis]
    lead_sounds = [make_md_tone(midi_to_freq(m), ms=720, volume=0.22, voice_type="LEAD") for m in lead_midis]

    op_not_sounds = [
        make_md_tone(midi_to_freq(m), ms=520, volume=0.20, voice_type="OP_CHIME", brightness_override=None)
        for m in SCALE_MIDIS[4:]
    ]
    op_xor_sounds = [
        make_md_tone(midi_to_freq(m + 12), ms=380, volume=0.18, voice_type="OP_GLITCH", brightness_override=None)
        for m in SCALE_MIDIS[1:6]
    ]

def trigger_sound_for_row_and_op(row, reg, col, root_idx, op):
    density_bits = bin(reg).count("1")
    melodic_offset = (density_bits + col) % 4

    if row <= 2 and bass_sounds:
        if random.random() < voice_density.get("BASS", 1.0):
            idx = (root_idx + (row % 2)) % len(bass_sounds)
            bass_sounds[idx].play()
    elif 3 <= row <= 5 and chord_sounds:
        if random.random() < voice_density.get("CHORD", 1.0):
            idx = (root_idx + (row % 3)) % len(chord_sounds)
            chord_sounds[idx].play()
    elif row >= 6 and lead_sounds:
        if random.random() < voice_density.get("LEAD", 1.0):
            idx = (root_idx + melodic_offset) % len(lead_sounds)
            lead_sounds[idx].play()

    if op == "NOT" and op_not_sounds and random.random() < 0.6 * voice_density.get("OP_CHIME", 1.0):
        idx = (col + row) % len(op_not_sounds)
        op_not_sounds[idx].play()
    elif op == "XOR" and op_xor_sounds and random.random() < 0.5 * voice_density.get("OP_GLITCH", 1.0):
        idx = (density_bits + col) % len(op_xor_sounds)
        op_xor_sounds[idx].play()

# -------------------- DSL PARSING --------------------

def apply_dsl_from_lines():
    global tempo_base, tempo_jitter, audio_mutation_rate, audio_trigger_prob
    global BASE_MIDI, SCALE_STEPS, CHORD_ROOT_INDEXES, dsl_status
    global soldier_move_mode, commander_move_mode, synth_params
    global composition_sections, composition_form

    text = "\n".join(dsl_lines)
    composition_sections = {}
    composition_form = []

    lines = text.splitlines()
    try:
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            key = parts[0].upper()

            if key == "TEMPO":
                for token in parts[1:]:
                    if token.startswith("base="):
                        tempo_base = int(token.split("=", 1)[1])
                    elif token.startswith("jitter="):
                        tempo_jitter = int(token.split("=", 1)[1])

            elif key == "AUDIO":
                for token in parts[1:]:
                    if token.startswith("mutation="):
                        audio_mutation_rate = float(token.split("=", 1)[1])
                    elif token.startswith("trigger="):
                        audio_trigger_prob = float(token.split("=", 1)[1])

            elif key == "SCALE":
                for token in parts[1:]:
                    if token.startswith("base="):
                        BASE_MIDI = int(token.split("=", 1)[1])
                    elif token.startswith("steps="):
                        rest = token.split("=", 1)[1]
                        SCALE_STEPS = [int(x) for x in rest.split(",") if x]

            elif key == "CHORDS":
                for token in parts[1:]:
                    if token.startswith("roots="):
                        rest = token.split("=", 1)[1]
                        CHORD_ROOT_INDEXES = [int(x) for x in rest.split(",") if x]

            elif key == "SECTION":
                if len(parts) < 2:
                    continue
                sec_name = parts[1].strip().upper()
                roots = None
                for token in parts[2:]:
                    if token.startswith("roots="):
                        rest = token.split("=", 1)[1]
                        roots = [int(x) for x in rest.split(",") if x]
                if roots is not None:
                    composition_sections[sec_name] = roots

            elif key == "FORM":
                for token in parts[1:]:
                    if token.startswith("seq="):
                        rest = token.split("=", 1)[1]
                        composition_form = [x.strip().upper() for x in rest.split(",") if x.strip()]

            elif key == "MOVES":
                for token in parts[1:]:
                    t_up = token.upper()
                    if t_up.startswith("SOLDIER="):
                        soldier_move_mode = token.split("=", 1)[1].upper()
                    elif t_up.startswith("COMMANDER="):
                        commander_move_mode = token.split("=", 1)[1].upper()

            elif key == "SYN":
                if len(parts) < 2:
                    continue
                voice = parts[1].upper()
                if voice not in synth_params:
                    continue
                for token in parts[2:]:
                    if token.startswith("mod="):
                        synth_params[voice]["mod_index"] = float(token.split("=", 1)[1])
                    elif token.startswith("sq="):
                        synth_params[voice]["square_mix"] = float(token.split("=", 1)[1])
                    elif token.startswith("bright="):
                        synth_params[voice]["brightness"] = float(token.split("=", 1)[1])

            elif key == "DENSITY":
                for token in parts[1:]:
                    if "=" in token:
                        name, val = token.split("=", 1)
                        name = name.upper()
                        if name in voice_density:
                            voice_density[name] = float(val)

        rebuild_scale_and_chords()
        rebuild_composition_timeline()
        init_sound()
        dsl_status = "DSL applied successfully."
    except Exception as e:
        dsl_status = f"DSL error: {e}"

# -------------------- GEOMETRY & INPUT ---------------

def screen_to_board(mx, my):
    ox, oy = BOARD_ORIGIN
    if ox <= mx < ox + BOARD_W * TILE and oy <= my < oy + BOARD_H * TILE:
        return ((mx - ox) // TILE, (my - oy) // TILE)
    return None

def set_caret_from_click(mx, my, pane_rect, smallfont):
    global caret_line, caret_col
    margin_x = pane_rect.x + 8
    max_w = pane_rect.width - 16
    line_height = smallfont.get_linesize()
    text_top = pane_rect.y + 4 + smallfont.get_height() + 6 + 32 + 6
    text_bottom = pane_rect.bottom - smallfont.get_height() - 8

    if my < text_top:
        caret_line = 0
        caret_col = 0
        return

    y = text_top
    last_line_index = len(dsl_lines) - 1 if dsl_lines else 0

    for li, line in enumerate(dsl_lines):
        L = len(line)
        if L == 0:
            if y <= my < y + line_height:
                caret_line = li
                caret_col = 0
                return
            y += line_height + 2
            continue

        pos = 0
        while pos < L:
            segment_end = pos + 1
            last_good = pos
            while segment_end <= L:
                text = line[pos:segment_end]
                w, _ = smallfont.size(text)
                if w > max_w:
                    break
                last_good = segment_end
                segment_end += 1
            if last_good == pos:
                last_good = pos + 1
            segment = line[pos:last_good]
            w_seg, h_seg = smallfont.size(segment)

            if y <= my < y + h_seg:
                rel_x = mx - margin_x
                if rel_x <= 0:
                    caret_line = li
                    caret_col = pos
                    return
                best_col = pos
                best_dist = float("inf")
                for c in range(pos, last_good + 1):
                    cw, _ = smallfont.size(line[pos:c])
                    dist = abs(cw - rel_x)
                    if dist < best_dist:
                        best_dist = dist
                        best_col = c
                caret_line = li
                caret_col = best_col
                return

            y += h_seg + 2
            pos = last_good

            if y > text_bottom:
                break

    caret_line = last_line_index
    caret_col = len(dsl_lines[last_line_index]) if dsl_lines else 0

# -------------------- BACKGROUND ---------------------

def draw_radial_background(surface, t):
    w, h = surface.get_size()
    cx, cy = w // 2, h // 2
    max_r = int((w * w + h * h) ** 0.5) // 2
    for i in range(6):
        phase = t * 0.00012 + i * 0.7
        r = int(max_r * (0.35 + 0.25 * math.sin(phase)))
        alpha = 40 + int(20 * math.sin(phase * 1.3))
        c0 = BG_BASE[0] + int(9 * math.sin(phase * 2.1))
        c1 = BG_BASE[1] + int(14 * math.cos(phase * 1.7))
        c2 = BG_BASE[2] + int(18 * math.sin(phase * 1.2))
        c0 = max(0, min(255, c0))
        c1 = max(0, min(255, c1))
        c2 = max(0, min(255, c2))
        alpha = max(0, min(255, alpha))
        color = (c0, c1, c2)
        layer = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.circle(layer, (*color, alpha), (cx, cy), r)
        surface.blit(layer, (0, 0))

# -------------------- RENDERING ----------------------

def draw_board_and_dsl(screen, board, selected, moves,
                       font, smallfont, bigfont,
                       team, winner, turn,
                       sw, sh, elapsed, right_pane_x):
    global dsl_button_rect

    screen.fill(BG_BASE)
    draw_radial_background(screen, elapsed)

    title = bigfont.render("Chess + Bitwise Audio Grid (MD Dream FM + DSL)", True, WHITE)
    screen.blit(title, (sw // 2 - title.get_width() // 2, 10))

    msg = f"WINNER: {winner} – press R" if winner else f"Turn: {team} (#{turn})"
    st = font.render(msg, True, WHITE)
    screen.blit(st, (sw // 2 - st.get_width() // 2, 10 + title.get_height() + 4))

    info = smallfont.render(
        "Left: board & bitwise music grid. Right: DSL editor (click APPLY DSL). ESC quits.",
        True, WHITE
    )
    screen.blit(info, (sw // 2 - info.get_width() // 2,
                       10 + title.get_height() + st.get_height() + 8))

    board_left, board_top = BOARD_ORIGIN
    tw = BOARD_W * TILE
    th = BOARD_H * TILE
    outer = pygame.Rect(board_left, board_top, tw, th)
    pygame.draw.rect(screen, (40, 40, 80), outer, 2)

    glow = pygame.Rect(board_left + audio_step_index * TILE, board_top,
                       TILE, BOARD_H * TILE)
    layer = pygame.Surface(glow.size, pygame.SRCALPHA)
    layer.fill((*AUDIO_STEP_GLOW, 55))
    screen.blit(layer, glow.topleft)

    for y in range(BOARD_H):
        for x in range(BOARD_W):
            r = pygame.Rect(board_left + x * TILE, board_top + y * TILE, TILE, TILE)
            base = (30, 34, 65) if (x + y) % 2 == 0 else (48, 52, 88)
            pygame.draw.rect(screen, base, r, border_radius=8)
            pygame.draw.rect(screen, GRID, r, 1, border_radius=8)

            cell = audio_grid[y][x]
            sym = {
                "AND": "&", "OR": "|", "XOR": "^",
                "SHL": "<<", "SHR": ">>", "NOT": "~"
            }.get(cell["op"], "?")
            t = smallfont.render(sym, True, WHITE)
            screen.blit(t, (r.right - t.get_width() - 5,
                            r.bottom - t.get_height() - 4))

    if selected:
        sx, sy = selected
        r = pygame.Rect(board_left + sx * TILE, board_top + sy * TILE, TILE, TILE)
        pygame.draw.rect(screen, HIGHLIGHT, r, 4, border_radius=10)

    for mx, my in moves:
        r = pygame.Rect(board_left + mx * TILE, board_top + my * TILE, TILE, TILE)
        pygame.draw.rect(screen, VALID, r, 3, border_radius=10)

    for y in range(BOARD_H):
        for x in range(BOARD_W):
            p = board[y][x]
            if not p:
                continue
            r = pygame.Rect(board_left + x * TILE, board_top + y * TILE, TILE, TILE)
            cx, cy = r.center

            shadow = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
            pygame.draw.circle(shadow, (0, 0, 0, 130),
                               (TILE // 2, TILE // 2 + 4), TILE // 2 - 12)
            screen.blit(shadow, (r.left, r.top))

            col = COL_BLUE if p.team == "BLUE" else COL_RED
            pygame.draw.circle(screen, col, (cx, cy - 2), TILE // 2 - 12)
            pygame.draw.circle(screen, (10, 10, 20),
                               (cx, cy - 2), TILE // 2 - 12, 2)

            lab = "C" if p.kind == "COMMANDER" else "S"
            t = bigfont.render(lab, True, (0, 0, 0))
            screen.blit(t, t.get_rect(center=(cx, cy - 4)))

    pane_rect = pygame.Rect(right_pane_x + 10, 60,
                            screen.get_width() - right_pane_x - 20,
                            screen.get_height() - 80)
    pygame.draw.rect(screen, (20, 22, 40), pane_rect)
    pygame.draw.rect(screen, (90, 95, 150), pane_rect, 2)

    dsl_title = font.render("DSL Editor (click to focus)", True, WHITE)
    screen.blit(dsl_title, (pane_rect.x + 8, pane_rect.y + 4))

    button_w, button_h = 150, 32
    dsl_button_rect = pygame.Rect(pane_rect.right - button_w - 10,
                                  pane_rect.y + 4,
                                  button_w, button_h)
    pygame.draw.rect(screen, (70, 120, 190), dsl_button_rect, border_radius=8)
    pygame.draw.rect(screen, (190, 220, 255), dsl_button_rect, 2, border_radius=8)
    bt = smallfont.render("APPLY DSL", True, WHITE)
    screen.blit(bt, bt.get_rect(center=dsl_button_rect.center))

    status = smallfont.render(dsl_status, True, WHITE)
    screen.blit(status, (pane_rect.x + 8,
                         pane_rect.bottom - status.get_height() - 4))

    margin_x = pane_rect.x + 8
    max_w = pane_rect.width - 16
    line_height = smallfont.get_linesize()
    text_top = pane_rect.y + 4 + dsl_title.get_height() + 6 + button_h + 6
    text_bottom = pane_rect.bottom - status.get_height() - 8

    y = text_top
    caret_px = None
    caret_py = None

    for li, line in enumerate(dsl_lines):
        L = len(line)
        if L == 0:
            if y + line_height > text_bottom:
                break
            t_surf = smallfont.render("", True, WHITE)
            screen.blit(t_surf, (margin_x, y))
            if li == caret_line and caret_col == 0:
                caret_px = margin_x
                caret_py = y
            y += line_height + 2
            continue

        pos = 0
        while pos < L:
            segment_end = pos + 1
            last_good = pos
            while segment_end <= L:
                text_seg = line[pos:segment_end]
                w_seg, _ = smallfont.size(text_seg)
                if w_seg > max_w:
                    break
                last_good = segment_end
                segment_end += 1
            if last_good == pos:
                last_good = pos + 1
            segment = line[pos:last_good]
            w_seg, h_seg = smallfont.size(segment)

            if y + h_seg > text_bottom:
                break

            t_surf = smallfont.render(segment, True, WHITE)
            screen.blit(t_surf, (margin_x, y))

            if li == caret_line and caret_col >= pos and caret_col <= last_good:
                caret_px = margin_x + smallfont.size(line[pos:caret_col])[0]
                caret_py = y

            y += h_seg + 2
            pos = last_good

        if y > text_bottom:
            break

    if dsl_focus and caret_px is not None and caret_py is not None and caret_visible:
        caret_h = smallfont.get_height()
        pygame.draw.line(screen, (230, 230, 255),
                         (caret_px, caret_py),
                         (caret_px, caret_py + caret_h), 2)

# -------------------- MAIN ---------------------------

def main():
    global BOARD_ORIGIN, dsl_focus, caret_timer, caret_visible
    global caret_line, caret_col

    pygame.mixer.pre_init(22050, -16, 1, 4096)
    pygame.init()
    pygame.mixer.set_num_channels(32)

    init_osc()

    # WINDOWED (1280x720) instead of fullscreen
    sw, sh = 1280, 720
    screen = pygame.display.set_mode((sw, sh), pygame.RESIZABLE)
    pygame.display.set_caption("Chess + Bitwise Audio Grid (MD Dream FM + DSL + OSC)")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, 20)
    smallfont = pygame.font.SysFont(FONT_NAME, 16)
    bigfont = pygame.font.SysFont(FONT_NAME, 26, bold=True)

    left_pane_width = int(sw * 0.55)
    board_w_px = BOARD_W * TILE
    board_h_px = BOARD_H * TILE
    board_left = (left_pane_width - board_w_px) // 2
    board_top = (sh - board_h_px) // 2
    BOARD_ORIGIN = (board_left, board_top)

    right_pane_x = left_pane_width

    rebuild_scale_and_chords()
    rebuild_composition_timeline()
    init_audio_grid()
    init_sound()

    board = initial_board()
    team = "BLUE"
    selected = None
    moves = []
    winner = None
    turn = 1

    global audio_step_accum, current_step_ms
    audio_step_accum = 0.0
    current_step_ms = tempo_base

    running = True
    elapsed = 0.0

    while running:
        dt = clock.tick(FPS)
        elapsed += dt
        audio_step_accum += dt
        caret_timer += dt
        if caret_timer >= CARET_BLINK_MS:
            caret_timer = 0.0
            caret_visible = not caret_visible

        while audio_step_accum >= current_step_ms:
            audio_step_accum -= current_step_ms
            advance_audio_step()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                else:
                    if dsl_focus:
                        caret_visible = True
                        caret_timer = 0.0
                        line = dsl_lines[caret_line] if dsl_lines else ""
                        if e.key == pygame.K_BACKSPACE:
                            if caret_col > 0:
                                dsl_lines[caret_line] = line[:caret_col-1] + line[caret_col:]
                                caret_col -= 1
                            else:
                                if caret_line > 0:
                                    prev_len = len(dsl_lines[caret_line-1])
                                    dsl_lines[caret_line-1] += line
                                    del dsl_lines[caret_line]
                                    caret_line -= 1
                                    caret_col = prev_len
                        elif e.key == pygame.K_DELETE:
                            if caret_col < len(line):
                                dsl_lines[caret_line] = line[:caret_col] + line[caret_col+1:]
                            else:
                                if caret_line < len(dsl_lines)-1:
                                    dsl_lines[caret_line] += dsl_lines[caret_line+1]
                                    del dsl_lines[caret_line+1]
                        elif e.key == pygame.K_RETURN:
                            new_line = line[caret_col:]
                            dsl_lines[caret_line] = line[:caret_col]
                            dsl_lines.insert(caret_line+1, new_line)
                            caret_line += 1
                            caret_col = 0
                        elif e.key == pygame.K_LEFT:
                            if caret_col > 0:
                                caret_col -= 1
                            elif caret_line > 0:
                                caret_line -= 1
                                caret_col = len(dsl_lines[caret_line])
                        elif e.key == pygame.K_RIGHT:
                            if caret_col < len(line):
                                caret_col += 1
                            elif caret_line < len(dsl_lines)-1:
                                caret_line += 1
                                caret_col = 0
                        elif e.key == pygame.K_UP:
                            if caret_line > 0:
                                caret_line -= 1
                                caret_col = min(caret_col, len(dsl_lines[caret_line]))
                        elif e.key == pygame.K_DOWN:
                            if caret_line < len(dsl_lines)-1:
                                caret_line += 1
                                caret_col = min(caret_col, len(dsl_lines[caret_line]))
                        elif e.key == pygame.K_HOME:
                            caret_col = 0
                        elif e.key == pygame.K_END:
                            caret_col = len(line)
                        else:
                            if e.unicode and e.unicode.isprintable():
                                ch = e.unicode
                                dsl_lines[caret_line] = line[:caret_col] + ch + line[caret_col:]
                                caret_col += len(ch)
                    else:
                        if e.key == pygame.K_r:
                            board = initial_board()
                            team = "BLUE"
                            selected = None
                            moves = []
                            winner = None
                            turn = 1
                            init_audio_grid()
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = pygame.mouse.get_pos()
                if mx < right_pane_x:
                    dsl_focus = False
                    if winner:
                        continue
                    t = screen_to_board(mx, my)
                    if t is None:
                        selected = None
                        moves = []
                    else:
                        x, y = t
                        p = get_piece(board, (x, y))
                        if not selected:
                            if p and p.team == team:
                                selected = (x, y)
                                moves = valid_moves(board, (x, y))
                        else:
                            if t == selected:
                                selected = None
                                moves = []
                            elif p and p.team == team:
                                selected = (x, y)
                                moves = valid_moves(board, (x, y))
                            else:
                                if t in moves:
                                    fx, fy = selected
                                    mp = get_piece(board, selected)
                                    tp = get_piece(board, t)

                                    rerandomise_audio_area(fx, fy)

                                    if mp:
                                        team_val = 0 if mp.team == "BLUE" else 1
                                        kind_val = 0 if mp.kind == "SOLDIER" else 1
                                        capture_flag = 1 if tp is not None else 0
                                        send_osc_message("/move", [fx, fy, x, y, team_val, kind_val, capture_flag])

                                    if tp and tp.kind == "COMMANDER":
                                        winner = mp.team
                                    set_piece(board, selected, None)
                                    set_piece(board, t, mp)
                                    selected = None
                                    moves = []
                                    if not winner:
                                        team = "RED" if team == "BLUE" else "BLUE"
                                        turn += 1
                else:
                    dsl_focus = True
                    caret_visible = True
                    caret_timer = 0.0
                    pane_rect = pygame.Rect(right_pane_x + 10, 60,
                                            screen.get_width() - right_pane_x - 20,
                                            screen.get_height() - 80)
                    button_w, button_h = 150, 32
                    button_rect = pygame.Rect(pane_rect.right - button_w - 10,
                                              pane_rect.y + 4,
                                              button_w, button_h)
                    if button_rect.collidepoint(mx, my):
                        apply_dsl_from_lines()
                    else:
                        set_caret_from_click(mx, my, pane_rect, smallfont)

        draw_board_and_dsl(screen, board, selected, moves,
                           font, smallfont, bigfont,
                           team, winner, turn,
                           sw, sh, elapsed, right_pane_x)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
