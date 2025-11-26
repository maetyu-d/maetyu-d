import sys
import math
import pygame
from pygame.locals import QUIT, MOUSEBUTTONDOWN, KEYDOWN, K_r, K_ESCAPE

import chess
import chess.variant

from pythonosc.udp_client import SimpleUDPClient


# -----------------------------
# Configuration
# -----------------------------

BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
INFO_BAR_HEIGHT = 80
WINDOW_SIZE = (BOARD_SIZE, BOARD_SIZE + INFO_BAR_HEIGHT)

FPS = 60

# Colors
COLOR_BG = (10, 15, 25)
COLOR_LIGHT = (230, 220, 210)
COLOR_DARK = (60, 80, 100)
COLOR_HIGHLIGHT = (255, 210, 80)
COLOR_HIGHLIGHT_MOVE = (140, 200, 255)
COLOR_LAST_MOVE = (160, 255, 160)
COLOR_TEXT_MAIN = (230, 230, 240)
COLOR_TEXT_SUB = (170, 180, 200)
COLOR_EXPLOSION = (255, 120, 80)

# OSC
OSC_HOST = "127.0.0.1"
OSC_PORT = 9001


# -----------------------------
# Utility Functions
# -----------------------------

def board_coords_from_pixel(x, y):
    """Convert pixel coordinates to chess square, or None if out of board."""
    if x < 0 or y < 0 or x >= BOARD_SIZE or y >= BOARD_SIZE:
        return None
    col = x // SQUARE_SIZE  # file: 0..7
    row = y // SQUARE_SIZE  # row: 0..7 from top
    file = col
    rank = 7 - row  # rank 0 at bottom
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return chess.square(file, rank)
    return None


def square_to_screen(square):
    """Convert chess square index to top-left pixel position."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    row = 7 - rank  # row 0 at top
    col = file
    x = col * SQUARE_SIZE
    y = row * SQUARE_SIZE
    return x, y


def piece_unicode(piece):
    """Return a nice Unicode symbol for a chess piece."""
    if piece is None:
        return ""
    # python-chess: uppercase white, lowercase black
    mapping = {
        "P": "♙",
        "N": "♘",
        "B": "♗",
        "R": "♖",
        "Q": "♕",
        "K": "♔",
        "p": "♟",
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
        "k": "♚",
    }
    return mapping.get(piece.symbol(), "?")


def piece_name(piece):
    """Return a human-friendly name like 'white_queen' or 'black_pawn'."""
    if piece is None:
        return "none"
    color = "white" if piece.color == chess.WHITE else "black"
    type_map = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    tname = type_map.get(piece.piece_type, "unknown")
    return f"{color}_{tname}"


# -----------------------------
# Explosion Animation
# -----------------------------

class ExplosionManager:
    """Manages atomic explosion animations on the board."""

    def __init__(self):
        # Each explosion: {"squares": [...], "start_ms": int}
        self.explosions = []
        self.duration_ms = 500  # explosion life

    def trigger_explosion(self, center_square, now_ms):
        affected_squares = []

        c_file = chess.square_file(center_square)
        c_rank = chess.square_rank(center_square)

        # Center + neighbors (8 surrounding squares)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                f = c_file + df
                r = c_rank + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    affected_squares.append(chess.square(f, r))

        self.explosions.append({
            "squares": affected_squares,
            "start_ms": now_ms
        })

    def update_and_draw(self, surface, now_ms):
        still_active = []
        for exp in self.explosions:
            age = now_ms - exp["start_ms"]
            if age > self.duration_ms:
                continue

            t = age / self.duration_ms  # 0..1
            # Radius and alpha fade over time
            max_radius = SQUARE_SIZE * 0.7
            radius = int((0.3 + 0.7 * t) * max_radius)
            alpha = int(255 * (1.0 - t))

            for sq in exp["squares"]:
                x, y = square_to_screen(sq)
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                pygame.draw.circle(
                    overlay,
                    COLOR_EXPLOSION + (alpha,),
                    (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                    radius,
                    0
                )
                surface.blit(overlay, (x, y))

            still_active.append(exp)

        self.explosions = still_active


# -----------------------------
# Game Class
# -----------------------------

class AtomicChessGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Atomic Chess – OSC @ 9001")
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_pieces = pygame.font.SysFont("dejavusans", 56, bold=True)
        self.font_info = pygame.font.SysFont("dejavusans", 24)
        self.font_big = pygame.font.SysFont("dejavusans", 32, bold=True)

        # Chess board / logic
        self.board = chess.variant.AtomicBoard()
        self.selected_square = None
               self.legal_targets = []
        self.last_move = None
        self.game_over = False
        self.result_text = ""

        # Explosion manager
        self.explosions = ExplosionManager()

        # OSC
        self.osc_client = SimpleUDPClient(OSC_HOST, OSC_PORT)

    # -------------------------
    # OSC helpers
    # -------------------------

    def get_neighbor_pieces(self, center_square, radius=2):
        """
        Return list of strings 'square:color_type' for all pieces within
        Chebyshev distance <= radius of center_square (including center).
        """
        neighbors = []
        c_file = chess.square_file(center_square)
        c_rank = chess.square_rank(center_square)

        for df in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                f = c_file + df
                r = c_rank + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    sq = chess.square(f, r)
                    p = self.board.piece_at(sq)
                    if p:
                        sq_name = chess.square_name(sq)
                        pname = piece_name(p)
                        neighbors.append(f"{sq_name}:{pname}")
        return neighbors

    def send_move_osc(self, move, san_before, fen_after,
                      captured_piece_name, neighbor_list,
                      white_count, black_count):
        try:
            fullmove = self.board.fullmove_number
            uci = move.uci()
            from_sq_name = chess.square_name(move.from_square)
            to_sq_name = chess.square_name(move.to_square)

            args = [
                fullmove,
                uci,
                san_before,
                fen_after,
                from_sq_name,
                to_sq_name,
                captured_piece_name,
                white_count,
                black_count,
            ] + neighbor_list

            self.osc_client.send_message("/move", args)
        except Exception as e:
            print(f"OSC /move send error: {e}")

    def send_game_over_osc(self, result, outcome_str):
        try:
            self.osc_client.send_message(
                "/game_over",
                [result, outcome_str]
            )
        except Exception as e:
            print(f"OSC /game_over send error: {e}")

    # -------------------------
    # Drawing
    # -------------------------

    def draw_board(self):
        # Board background
        self.screen.fill(COLOR_BG)

        # Squares
        for rank in range(8):
            for file in range(8):
                row = 7 - rank  # row 0 at top
                col = file

                x = col * SQUARE_SIZE
                y = row * SQUARE_SIZE

                if (rank + file) % 2 == 0:
                    color = COLOR_LIGHT
                else:
                    color = COLOR_DARK

                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

        # Last move highlight
        if self.last_move is not None:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                x, y = square_to_screen(sq)
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                overlay.fill(COLOR_LAST_MOVE + (70,))
                self.screen.blit(overlay, (x, y))

        # Selected square highlight
        if self.selected_square is not None:
            x, y = square_to_screen(self.selected_square)
            pygame.draw.rect(
                self.screen,
                COLOR_HIGHLIGHT,
                (x, y, SQUARE_SIZE, SQUARE_SIZE),
                4
            )

        # Legal move dots
        for sq in self.legal_targets:
            x, y = square_to_screen(sq)
            center = (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)
            radius = SQUARE_SIZE // 8
            pygame.draw.circle(self.screen, COLOR_HIGHLIGHT_MOVE, center, radius)

        # Pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x, y = square_to_screen(square)
                text = piece_unicode(piece)
                if piece.color == chess.WHITE:
                    color = (245, 245, 250)
                    shadow = (30, 30, 40)
                else:
                    color = (30, 40, 50)
                    shadow = (220, 220, 230)

                # Shadow
                shadow_surf = self.font_pieces.render(text, True, shadow)
                self.screen.blit(shadow_surf, (x + 4, y + 4))

                # Main piece
                txt_surf = self.font_pieces.render(text, True, color)
                rect = txt_surf.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
                self.screen.blit(txt_surf, rect.topleft)

        # Explosions on top
        now_ms = pygame.time.get_ticks()
        self.explosions.update_and_draw(self.screen, now_ms)

    def draw_info_bar(self):
        bar_rect = (0, BOARD_SIZE, BOARD_SIZE, INFO_BAR_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_BG, bar_rect)

        turn = "White" if self.board.turn == chess.WHITE else "Black"
        if not self.game_over:
            main_text = f"Atomic Chess – {turn} to move"
        else:
            main_text = f"Game Over – {self.result_text}"

        # Main line
        main_surf = self.font_big.render(main_text, True, COLOR_TEXT_MAIN)
        self.screen.blit(main_surf, (20, BOARD_SIZE + 10))

        # Sub line
        if not self.game_over:
            sub = "Click a piece, then a target square. Press R to reset, Esc to quit."
        else:
            sub = "Press R to start a new game. Esc to quit."
        sub_surf = self.font_info.render(sub, True, COLOR_TEXT_SUB)
        self.screen.blit(sub_surf, (20, BOARD_SIZE + 45))

    # -------------------------
    # Input Handling
    # -------------------------

    def handle_click(self, pos):
        if self.game_over:
            return  # Ignore board input when game over

        x, y = pos
        square = board_coords_from_pixel(x, y)
        if square is None:
            return

        piece = self.board.piece_at(square)
        # Same-side selection or new selection
        if self.selected_square is None:
            # Select if there's a piece of the side to move
            if piece is not None and piece.color == self.board.turn:
                self.select_square(square)
        else:
            if square == self.selected_square:
                # Deselect
                self.selected_square = None
                self.legal_targets = []
                return

            # Try to move from selected_square to this one
            moved = self.try_make_move(self.selected_square, square)
            if not moved:
                # If click on another own piece, update selection instead
                if piece is not None and piece.color == self.board.turn:
                    self.select_square(square)
                else:
                    # Keep current selection if move invalid
                    pass

    def select_square(self, square):
        self.selected_square = square
        # Compute legal targets from this square
        self.legal_targets = [
            move.to_square for move in self.board.legal_moves
            if move.from_square == square
        ]

    def try_make_move(self, from_sq, to_sq):
        piece = self.board.piece_at(from_sq)
        if piece is None:
            return False

        # Simple promotion: always to queen when pawn reaches last rank
        promotion = None
        to_rank = chess.square_rank(to_sq)
        if piece.piece_type == chess.PAWN and (to_rank == 0 or to_rank == 7):
            promotion = chess.QUEEN

        move = chess.Move(from_sq, to_sq, promotion=promotion)
        if move not in self.board.legal_moves:
            return False

        # Capture info BEFORE push
        direct_captured_piece = self.board.piece_at(move.to_square)
        captured_name = piece_name(direct_captured_piece)

        # Explosion animation if capture
        is_capture = self.board.is_capture(move)
        if is_capture:
            now_ms = pygame.time.get_ticks()
            self.explosions.trigger_explosion(to_sq, now_ms)

        # SAN before push (python-chess requirement)
        san = self.board.san(move)

        # Make move
        self.board.push(move)
        fen_after = self.board.fen()
        self.last_move = move

        # Neighbor pieces within 2 squares of destination AFTER the move
        neighbor_list = self.get_neighbor_pieces(move.to_square, radius=2)

        # Count pieces on each side AFTER the move
        white_count = sum(1 for p in self.board.piece_map().values() if p.color == chess.WHITE)
        black_count = sum(1 for p in self.board.piece_map().values() if p.color == chess.BLACK)

        # Send OSC
        self.send_move_osc(
            move, san, fen_after,
            captured_name, neighbor_list,
            white_count, black_count
        )

        # Reset selection state
        self.selected_square = None
        self.legal_targets = []

        # Check game over
        if self.board.is_game_over():
            self.game_over = True
            result = self.board.result(claim_draw=True)
            outcome = self.board.outcome()
            if outcome is not None and outcome.termination.name:
                outcome_str = outcome.termination.name
            else:
                outcome_str = "UNKNOWN"

            self.result_text = f"{result} ({outcome_str})"
            self.send_game_over_osc(result, outcome_str)

        return True

    def reset_game(self):
        self.board = chess.variant.AtomicBoard()
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.game_over = False
        self.result_text = ""
        self.explosions = ExplosionManager()

    # -------------------------
    # Main Loop
    # -------------------------

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_r:
                        self.reset_game()

            self.draw_board()
            self.draw_info_bar()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    game = AtomicChessGame()
    game.run()
