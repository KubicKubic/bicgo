"""Plain-NumPy reference implementation of the bicgo rules.

This is deliberately written with simple BFS/loops so it can act as an
independent oracle for the JAX environment (see ``tests/test_go_rules.py``).
It also provides a tiny human-playable terminal driver.

Rules: black=1 moves first, white=2, 0 empty; capturing; simple ko; suicide
illegal; pass = N*N; two consecutive passes end the game; Chinese area scoring
with komi 7.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EMPTY, BLACK, WHITE, DRAW = 0, 1, 2, 3


def neighbours(n: int, idx: int):
    r, c = divmod(idx, n)
    if r > 0:
        yield idx - n
    if r < n - 1:
        yield idx + n
    if c > 0:
        yield idx - 1
    if c < n - 1:
        yield idx + 1


def _group_and_liberties(flat, n, start):
    """BFS a monochromatic group; return (cells, liberty_cells)."""
    colour = flat[start]
    seen = {start}
    stack = [start]
    liberties = set()
    while stack:
        cur = stack.pop()
        for nb in neighbours(n, cur):
            if flat[nb] == EMPTY:
                liberties.add(nb)
            elif flat[nb] == colour and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen, liberties


def is_legal(board, colour, idx, ko) -> bool:
    n = board.shape[0]
    if idx >= n * n or idx < 0:
        return True  # pass
    if idx == ko or board.reshape(-1)[idx] != EMPTY:
        return False
    trial = board.copy()
    trial.reshape(-1)[idx] = colour
    opp = 3 - colour
    captured = 0
    for nb in neighbours(n, idx):
        r, c = divmod(nb, n)
        if trial[r, c] == opp:
            cells, libs = _group_and_liberties(trial.reshape(-1), n, nb)
            if not libs:
                captured += len(cells)
                for x in cells:
                    trial.reshape(-1)[x] = EMPTY
    _, own_libs = _group_and_liberties(trial.reshape(-1), n, idx)
    return bool(own_libs)


def play(board, colour, idx):
    """Return ``(new_board, captured_cells)``; assumes the move is legal."""
    n = board.shape[0]
    new = board.copy()
    flat = new.reshape(-1)
    flat[idx] = colour
    opp = 3 - colour
    captured = []
    for nb in neighbours(n, idx):
        if flat[nb] == opp:
            cells, libs = _group_and_liberties(flat, n, nb)
            if not libs:
                captured.extend(cells)
                for x in cells:
                    flat[x] = EMPTY
    return new, captured


def ko_point(board_after, colour, idx, captured):
    """Simple-ko forbidden point after a move (or -1)."""
    n = board_after.shape[0]
    if len(captured) != 1:
        return -1
    cells, libs = _group_and_liberties(board_after.reshape(-1), n, idx)
    if len(cells) == 1 and len(libs) == 1:
        return captured[0]
    return -1


def area_score(board, komi: float = 7.5):
    """Chinese area scoring -> ``(black_area, white_area, winner)``."""
    n = board.shape[0]
    flat = board.reshape(-1)
    visited = np.zeros(n * n, dtype=bool)
    black_area = int((flat == BLACK).sum())
    white_area = int((flat == WHITE).sum())
    for start in range(n * n):
        if flat[start] != EMPTY or visited[start]:
            continue
        region = []
        stack = [start]
        visited[start] = True
        borders = set()
        while stack:
            cur = stack.pop()
            region.append(cur)
            for nb in neighbours(n, cur):
                if flat[nb] == EMPTY:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
                else:
                    borders.add(flat[nb])
        if borders == {BLACK}:
            black_area += len(region)
        elif borders == {WHITE}:
            white_area += len(region)
    winner = BLACK if black_area - white_area - komi > 0 else WHITE
    return black_area, white_area, winner


@dataclass
class RefGame:
    size: int = 19
    komi: float = 7.5
    board: np.ndarray = field(default=None)
    to_play: int = BLACK
    ko: int = -1
    passes: int = 0
    move_num: int = 0
    done: bool = False
    winner: int = 0
    max_moves: int = 4 * 19 * 19  # hard cap -> DRAW

    def __post_init__(self):
        if self.board is None:
            self.board = np.zeros((self.size, self.size), dtype=np.int8)

    @property
    def pass_index(self) -> int:
        return self.size * self.size

    def legal(self, idx: int) -> bool:
        return is_legal(self.board, self.to_play, idx, self.ko)

    def step(self, idx: int):
        assert not self.done
        n = self.size
        if idx >= n * n or not is_legal(self.board, self.to_play, idx, self.ko):
            self.passes += 1
            self.ko = -1
        else:
            new_board, captured = play(self.board, self.to_play, idx)
            self.ko = ko_point(new_board, self.to_play, idx, captured)
            self.board = new_board
            self.passes = 0
        self.move_num += 1
        self.to_play = 3 - self.to_play
        if self.passes >= 2:
            self.done = True
            _, _, self.winner = area_score(self.board, self.komi)
        elif self.move_num >= self.max_moves:
            self.done = True
            self.winner = DRAW
        return self

    def legal_moves(self):
        return [i for i in range(self.size * self.size + 1) if self.legal(i)]
