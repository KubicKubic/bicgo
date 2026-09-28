"""Minimal SGF import/export (no variations, no handicap).

Move indices are ``0..N*N-1`` with ``N*N`` meaning pass, matching the env.
"""
from __future__ import annotations

import re

LETTERS = "abcdefghijklmnopqrs"  # SGF: columns a..s (19), rows a..s top to bottom

# Human-facing Go coordinates skip the letter I.
GO_COLS = "ABCDEFGHJKLMNOPQRST"


def index_to_sgf(idx: int, size: int) -> str:
    if idx >= size * size:
        return ""
    r, c = divmod(idx, size)
    return LETTERS[c] + LETTERS[r]


def sgf_to_index(text: str, size: int) -> int:
    if len(text) != 2:
        return size * size
    c = LETTERS.index(text[0])
    r = LETTERS.index(text[1])
    return r * size + c


def moves_to_sgf(moves, size: int, komi: float = 7.5, start: int = 1) -> str:
    parts = [f"(;GM[1]FF[4]CA[UTF-8]SZ[{size}]KM[{komi}]"]
    colour = start
    for idx in moves:
        parts.append(f";{'B' if colour == 1 else 'W'}[{index_to_sgf(idx, size)}]")
        colour = 3 - colour
    parts.append(")")
    return "".join(parts)


def sgf_to_game(text: str):
    """Return ``(size, komi, [(colour, index), ...])``."""
    size, komi = 19, 7.5
    m = re.search(r"SZ\[(\d+)\]", text)
    if m:
        size = int(m.group(1))
    m = re.search(r"KM\[([0-9.]+)\]", text)
    if m:
        komi = float(m.group(1))
    moves = []
    for mm in re.finditer(r";([BW])\[([a-z]*)\]", text):
        colour = 1 if mm.group(1) == "B" else 2
        s = mm.group(2)
        moves.append((colour, size * size if s == "" else sgf_to_index(s, size)))
    return size, komi, moves


def vertex_to_index(text: str, size: int) -> int | None:
    """Parse a human coordinate like ``D4`` or ``pass`` into an action index."""
    t = text.strip().upper()
    if t in ("PASS", "P"):
        return size * size
    m = re.fullmatch(r"([A-HJ-T])(\d{1,2})", t)
    if not m or int(m.group(2)) < 1 or int(m.group(2)) > size:
        return None
    col = GO_COLS.index(m.group(1))
    row = size - int(m.group(2))  # row 1 is the bottom
    return row * size + col


def index_to_vertex(idx: int, size: int) -> str:
    if idx >= size * size:
        return "pass"
    r, c = divmod(idx, size)
    return f"{GO_COLS[c]}{size - r}"
