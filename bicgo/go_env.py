"""A fully jit/vmap-able 19x19 Go environment in JAX.

Rule set (aligned with the HullQin weiqi implementation at game.hullqin.cn):

  * ``board`` is an int8 array with 0 = empty, 1 = black, 2 = white.
  * Black (1) moves first; players alternate.
  * A move is a flat board index ``0 .. N*N-1``; ``N*N`` is a pass.
  * Capturing, simple ko and the suicide prohibition follow the reference JS
    implementation (the ``dp``/``nT`` functions of chunk ``wq``):
      - occupied points are illegal,
      - a move may not leave its own group without liberties unless it captures,
      - the single-stone recapture (ko) is forbidden for one turn.
  * Two consecutive passes end the game (classic rule; the HullQin UI instead
    offers a "reject pass" button and never auto-ends, so this is the standard
    completion rule layered on top of their state transitions).
  * Final score is Chinese area scoring (stones + territory) with komi 7.5;
    black wins iff ``black_area - komi > white_area``.

Every public function accepts an optional leading batch dimension: pass a
single ``(N, N)`` board or a ``(B, N, N)`` batch.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

EMPTY = 0
BLACK = 1
WHITE = 2
DRAW = 3

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
class GoState(NamedTuple):
    """Batched game state; every array carries a leading ``(B,)`` axis."""

    board: jnp.ndarray       # (B, N, N) int8
    hist: jnp.ndarray        # (B, H, N, N) int8, hist[:, 0] is the latest board
    to_play: jnp.ndarray     # (B,) int8, BLACK or WHITE
    ko: jnp.ndarray          # (B,) int32, forbidden point or -1
    passes: jnp.ndarray      # (B,) int32, consecutive pass counter
    move_num: jnp.ndarray    # (B,) int32
    done: jnp.ndarray        # (B,) bool
    winner: jnp.ndarray      # (B,) int8, EMPTY until finished, else BLACK/WHITE/DRAW
    black_area: jnp.ndarray  # (B,) int32, filled at termination
    white_area: jnp.ndarray  # (B,) int32, filled at termination


# ---------------------------------------------------------------------------
# array helpers
# ---------------------------------------------------------------------------
def _shift(x: jnp.ndarray, dr: int, dc: int, fill: float = 0):
    """Shift the last two axes by ``(dr, dc)`` padding out-of-board with ``fill``.

    Use ``fill=-1`` when comparing board values so the border never matches.
    """
    n = x.shape[-1]
    pad = [(0, 0)] * (x.ndim - 2)
    pad += [(max(dr, 0), max(-dr, 0)), (max(dc, 0), max(-dc, 0))]
    xp = jnp.pad(x, pad, constant_values=fill)
    r0 = max(-dr, 0)
    c0 = max(-dc, 0)
    return xp[..., r0:r0 + n, c0:c0 + n]


def _neighbours(x: jnp.ndarray) -> jnp.ndarray:
    out = _shift(x, 1, 0)
    out = out | _shift(x, -1, 0)
    out = out | _shift(x, 0, 1)
    return out | _shift(x, 0, -1)


def _expand(seed: jnp.ndarray, allowed: jnp.ndarray) -> jnp.ndarray:
    """Flood-fill ``seed`` through ``allowed`` with a converging while_loop."""

    def cond(s):
        return jnp.any(allowed & _neighbours(s) & ~s)

    def body(s):
        return s | (allowed & _neighbours(s))

    return lax.while_loop(cond, body, seed)


# ---------------------------------------------------------------------------
# connected components and liberties
# ---------------------------------------------------------------------------
def component_labels(board: jnp.ndarray) -> jnp.ndarray:
    """Min-index connected-component labels over same-valued cells.

    Empty regions are labelled too (needed by the scorer). Converges in at most
    the component diameter iterations.
    """
    n = board.shape[-1]
    big = jnp.int32(n * n)
    idx = jnp.arange(n * n, dtype=jnp.int32).reshape(n, n)
    labels = jnp.broadcast_to(idx, board.shape).copy()
    b = board.astype(jnp.int32)

    def relax(labels):
        m = labels
        for dr, dc in _DIRS:
            nbl = _shift(labels, dr, dc, fill=big)
            nbc = _shift(b, dr, dc, fill=-1)
            m = jnp.minimum(m, jnp.where(nbc == b, nbl, big))
        return m

    def cond(labels):
        return jnp.any(relax(labels) != labels)

    return lax.while_loop(cond, relax, labels)


def _group_liberty_counts(board: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Distinct-liberty count per group, broadcast back onto every stone cell."""
    n = board.shape[-1]
    cells = n * n
    batch = board.reshape(-1, cells).shape[0]
    stone_labels = jnp.where(board != EMPTY, labels, -1)
    nb = [_shift(stone_labels, dr, dc, fill=-1).reshape(batch, cells) for dr, dc in _DIRS]
    empty_flat = (board == EMPTY).reshape(batch, cells)

    per_group = jnp.zeros((batch, cells), dtype=jnp.int32)
    batch_idx = jnp.arange(batch)[:, None]
    for i in range(4):
        # only empty cells contribute liberties; each distinct adjacent group once
        first = (nb[i] >= 0) & empty_flat
        for j in range(i):
            first = first & (nb[i] != nb[j])
        index = jnp.where(first, nb[i], 0)
        per_group = per_group.at[batch_idx, index].add(first.astype(jnp.int32))

    counts = jnp.take_along_axis(
        per_group, labels.reshape(batch, cells), axis=-1
    ).reshape(board.shape)
    return jnp.where(board != EMPTY, counts, 0)


# ---------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------
def legal_mask(board, ko, to_play, allow_suicide: bool = False) -> jnp.ndarray:
    """Boolean ``(..., N*N+1)`` mask of legal moves (last entry is pass).

    Set ``allow_suicide=True`` to also mark self-capture points as legal.
    """
    if board.ndim == 2:
        board = board[None]
        squeeze = True
    else:
        squeeze = False
    n = board.shape[-1]
    b = board.astype(jnp.int32)
    c = jnp.broadcast_to(jnp.asarray(to_play).astype(jnp.int32), (b.shape[0],))
    c = c[:, None, None]
    opp = 3 - c

    labels = component_labels(b)
    libs = _group_liberty_counts(b, labels)
    lib1 = (b != EMPTY) & (libs == 1)
    lib2 = (b != EMPTY) & (libs >= 2)

    empty = b == EMPTY
    legal = empty & (
        _neighbours(empty)
        | _neighbours((b == c) & lib2)
        | _neighbours((b == opp) & lib1)
        | allow_suicide
    )

    ko_b = jnp.broadcast_to(jnp.asarray(ko).astype(jnp.int32), (b.shape[0],))
    grid = jnp.arange(n * n).reshape(n, n)[None]
    legal = legal & (grid != ko_b[:, None, None])

    legal = legal.reshape(b.shape[0], n * n)
    legal = jnp.concatenate([legal, jnp.ones((b.shape[0], 1), dtype=bool)], axis=-1)
    return legal[0] if squeeze else legal


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------
def _group_alive(board: jnp.ndarray) -> jnp.ndarray:
    """True for every stone cell whose own-colour group has a liberty."""
    lib_seed = _neighbours(board == EMPTY)
    alive = jnp.zeros_like(board, dtype=bool)
    for colour in (BLACK, WHITE):
        m = board == colour
        alive = alive | _expand(m & lib_seed, m)
    return alive


def _group_from(board: jnp.ndarray, pos, colour) -> jnp.ndarray:
    n = board.shape[-1]
    seed = jnp.zeros((n * n,), dtype=bool).at[pos].set(True).reshape(n, n)
    return _expand(seed & (board == colour), board == colour)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score(board, komi: float):
    """Chinese area score -> ``(black_area, white_area, winner)``."""
    if board.ndim == 2:
        board = board[None]
        squeeze = True
    else:
        squeeze = False
    b = board.astype(jnp.int32)
    empty = b == EMPTY
    black = b == BLACK
    white = b == WHITE

    touch_black = (empty & _neighbours(black)).reshape(b.shape[0], -1)
    touch_white = (empty & _neighbours(white)).reshape(b.shape[0], -1)
    labels = component_labels(b).reshape(b.shape[0], -1)
    batch = jnp.arange(b.shape[0])[:, None]

    reg_black = jnp.zeros_like(touch_black, dtype=jnp.int32)
    reg_white = jnp.zeros_like(touch_white, dtype=jnp.int32)
    reg_black = reg_black.at[batch, labels].max(touch_black.astype(jnp.int32))
    reg_white = reg_white.at[batch, labels].max(touch_white.astype(jnp.int32))

    cell_black = jnp.take_along_axis(reg_black, labels, axis=-1).reshape(b.shape) > 0
    cell_white = jnp.take_along_axis(reg_white, labels, axis=-1).reshape(b.shape) > 0

    terr_black = empty & cell_black & ~cell_white
    terr_white = empty & cell_white & ~cell_black

    black_area = black.sum(axis=(-2, -1)) + terr_black.sum(axis=(-2, -1))
    white_area = white.sum(axis=(-2, -1)) + terr_white.sum(axis=(-2, -1))
    winner = jnp.where(black_area - white_area - komi > 0, BLACK, WHITE).astype(jnp.int8)

    if squeeze:
        return black_area[0], white_area[0], winner[0]
    return black_area, white_area, winner


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------
def _step_single(board, hist, ko, passes, move_num, done, to_play, action, cfg, score_now=True):
    n = board.shape[-1]
    colour = to_play
    opp = 3 - colour
    is_pass = action >= n * n

    pos = jnp.minimum(action, n * n - 1)
    r = pos // n
    c = pos % n
    placed = board.at[r, c].set(colour)

    alive_placed = _group_alive(placed)
    enemy_dead = (placed == opp) & ~alive_placed
    captured_count = enemy_dead.sum()
    board_after = jnp.where(enemy_dead, EMPTY, placed)

    # A capture always grants a liberty, so if anything was captured the new
    # stone is alive; otherwise reuse the liberty computed on the placed board.
    own_alive = (captured_count > 0) | alive_placed[r, c]
    own_group = _group_from(board_after, pos, colour)
    own_size = own_group.sum()
    own_liberties = ((board_after == EMPTY) & _neighbours(own_group)).sum()
    suicide = ~own_alive

    if cfg.allow_suicide:
        # self-capture: the suicided group is removed from the board
        alive_after = _group_alive(board_after)
        self_dead = own_group & ~alive_after
        board_after = jnp.where(self_dead, EMPTY, board_after)

    # simple ko: one stone captured and the capturing stone is a lone stone
    # whose only liberty is the captured point
    ko_here = (captured_count == 1) & (own_size == 1) & (own_liberties == 1)
    ko_pos = jnp.argmax(enemy_dead.reshape(-1)).astype(jnp.int32)

    # occupied points and the ko point are illegal; illegal moves are treated
    # as passes so the transition stays total (the policy masks them anyway).
    # Suicide is forbidden unless cfg.allow_suicide is set.
    occupied = board[r, c] != EMPTY
    illegal = occupied | (pos == ko)
    do_pass = is_pass | (suicide & (not cfg.allow_suicide)) | illegal
    board_new = jnp.where(do_pass, board, board_after)
    passes_new = jnp.where(do_pass, passes + 1, jnp.int32(0))
    ko_new = jnp.where(
        do_pass, jnp.int32(-1), jnp.where(ko_here, ko_pos, jnp.int32(-1))
    )

    # Two consecutive passes -> area score.  Reaching the long hard cap
    # (max_moves) is a DRAW, not an area-score truncation.
    finished_pass = cfg.two_pass_end & (passes_new >= 2)
    move_num_new = move_num + 1
    finished_cap = move_num_new >= cfg.max_moves
    finished = finished_pass | finished_cap

    if score_now:
        black_area, white_area, area_winner = lax.cond(
            finished_pass,
            lambda b: score(b, cfg.komi),
            lambda b: (jnp.int32(0), jnp.int32(0), jnp.int8(EMPTY)),
            board_new,
        )
        winner = jnp.where(
            finished_pass, area_winner,
            jnp.where(finished_cap, jnp.int8(DRAW), jnp.int8(EMPTY)),
        )
    else:
        black_area = jnp.int32(0)
        white_area = jnp.int32(0)
        winner = jnp.int8(EMPTY)

    hist_new = jnp.concatenate([board_new[None], hist[:-1]], axis=0)
    to_play_new = (3 - colour).astype(jnp.int8)

    # already-finished games are frozen
    keep = done
    board_new = jnp.where(keep, board, board_new)
    hist_new = jnp.where(keep, hist, hist_new)
    passes_new = jnp.where(keep, passes, passes_new)
    ko_new = jnp.where(keep, ko, ko_new)
    move_num_new = jnp.where(keep, move_num, move_num_new)
    winner = jnp.where(keep, jnp.int8(EMPTY), winner)
    finished = jnp.where(keep, done, finished)
    to_play_new = jnp.where(keep, to_play, to_play_new).astype(jnp.int8)

    return GoState(
        board=board_new,
        hist=hist_new,
        to_play=to_play_new,
        ko=ko_new,
        passes=passes_new,
        move_num=move_num_new,
        done=finished,
        winner=winner,
        black_area=black_area,
        white_area=white_area,
    )


def step(state: GoState, action: jnp.ndarray, cfg, score_now: bool = True) -> GoState:
    """Batched transition; ``action`` has shape ``(B,)``.

    ``cfg`` may be an :class:`EnvConfig` or any object exposing ``.env``.
    Set ``score_now=False`` to skip the (expensive) area scoring inside the
    step and run it once per rollout with :func:`finalize_scores` instead.
    """
    env = getattr(cfg, "env", cfg)
    return jax.vmap(lambda s, a: _step_single(
        s.board, s.hist, s.ko, s.passes, s.move_num, s.done, s.to_play, a, env, score_now
    ))(state, action)


def finalize_scores(state: GoState, just_finished: jnp.ndarray, komi: float) -> GoState:
    """Compute area scores for a batch, but only when some game just ended.

    The ``lax.cond`` predicate is a scalar over the whole batch, so the
    connected-component scoring loop is skipped on the (common) steps where no
    game terminates.
    """
    def do(board):
        return score(board, komi)

    def no(board):
        b = board.shape[0]
        z = jnp.zeros((b,), jnp.int32)
        return z, z, jnp.full((b,), EMPTY, jnp.int8)

    black_area, white_area, area_winner = lax.cond(
        jnp.any(just_finished), do, no, state.board
    )
    # area score only for two-pass endings; hard-cap endings are draws
    area_winner = jnp.where(state.passes >= 2, area_winner, jnp.int8(DRAW))
    winner = jnp.where(just_finished, area_winner, jnp.int8(EMPTY))
    return state._replace(
        black_area=black_area, white_area=white_area, winner=winner
    )


# ---------------------------------------------------------------------------
# reset / reward
# ---------------------------------------------------------------------------
def reset(batch_size: int, cfg) -> GoState:
    cfg = getattr(cfg, "env", cfg)
    n = cfg.board_size
    zeros_i = jnp.zeros((batch_size,), dtype=jnp.int32)
    return GoState(
        board=jnp.zeros((batch_size, n, n), dtype=jnp.int8),
        hist=jnp.zeros((batch_size, cfg.history, n, n), dtype=jnp.int8),
        to_play=jnp.full((batch_size,), BLACK, dtype=jnp.int8),
        ko=jnp.full((batch_size,), -1, dtype=jnp.int32),
        passes=zeros_i,
        move_num=zeros_i,
        done=jnp.zeros((batch_size,), dtype=bool),
        winner=jnp.zeros((batch_size,), dtype=jnp.int8),
        black_area=zeros_i,
        white_area=zeros_i,
    )


def terminal_reward(state: GoState, prev_state: GoState) -> jnp.ndarray:
    """Terminal reward for the mover, from the mover's own perspective."""
    just_finished = state.done & ~prev_state.done
    mover = prev_state.to_play
    won = state.winner == mover
    lost = (state.winner != mover) & (state.winner != EMPTY) & (state.winner != DRAW)
    r = jnp.where(won, 1.0, jnp.where(lost, -1.0, 0.0))
    return jnp.where(just_finished, r, 0.0)


def outcome_for_player(state: GoState, player) -> jnp.ndarray:
    """Final result ``+1/-1/0`` from ``player``'s perspective (draw = 0)."""
    return jnp.where(
        state.winner == player, 1.0,
        jnp.where((state.winner == EMPTY) | (state.winner == DRAW), 0.0, -1.0),
    )
