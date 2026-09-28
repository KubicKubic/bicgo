"""Validate the JAX environment against the NumPy reference implementation."""
import numpy as np
import jax
import jax.numpy as jnp

from bicgo import go_env as E
from bicgo.config import EnvConfig
from bicgo import reference as R


def _jax_state(cfg, board, to_play, ko, passes=0, move_num=0, done=False):
    n = cfg.board_size
    b = jnp.asarray(board, jnp.int8)[None]
    hist = jnp.broadcast_to(b, (1, cfg.history, n, n)).copy()
    return E.GoState(
        board=b,
        hist=hist,
        to_play=jnp.array([to_play], jnp.int8),
        ko=jnp.array([ko], jnp.int32),
        passes=jnp.array([passes], jnp.int32),
        move_num=jnp.array([move_num], jnp.int32),
        done=jnp.array([done]),
        winner=jnp.zeros(1, jnp.int8),
        black_area=jnp.zeros(1, jnp.int32),
        white_area=jnp.zeros(1, jnp.int32),
    )


def _assert_match(state, ref, cfg, tag=""):
    assert int(state.to_play[0]) == ref.to_play, f"{tag} to_play"
    assert int(state.ko[0]) == ref.ko, f"{tag} ko"
    assert int(state.passes[0]) == ref.passes, f"{tag} passes"
    assert bool(state.done[0]) == ref.done, f"{tag} done"
    assert np.array_equal(np.asarray(state.board[0]), ref.board), f"{tag} board"
    if ref.done:
        assert int(state.winner[0]) == ref.winner, f"{tag} winner"
        ba, wa, _ = R.area_score(ref.board, cfg.komi)
        assert int(state.black_area[0]) == ba
        assert int(state.white_area[0]) == wa


def test_random_playouts_match_reference():
    cfg = EnvConfig(board_size=7, history=2, max_moves=80)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        ref = R.RefGame(size=cfg.board_size, komi=cfg.komi, max_moves=cfg.max_moves)
        state = _jax_state(cfg, ref.board, ref.to_play, ref.ko)
        while not ref.done:
            action = int(rng.integers(0, cfg.board_size * cfg.board_size + 1))
            ref.step(action)
            state = E.step(state, jnp.array([action], jnp.int32), cfg)
            _assert_match(state, ref, cfg, tag=f"seed{seed} move{ref.move_num}")
            if ref.move_num > cfg.max_moves:
                break


def test_capture():
    cfg = EnvConfig(board_size=19, history=1)
    board = np.zeros((19, 19), np.int8)
    board[0, 0] = R.WHITE
    board[0, 1] = R.BLACK
    ref = R.RefGame(size=19, board=board, to_play=R.BLACK)
    state = _jax_state(cfg, board, R.BLACK, -1)
    action = 1 * 19 + 0  # black plays (1,0), capturing white (0,0)
    ref.step(action)
    state = E.step(state, jnp.array([action], jnp.int32), cfg)
    assert ref.board[0, 0] == R.EMPTY
    _assert_match(state, ref, cfg, "capture")


def test_suicide_illegal():
    cfg = EnvConfig(board_size=19, history=1)
    board = np.zeros((19, 19), np.int8)
    board[0, 1] = R.BLACK
    board[1, 0] = R.BLACK
    ref = R.RefGame(size=19, board=board, to_play=R.WHITE)
    state = _jax_state(cfg, board, R.WHITE, -1)
    action = 0  # white at (0,0) has no liberty -> illegal
    assert not ref.legal(action)
    ref.step(action)
    state = E.step(state, jnp.array([action], jnp.int32), cfg)
    assert ref.board[0, 0] == R.EMPTY
    assert ref.passes == 1
    _assert_match(state, ref, cfg, "suicide")


def test_ko_recapture_illegal():
    # canonical ko shape on 5x5:
    #   . X O .
    #   X O . O
    #   . X O .
    board = np.zeros((5, 5), np.int8)
    board[0, 1] = R.BLACK
    board[0, 2] = R.WHITE
    board[1, 0] = R.BLACK
    board[1, 1] = R.WHITE
    board[1, 3] = R.WHITE
    board[2, 1] = R.BLACK
    board[2, 2] = R.WHITE
    cfg = EnvConfig(board_size=5, history=1)
    ref = R.RefGame(size=5, board=board, to_play=R.BLACK)
    state = _jax_state(cfg, board, R.BLACK, -1)
    action = 1 * 5 + 2  # black captures the white stone at (1,1)
    assert ref.legal(action)
    ref.step(action)
    state = E.step(state, jnp.array([action], jnp.int32), cfg)
    _assert_match(state, ref, cfg, "ko-capture")
    assert ref.ko == 1 * 5 + 1
    assert int(state.ko[0]) == 1 * 5 + 1
    # white recapture at (1,1) must be illegal (ko)
    assert not ref.legal(1 * 5 + 1)
    ref.step(1 * 5 + 1)
    state = E.step(state, jnp.array([1 * 5 + 1], jnp.int32), cfg)
    _assert_match(state, ref, cfg, "ko-recapture")
    assert ref.passes == 1


def test_area_score_and_komi():
    board = np.zeros((9, 9), np.int8)
    # black owns the whole left column region; white owns right column region
    board[:, 4] = R.BLACK
    board[0, 0] = R.WHITE  # keep it simple, just check symmetry of scoring
    ba, wa, winner = R.area_score(board, 7.5)
    jba, jwa, jwin = E.score(jnp.asarray(board, jnp.int8), 7.5)
    assert (int(jba), int(jwa)) == (ba, wa)
    assert int(jwin) == winner


def test_two_passes_end():
    cfg = EnvConfig(board_size=9, history=1)
    ref = R.RefGame(size=9)
    state = _jax_state(cfg, ref.board, ref.to_play, ref.ko)
    for _ in range(2):
        ref.step(ref.pass_index)
        state = E.step(state, jnp.array([ref.pass_index], jnp.int32), cfg)
    assert ref.done and bool(state.done[0])
    _assert_match(state, ref, cfg, "two-pass")


def test_hard_cap_is_draw():
    """Hitting the long hard cap ends the game as a draw (reward 0)."""
    cfg = EnvConfig(board_size=9, history=1, max_moves=5)
    state = _jax_state(cfg, np.zeros((9, 9), np.int8), R.BLACK, -1)
    prev = state
    for action in (0, 1, 2, 3, 4):
        prev = state
        state = E.step(state, jnp.array([action], jnp.int32), cfg, score_now=False)
    just = state.done & ~prev.done
    assert bool(just[0]) and state.passes[0] < 2
    state = E.finalize_scores(state, just, cfg.komi)
    assert int(state.winner[0]) == R.DRAW
    assert float(E.terminal_reward(state, prev)[0]) == 0.0


def test_finalize_scores_and_reward():
    """The rollout's deferred scoring + terminal reward agree with the rules."""
    cfg = EnvConfig(board_size=9, history=1, komi=7.5)
    prev = _jax_state(cfg, np.zeros((9, 9), np.int8), R.BLACK, -1)
    prev = prev._replace(
        passes=jnp.array([1], jnp.int32), to_play=jnp.array([R.BLACK], jnp.int8)
    )
    nxt = E.step(prev, jnp.array([81], jnp.int32), cfg, score_now=False)  # pass
    assert bool(nxt.done[0])
    just = nxt.done & ~prev.done
    nxt = E.finalize_scores(nxt, just, cfg.komi)
    # empty board with komi 7.5 -> white wins, so black's terminal reward is -1
    assert int(nxt.winner[0]) == R.WHITE
    reward = E.terminal_reward(nxt, prev)
    assert float(reward[0]) == -1.0


def test_legal_mask_matches_reference():
    """JAX legal_mask must equal the reference legality oracle on random boards."""
    cfg = EnvConfig(board_size=7, history=1)
    rng = np.random.default_rng(1)
    for _ in range(30):
        board = rng.integers(0, 3, size=(7, 7)).astype(np.int8)
        colour = int(rng.integers(1, 3))
        ko = int(rng.choice([-1, rng.integers(0, 49)]))
        jm = np.asarray(
            E.legal_mask(jnp.asarray(board, jnp.int8), jnp.int32(ko), jnp.int8(colour))
        )
        ref = R.RefGame(size=7, board=board.copy(), to_play=colour, ko=ko)
        rm = np.array([ref.legal(i) for i in range(50)])
        assert np.array_equal(jm, rm), (board, colour, ko, jm, rm)

