# Third-party notices

## HullQin Go rules (`game.hullqin.cn`)

The rule semantics implemented in `bicgo/go_env.py` (black-first alternation,
capture, simple ko, the suicide prohibition, pass as the last index, and
Chinese area scoring with komi 7.5) were derived from the behaviour of the
public HullQin weiqi game (webpack chunk `wq`, rules module `9568`, state enum
`8601`). No JavaScript source was copied; the rules were independently
re-implemented in JAX and are cross-checked against a NumPy oracle in
`bicgo/reference.py`.

The reference implementation's own automatic win/loss and its "reject pass"
interaction are deliberately not reproduced: bicgo uses the classic
two-consecutive-pass termination and exact Chinese area scoring. See the
"Rules" section of `README.md` for the exact deviations.
