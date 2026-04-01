# Chal

**Chal** is a complete, FIDE-rules-compliant chess engine in **999 lines of C99** it has one file, no dependencies, no magic.

The name is Gujarati for "move." The goal is not to be the strongest engine, but the most readable one. Every subsystem such as move generation, search, evaluation, UCI fits in a single scroll. The source is written to be studied.

## What "complete" means here

A lot of minimal engines cut corners: no en passant, no underpromotion, bugged castling after a rook is captured, draw detection that breaks perpetual-check defence. Chal doesn't. It passes all five standard perft positions and handles:

- En passant, all three underpromotions (knight, bishop, rook)
- Castling rights stripped correctly when a rook is captured at its home square
- 2-fold repetition detection (same hash, same side to move)
- 50-move rule (halfmove clock maintained incrementally and read from FEN)
- Stalemate and checkmate reported correctly to the GUI

It speaks UCI properly by streaming `info depth ... score cp ... pv` lines with the full principal variation during search, responding to `isready` before any position is set, handling `ucinewgame` with a clean TT flush.

## Philosophy

Most chess engine tutorials introduce concepts in isolation: here is a move generator, here is alpha-beta. You then spend weeks stitching them together into something that actually plays. Chal is the stitched result, the whole thing, readable as a book.

The file is split into 13 sections. Each section opens with a short comment explaining exactly why the technique exists and how the implementation works. Reading top-to-bottom gives you a complete picture of how a real engine is built: from the 0x88 board trick through Zobrist hashing, pseudo-legal generation, iterative deepening, null move pruning, aspiration windows, and a handwritten time manager.

There are no abstractions introduced for their own sake. No classes, no polymorphism, no templated containers. Just functions calling functions, documented as clearly as possible.

## Engine at a glance

### Board

- **0x88 representation** -- 128-element array; `sq & 0x88` detects off-board in one operation. No branch, no bounds table.
- **Transposition table** -- TT scores for mate are stored relative to the node that proved them and adjusted on retrieval, so the same mate is recognised correctly regardless of the transposition depth.
- **Incremental Zobrist hashing** -- 64-bit XOR key (Sungorus LCG) updated on every make/undo.
- **Piece list** -- compact list of up to 32 live pieces per side, maintained incrementally by make/undo. Eliminates full-board scans in evaluation and move generation.

### Move generation

- Unified pseudo-legal generator. Legality (king not left in check) is verified in the search loop after `make_move`.
- Leaper (knight, king) and slider (bishop, rook, queen) paths are split so the slider inner loop contains no leaper branch.
- `generate_captures()` is a dedicated capture generator for quiescence search: no `!caps_only` branches, no castling block, sliders use a skip-empty ray idiom (advance while empty, single capture check at end).
- Castling driven by four parallel data arrays (king-from, king-to, rook-from, rook-to). The generator validates rights, clear path, and unattacked king-path without any special-case branching.

### Search

- **Negamax alpha-beta** with a triangular PV table. The full planned line is tracked at every node and printed on every `info` line.
- **Iterative deepening** -- depth 1, 2, ... up to the limit. Each completed depth feeds the next via the TT best-move, making the overhead near zero.
- **Aspiration windows** -- full window at depths 1-4; dynamic initial delta from depth 5: `15 + prev_score^2 / 16384` (wider in unbalanced positions). Fail-low collapses beta to the midpoint before widening. Delta grows by 50% on each retry.
- **Reverse futility pruning** -- at depths 1-7, if static eval - 70*depth >= beta, prune without searching any moves.
- **Null move pruning** -- R=3 (R=4 at depth >= 7). Guarded against zugzwang: skipped when the side to move has no non-pawn, non-king piece. Guard is O(1) via incrementally-maintained per-piece-type counts `count[2][7]`. Double null moves prevented by a `was_null` flag.
- **Principal variation search** -- first legal move at each node gets the full window; all later moves are probed with a null window and only re-searched at full depth on a fail-high.
- **Check extension** -- if a move gives check, search depth is extended by 1 ply. Ensures the engine never horizon-drops into quiescence while the opponent is in check.
- **Late move reductions (adaptive LMR)** -- a precomputed table gives R = round(ln(depth) * ln(move_number) / 1.6), clamped to [1, 5]. Captures, promotions, and check-giving moves are never reduced. R grows an extra ply on non-PV nodes. Any move whose reduced score beats alpha is re-searched at full depth.
- **Internal iterative reductions (IIR)** -- if a node has no TT move and depth >= 4, depth is reduced by 1. The resulting TT entry improves ordering on the re-search.
- **PVS SEE pruning** -- in the main search loop, captures whose static exchange evaluation is below `-pawn * depth` are skipped once at least one legal move has been searched. No make_move needed, so no undo cost.
- **Late move pruning (LMP)** -- at depth < 4 on non-PV nodes, quiet moves beyond a threshold (`4 * depth + 1`) are skipped if they don't give check.
- **Razoring** -- at depth <= 3 on non-PV nodes, if static eval + margin is still below alpha, drop directly into quiescence search.
- **Quiescence search** -- fail-soft stand-pat with delta pruning (skip captures that cannot raise alpha even at best) and SEE pruning (skip captures where the full exchange is negative, via `is_bad_capture()`).
- **Repetition detection** -- two-tier: inside the search tree one prior occurrence returns draw immediately; in game history before the root, two prior occurrences required. Bounded by the halfmove clock.
- **50-move rule** -- halfmove clock tracked incrementally in `make_move`, read from FEN field 5; search returns draw score at 100 half-moves.

### Move ordering

1. TT / hash move -- 30 000
2. MVV-LVA captures -- 20 000 + 10 * victim value - attacker value; pawn-defended pawn captures scored below killers (-17 000 range)
3. Promotions -- 19 999
4. Killer move slot 0 -- 19 998
5. Killer move slot 1 -- 19 997
6. History -- bonus/malus from beta-cutoff tracking. The cutoff move receives `+depth^2` and every quiet move tried before it receives `-depth^2`. Both updates use a gravity formula so entries self-correct instead of saturating.

### Transposition table

- 1 048 576 entries (16 MB default); configurable via `setoption name Hash`. Power-of-two size for fast modulo.
- Stores exact score, upper bound, or lower bound with the best move.
- Depth-preferred replacement. Always stores on fail-low nodes to preserve the best move for future ordering.

### Evaluation

All terms are from the side-to-move's perspective; midgame and endgame scores are blended by game phase.

- **Tapered evaluation** -- separate MG and EG scores blended as `(mg*phase + eg*(24-phase)) / 24`, where phase counts remaining piece material (24 = opening, 0 = pure endgame).
- **Material** -- Rofchade Texel-tuned values: MG P=82 N=337 B=365 R=477 Q=1025; EG P=94 N=281 B=297 R=513 Q=937.
- **Piece-square tables** -- `mg_pst[6][64]` and `eg_pst[6][64]`, PeSTO tables.
- **Pawn evaluation** -- passed pawn detection (no opposing pawn ahead on same or adjacent file), with rank-dependent bonuses enhanced by king distance in the endgame. Doubled pawns -20 cp, isolated pawns -10 cp (MG and EG).
- **Mobility** -- pseudo-legal reachable squares above a per-piece centre target. Knights, bishops, rooks +/-3-4 cp/sq; queens +/-2 cp/sq; applied to both MG and EG.
- **Bishop pair** -- +31 MG, +30 EG.
- **Rook activity** -- semi-open file +10, open file +20; applied to both MG and EG.
- **King safety** -- pawn-shield check on the three files in front of a castled king; MG only.

### Time management

- Reads `wtime`, `btime`, `movestogo`, `winc`, `binc` from the `go` command.
- Budget = `our_time / movestogo + our_inc * 3/4`, capped 50 ms below the clock floor, minimum 5 ms. Defaults to 20 moves remaining when `movestogo` is absent.
- Hard abort every 1 024 nodes; soft stop after any depth that consumed more than half the budget.
- `go depth N` ignores the clock; used by test suites and analysis.

## Building

Requires a C99-compliant compiler.

```bash
gcc chal.c -O2 -Wall -Wextra -pedantic -std=c99 -o chal
```

## UCI

Chal works with any UCI-compatible GUI such as Arena, Cutechess, Banksia, or from the terminal directly:

```
uci
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000 movestogo 40
```

**Extension:** `perft [depth]` counts leaf nodes for move-generation validation.
`perft 6` from the start position must return `119 060 324`.

## Code map

`chal.c` runs linearly so no forward declarations are needed:

| Section | Content |
|---------|---------|
| S1 | Constants, types, move encoding, TT, PV, history |
| S2 | Board state, globals, piece list, time-control variables |
| S3 | Direction vectors and castling tables |
| S4 | Zobrist hashing |
| S5 | Attack detection |
| S6 | Make / undo move |
| S7 | Move generation |
| S8 | FEN parser |
| S9 | Evaluation |
| S10 | Move ordering |
| S11 | Search |
| S12 | Perft |
| S13 | UCI loop |

## Acknowledgements

**Pawel Koziol** ([nescitus](https://github.com/nescitus)) -- for thorough testing, bug reports, and architectural guidance throughout development. His feedback directly shaped the killer-move ply-indexing fix, the NMP ply-bookkeeping refactor, the PeSTO evaluation upgrade, the lazy pick-move sort, the history malus formula in v1.3.1, and the pawn evaluation refinements, state structure refactoring, search clarity initiatives, piece list implementation, QS pruning improvements, and the `is_square_attacked` split with dedicated capture generator in v1.4.0.

**Anik Patel** ([Bobingstern](https://github.com/Bobingstern)) -- for guiding the SPRT testing setup using [fastchess](https://github.com/Disservin/fastchess), making it possible to measure strength gains objectively across versions.
