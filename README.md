# Chal

**Chal** is a complete, FIDE-rules-compliant chess engine in **891 lines of C99** it has one file, no dependencies, no magic.

The name is Gujarati for "move." The goal is not to be the strongest engine, but the most readable one. Every subsystem such as move generation, search, evaluation, UCI fits in a single scroll. The source is written to be studied.

## What "complete" means here

A lot of minimal engines cut corners: no en passant, no underpromotion, bugged castling after a rook is captured, draw detection that breaks perpetual-check defence. Chal doesn't. It passes all five standard perft positions and handles:

- En passant, all four underpromotions
- Castling rights stripped correctly when a rook is captured at its home square
- 2-fold repetition detection (same hash, same side to move)
- 50-move rule (halfmove clock maintained incrementally and read from FEN)
- Stalemate and checkmate reported correctly to the GUI

It speaks UCI properly by streaming `info depth … score cp … pv` lines with the full principal variation during search, responding to `isready` before any position is set, handling `ucinewgame` with a clean TT flush.

## Philosophy

Most chess engine tutorials introduce concepts in isolation: here is a move generator, here is alpha-beta. You then spend weeks stitching them together into something that actually plays. Chal is the stitched result, the whole thing, readable as a book.

The file is split into 13 sections. Each section opens with a short comment explaining exactly why the technique exists and how the implementation works. Reading top-to-bottom gives you a complete picture of how a real engine is built: from the 0x88 board trick through Zobrist hashing, pseudo-legal generation, iterative deepening, null move pruning, aspiration windows, and a handwritten time manager.

There are no abstractions introduced for their own sake. No classes, no polymorphism, no templated containers. Just functions calling functions, documented as clearly as possible.

## Engine at a glance

### Board

- **0x88 representation** — 128-element array; `sq & 0x88` detects off-board in one operation. No branch, no bounds table.
- **Transposition table** — TT scores for mate are stored relative to the node that proved them and adjusted on retrieval, so the same mate is recognised correctly regardless of the transposition depth.
- **Incremental Zobrist hashing** — 64-bit XOR key (Sungorus LCG) updated on every make/undo.

### Move generation

- Unified pseudo-legal generator. Legality (king not left in check) is verified in the search loop after `make_move`.
- `caps_only=1` restricts output to captures and promotions — quiescence search calls the same function, not a separate one.
- Castling driven by four parallel data arrays (king-from, king-to, rook-from, rook-to). The generator validates rights, clear path, and unattacked king-path without any special-case branching.

### Search

- **Negamax alpha-beta** with a triangular PV table. The full planned line is tracked at every node and printed on every `info` line.
- **Iterative deepening** — depth 1, 2, … up to the limit. Each completed depth feeds the next via the TT best-move, making the overhead near zero.
- **Aspiration windows** — full window at depths 1–3; ±50 cp window from depth 4. Delta doubles on fail-low or fail-high.
- **Reverse futility pruning** — at depths 1–7, if static eval − 70·depth ≥ β, prune without searching any moves. Zero nodes spent per pruned node.
- **Null move pruning** — R=3 (R=4 at depth ≥ 7). Guarded against zugzwang: skipped when the side to move has no non-pawn, non-king piece. Guard is O(1) via an incrementally-maintained `non_pawn_count[2]` table. Double null moves prevented by a `was_null` flag.
- **Principal variation search** — first legal move at each node gets the full window; all later moves are probed with a null window and only re-searched at full depth on a fail-high.
- **Late move reductions (adaptive LMR)** — a precomputed table gives reduction R = round(ln(depth) × ln(move_number) / 1.6), clamped to [1, 5]. Captures, promotions, and check-giving moves are never reduced. R grows an extra ply on non-PV nodes. Any move whose reduced score beats alpha is re-searched at full depth.
- **Quiescence search** — fail-soft stand-pat with delta pruning.
- **Repetition detection** — two-tier: inside the search tree one prior occurrence returns draw immediately (opponent can always force a third); in game history before the root, two prior occurrences required (strict threefold). Bounded by the halfmove clock.
- **50-move rule** — halfmove clock tracked incrementally in `make_move`, read from FEN field 5; search returns draw score at 100 half-moves.

### Move ordering

1. TT / hash move — 30 000
2. MVV-LVA captures — 20 000 + 10 × victim value − attacker value
3. Queen promotion — 19 999
4. Killer move slot 0 — 19 998
5. Killer move slot 1 — 19 997
6. History: bonus/malus from beta-cutoff tracking. The cutoff move receives +depth^2 and every quiet move tried before it receives −depth^2 (history malus). Both updates use a formula (diminishing returns) so entries self-correct instead of saturating. Negative scores push failing moves to the bottom of the ordering without ever skipping them.

### Transposition table

- 1 048 576 entries (16 MB); power-of-two size for fast `% TT_SIZE` via bitwise AND.
- Stores exact score, upper bound, or lower bound with the best move.
- Depth-preferred replacement.

### Evaluation

All terms are from the side-to-move's perspective; middlegame and endgame scores are blended by game phase.

- **Tapered evaluation** — separate MG and EG scores blended as `(mg×phase + eg×(24−phase)) / 24`, where phase counts remaining piece material (24 = opening, 0 = pure endgame).
- **Material** — Rofchade Texel-tuned values: MG P=82 N=337 B=365 R=477 Q=1025; EG P=94 N=281 B=297 R=512 Q=936.
- **Piece-square tables** — `mg_pst[6][64]` and `eg_pst[6][64]`, Rofchade PeSTO tables. Passed-pawn rewards are folded implicitly into the EG pawn table.
- **Mobility** — pseudo-legal reachable squares above a per-piece centre target. Knights, bishops, rooks ±3 cp/sq; queens ±2 cp/sq; applied to both MG and EG.
- **Bishop pair** — +30 cp in both MG and EG.
- **Rook activity** — semi-open file +10, open file +20, 7th rank +20; applied to both MG and EG.
- **Pawn structure** — doubled −20 cp, isolated −10 cp; applied to both MG and EG.
- **King safety** — pawn-shield check on the three files in front of a castled king; −15 cp per missing pawn, −25 cp extra on a fully open file; MG only (fades naturally with phase).

### Time management

- Reads `wtime`, `btime`, `movestogo`, `winc`, `binc` from the `go` command.
- Budget = `our_time / movestogo + our_inc × 3/4`, capped 50 ms below the clock floor, minimum 5 ms.
- Hard abort every 1 024 nodes; soft stop after any depth that consumed more than half the budget.
- `go depth N` ignores the clock; used by test suites and analysis.

## Building

Requires a C99-compliant compiler.

```bash
make
```

```bash
# or manually:
gcc src/chal.c -O2 -Wall -Wextra -pedantic -std=gnu99 -o chal
```

## UCI

Chal works with any UCI-compatible GUI such as Arena, Cutechess, Banksia or from the terminal directly:

```
uci
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000 movestogo 40
```

**Extension:** `perft [depth]` counts leaf nodes for move-generation validation.
`perft 6` from the start position must return `119 060 324`.

## Code map

`src/chal.c` runs linearly so no forward declarations are needed:

| Section | Content |
|---------|---------|
| S1 | Constants, types, move encoding, TT, PV, history |
| S2 | Board state, globals, time-control variables |
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

**Pawel Koziol** ([nescitus](https://github.com/nescitus)) — for thorough testing, bug reports, and architectural guidance throughout the v1.2.1 → v1.3.1 history.
His feedback directly shaped the killer-move ply-indexing fix, the NMP ply-bookkeeping refactor, the PeSTO evaluation upgrade, the lazy pick-move sort, and the history malus + formula in v1.3.1.

**Anik Patel** ([Bobingstern](https://github.com/Bobingstern)) — for guiding the SPRT testing setup using [fastchess](https://github.com/Disservin/fastchess), making it possible to measure strength gains objectively across versions.

