# Chal

**Chal** is a complete, FIDE-rules-compliant chess engine in **776 lines of C90** it has one file, no dependencies, no magic.

The name is Gujarati for "move." The goal is not to be the strongest engine, but the most readable one. Every subsystem such as move generation, search, evaluation, UCI fits in a single scroll. The source is written to be studied.

## What "complete" means here

A lot of minimal engines cut corners: no en passant, no underpromotion, bugged castling after a rook is captured, draw detection that breaks perpetual-check defence. Chal doesn't. It passes all five standard perft positions and handles:

- En passant, all four underpromotions
- Castling rights stripped correctly when a rook is captured at its home square
- 2-fold repetition detection (same hash, same side to move)
- Stalemate and checkmate reported correctly to the GUI

It speaks UCI properly by streaming `info depth … score cp … pv` lines with the full principal variation during search, responding to `isready` before any position is set, handling `ucinewgame` with a clean TT flush.

## Philosophy

Most chess engine tutorials introduce concepts in isolation: here is a move generator, here is alpha-beta. You then spend weeks stitching them together into something that actually plays. Chal is the stitched result, the whole thing, readable as a book.

The file is split into 13 sections. Each section opens with a short comment explaining exactly why the technique exists and how the implementation works. Reading top-to-bottom gives you a complete picture of how a real engine is built: from the 0x88 board trick through Zobrist hashing, pseudo-legal generation, iterative deepening, null move pruning, aspiration windows, and a handwritten time manager.

There are no abstractions introduced for their own sake. No classes, no polymorphism, no templated containers. Just functions calling functions, documented as clearly as possible.

## Engine at a glance

### Board

- **0x88 representation** — 128-element array; `sq & 0x88` detects off-board in one operation. No branch, no bounds table.
- **Incremental Zobrist hashing** — 32-bit XOR key updated on every make/undo. Restoring the hash in `undo_move` is a single assignment from the history stack.

### Move generation

- Unified pseudo-legal generator. Legality (king not left in check) is verified in the search loop after `make_move`.
- `caps_only=1` restricts output to captures and promotions — quiescence search calls the same function, not a separate one.
- Castling driven by four parallel data arrays (king-from, king-to, rook-from, rook-to). The generator validates rights, clear path, and unattacked king-path without any special-case branching.

### Search

- **Negamax alpha-beta** with a triangular PV table. The full planned line is tracked at every node and printed on every `info` line.
- **Iterative deepening** — depth 1, 2, … up to the limit. Each completed depth feeds the next via the TT best-move, making the overhead near zero.
- **Aspiration windows** — full window at depths 1–3; ±50 cp window from depth 4. Delta doubles on fail-low or fail-high.
- **Null move pruning** — R=2 (R=3 at depth ≥ 6). Guarded against zugzwang: skipped when the side to move has no non-pawn, non-king piece.
- **Late move reductions** — quiet moves after the 4th at depth ≥ 3 are searched at depth−2 with a null window; re-searched at depth−1 only if they beat alpha.
- **Quiescence search** — stand-pat evaluation with delta pruning; capped at 6 plies beyond the horizon.
- **Repetition detection** — returns 0 immediately on a hash match (step by 2, bounded to the last 50 half-moves).

### Move ordering

1. TT / hash move — 20 000
2. MVV-LVA captures — 1 000 + victim value − attacker value
3. Underpromotions — 900
4. Killer moves — 2 slots per ply, score 800
5. History heuristic — depth² credit on quiet beta cutoffs, 128×128 from/to table, capped at 32 000

### Transposition table

- 1 048 576 entries (16 MB); power-of-two size for fast `% TT_SIZE` via bitwise AND.
- Stores exact score, upper bound, or lower bound with the best move.
- Depth-preferred replacement.

### Evaluation

All terms are white-perspective centipawns; returned from side-to-move's perspective.

- **Material** — standard centipawn values (P=100, N=320, B=330, R=500, Q=900).
- **Piece-square tables** — `pst[8][64]` signed-char array, one row per piece type. King uses two rows blended by phase.
- **Game phase** — interpolated from total non-pawn material; smoothly transitions king evaluation from middlegame safety to endgame centralisation.
- **Mobility** — pseudo-legal reachable squares. Knights, bishops, rooks score +2 cp each; queens +1 cp.
- **Bishop pair** — +30 cp.
- **Rook activity** — semi-open file +10, open file +20, 7th rank +20.
- **Passed pawns** — rank² × 2 bonus (2 → 8 → 18 → 32 → 50 → 72 cp).
- **Pawn structure** — doubled −20 cp, isolated −10 cp.
- **King safety** — pawn-shield check on the three files in front of a castled king; −15 cp per missing pawn, −25 cp extra on a fully open file; fades to zero in the endgame.

### Time management

- Reads `wtime`, `btime`, `movestogo`, `winc`, `binc` from the `go` command.
- Budget = `our_time / movestogo + our_inc × 3/4`, capped 50 ms below the clock floor, minimum 5 ms.
- Hard abort every 1 024 nodes; soft stop after any depth that consumed more than half the budget.
- `go depth N` ignores the clock; used by test suites and analysis.

## Building

Requires only a C90-compliant compiler.

```bash
make
```

```bash
# or manually:
gcc src/chal.c -O2 -Wall -Wextra -pedantic -std=gnu90 -o chal
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

