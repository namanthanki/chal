# Chal (v1.1)

**Chal** (Gujarati for "move") is a minimal, didactic chess engine written entirely in a single file of strictly-compliant classical C (ANSI C89 / C90).

The core goal of this project is to serve as an educational resource to teach the fundamental concepts of chess engine development. The source file reads top-to-bottom, commented with minimal theories avoiding heavy abstraction.

## Architecture

A single file (`src/chal.c`, **913 lines of code / 521 lines of comments**) split into 13 clearly labelled sections. Each section is a self-contained lesson:

### Board Representation
- **0x88 Geometry** — 128-square 1D array; off-board detection is one bitwise `AND` (`sq & 0x88`).
- **Incremental Zobrist Hashing** — 32-bit XOR fingerprints updated on every make/undo; O(1) hash restoration from the history stack.

### Move Generation
- Unified pseudo-legal generator; legality (king not in check) resolved in the search loop.
- `caps_only=1` mode gates to captures and promotions only — powers quiescence search without a separate routine.
- Table-driven castling: four hard-coded king/rook configurations, validated by rights, emptiness, and attack checks.

### Search
- **Negamax Alpha-Beta** with a triangular PV table.
- **Iterative Deepening** — each completed depth seeds the next via the TT best-move.
- **Aspiration Windows** — ±50 cp window from depth 4 onward; doubles on fail-low/fail-high.
- **Quiescence Search** — stand-pat with delta pruning; capped at 6 extra plies.
- **Null Move Pruning** — R=2 (R=3 at depth ≥ 6); skipped in pure pawn/king endings to avoid zugzwang.
- **Late Move Reductions** — quiet moves sorted after the 4th at depth ≥ 3 are searched at depth−2, re-searched full-width only if they beat alpha.
- **Repetition Detection** — 2-fold draw by hash comparison over the game history.

### Move Ordering
1. TT / hash move (20 000)
2. MVV-LVA captures (1 000+)
3. Promotion (900)
4. Killer moves — 2 slots per ply (800)
5. History heuristic — depth² credit on beta cutoffs, capped at 32 000

### Transposition Table
- 1 048 576 entries (16 MB); power-of-two size for fast modular indexing.
- Depth-preferred replacement; stores exact / lower-bound / upper-bound flags.

### Evaluation
- **Material** — centipawn values per piece type.
- **Piece-Square Tables** — signed-char PSTs for pawn, knight, bishop, rook, queen.
- **Phase-Aware King** — smoothly blends a middlegame safety table and an endgame centralisation table by remaining non-pawn material.
- **Mobility** — pseudo-legal ray count bonus for knights, bishops, rooks, and queens.
- **Pawn Structure** — doubled-pawn penalty, isolated-pawn penalty, passed-pawn bonus scaled by rank squared.
- **Bishop Pair** — +30 cp when a side retains both bishops.
- **Rook Activity** — semi-open (+10) / open-file (+20) bonuses; 7th-rank bonus (+20).
- **King Safety** — pawn-shield penalty on the three files in front of the king, scaled by game phase.

### Time Management
- Parses `wtime`, `btime`, `movestogo`, `winc`, `binc` from the UCI `go` command.
- Budget: `our_time / movestogo + our_inc * 3/4`, capped 50 ms below the clock floor.
- Hard abort every 1 024 nodes; soft stop after any depth that consumed > half the budget.

## Building

Requires only a C90-compliant compiler; no external libraries.

```bash
make
```

### Manual Compilation
```bash
gcc src/chal.c -O2 -Wall -Wextra -pedantic -std=gnu90 -o bin/chal
```

## Protocol (UCI)

Chal communicates over the Universal Chess Interface (UCI) and works with any UCI-compatible GUI (Arena, Cutechess, Banksia) or the raw console:

```text
uci
position startpos moves e2e4 e7e5
go wtime 10000 btime 10000 movestogo 40
```

**Extension command** — `perft [depth]`  
Leaf-node count for move-generation validation.  
`perft 6` from the start position must return `119 060 324`.

## Code Map

`src/chal.c` is organised linearly — no forward declarations needed:

| Section | Topic |
|---------|-------|
| S1 | Constants, types, move encoding, TT, PV, history |
| S2 | Board state, globals, time-control variables |
| S3 | Direction vectors & castling tables |
| S4 | Zobrist hashing |
| S5 | Attack detection (`is_square_attacked`) |
| S6 | Make / undo move |
| S7 | Move generation |
| S8 | FEN parser |
| S9 | Evaluation |
| S10 | Move ordering |
| S11 | Search (negamax, QS, NMP, LMR, aspiration) |
| S12 | Perft |
| S13 | UCI loop |

## Changelog

### v1.1
- **fix:** expand UCI input buffer from 1 024 B to 64 KiB — prevents illegal moves when the `position … moves` command line exceeds 1 024 characters (games longer than ~200 half-moves).

### v1.0
- Initial release.
