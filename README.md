# Chal

**Chal** is a complete, FIDE-rules-compliant chess engine in **999 lines of C99**.  
It follows FIDE rules, uses a single file, and has no dependencies.

The name is Gujarati for "move." The goal is not to be the strongest engine, but a readable one. Every subsystem such as move generation, search, evaluation, UCI fits in a single scroll. The source is written to be read in a single sitting.

## Why 999 lines?

It’s a personal challenge.

I wanted to see how far a complete and readable chess engine could go within roughly a thousand lines of C. The limit forces trade-offs, keeps the code tight, and makes every part of the engine easier to follow in one pass.

It’s not about strength. A constraint like this naturally limits performance. The goal is clarity, compactness, and the fun of building something complete within a strict boundary.

## Ratings

| Version    | CCRL 40/15 | CCRL Blitz |
|------------|------------|------------|
| Chal 1.3.0 | —          | 2283       |
| Chal 1.3.2 | 2505       | 2465       |
| Chal 1.4.0 | 2718       | —          |
| Chal 1.4.1 | _          | 2764       |

## What "complete" means here

A lot of minimal engines cut corners: no en passant, no underpromotion, bugged castling, bad draw detection, etc. Chal doesn't. It handles:

- En passant, all three underpromotions (knight, bishop, rook)
- Castling rules
- Repetition detection
- 50-move rule
- Stalemate and checkmate reported correctly to the GUI

## Philosophy

Most chess engine tutorials introduce concepts in isolation: here is a move generator, here is alpha-beta. You then spend weeks stitching them together into something that actually plays. Chal is the stitched result, the whole thing, readable as a book.

The file is split into 13 sections. Each section opens with a short comment explaining exactly why the technique exists and how the implementation works. Reading top-to-bottom gives you a complete picture of how a real engine is built: from the 0x88 board trick through Zobrist hashing, pseudo-legal generation, iterative deepening, null move pruning, aspiration windows, and a time manager.

There are no abstractions introduced for their own sake. No classes, no polymorphism, no templated containers. Just functions calling functions, documented as clearly as possible.

## Engine at a glance

### Board

- 0x88 representation
- Transposition table
- Zobrist hashing
- Piece list

### Move generation

- pseudo-legal generator
- Leaper (knight, king) and slider (bishop, rook, queen) paths are split so the slider inner loop contains no leaper branch.
- `generate_captures()` is a dedicated capture generator for quiescence search: no `!caps_only` branches, no castling block, sliders use a skip-empty ray idiom (advance while empty, single capture check at end).
- Castling driven by four parallel data arrays (king-from, king-to, rook-from, rook-to). The generator validates rights, clear path, and unattacked king-path without any special-case branching.

### Search

- Negamax with alpha-beta
- Iterative deepening
- Aspiration windows
- Reverse futility pruning
- Null move pruning
- Principal variation search
- Check extension
- Late move reductions (adaptive LMR)
- Internal iterative reductions
- PVS SEE pruning
- Late move pruning
- Razoring
- Quiescence search
- Repetition detection
- 50-move rule

### Move ordering

1. TT / hash move
2. MVV-LVA captures
3. Promotions
4. Killer move slot 0
5. Killer move slot 1
6. History

### Transposition table

- Default size: 16 MB
- Stores exact scores and bounds
- Depth-based replacement strategy

### Evaluation

All terms are from the side-to-move's perspective; midgame and endgame scores are blended by game phase.

- Tapered evaluation
- Material (Rofchade Texel-tuned values: MG P=82 N=337 B=365 R=477 Q=1025; EG P=94 N=281 B=297 R=513 Q=937)
- Piece-square tables (Tuned PeSTO tables)
- Pawn evaluation
- Mobility
- Bishop pair
- Rook activity
- Psuedo-King safety

## Building

Requires a C99-compliant compiler.

```bash
gcc chal.c -O3 -Wall -Wextra -pedantic -std=c99 -o chal
```
Notes:
- add -lm flag for linux machines
- can directly use the provided binaries which will work with most of the 64-bit machines out there

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

**Gediminas Masaitis ([GediminasMasaitis](https://github.com/GediminasMasaitis)) -- for tuning the entire eval which results in 30+ ELO performance! it's not part of 1.4.0, but will be released soon.
