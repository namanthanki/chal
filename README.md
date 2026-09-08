<div align="center">
<img src="logo.png" alt="Chal logo" width="200"/>

# Chal
</div>

**Chal** is a complete, FIDE-rules-compliant chess engine in **under 999 lines of C99**.  
It follows all FIDE rules, lives in a single source file, embeds its neural network, and has zero external dependencies.

The name is Gujarati for "move." The goal is not merely to write an engine, but to write a readable one. Every subsystem—bitboards, magic attack tables, an embedded neural network (NNUE), search heuristics, and UCI—is written to be read from top to bottom like a short book in a single sitting.

## Why 999 lines?

It is a personal challenge.

Most modern chess engines span anywhere from 8,000 to over 16,000 lines of code. Chal is now a complete rewrite from the ground up: replacing the original 0x88 board and handcrafted evaluation with a 64-bit bitboard architecture, Fancy Magic Bitboards for ray attacks, and an embedded NNUE network (768 &rarr; 32 &rarr; 1) with portable 256-bit SIMD vector accumulation.

Despite the added complexity of bitboards and neural evaluation, Chal packs the entire engine into **934 lines of C code**—fewer lines than the 999 LOC of the original 0x88 version. At the same time, search throughput jumped from ~800k NPS to **~4 MNPS** (a 5x speedup), and playing strength gained hundreds of Elo.

The strict constraint forces trade-offs, eliminates bloat, and keeps every algorithm direct, readable, and focused.

## Ratings

| Version    | CCRL 40/15 | CCRL Blitz | Architecture |
|------------|------------|------------|--------------|
| Chal 1.3.0 | —          | 2284       | 0x88 + HCE   |
| Chal 1.3.2 | 2505       | 2466       | 0x88 + HCE   |
| Chal 1.4.0 | 2713       | —          | 0x88 + HCE   |
| Chal 1.4.1 | 2756       | 2764       | 0x88 + HCE   |

## What "complete" means here

Minimalist engines often cut corners but Chal does not. It strictly implements:

- En passant and all four promotion variants (queen, rook, bishop, knight)
- Full castling rights and path-safety checks
- Threefold repetition detection
- 50-move rule
- Insufficient material draw adjudication
- Stalemate and checkmate reported accurately to the GUI

## Philosophy

Most chess programming tutorials introduce components in isolation: here is a move generator, here is alpha-beta. Bridging them into a functional, competitive engine often takes weeks of debugging. Chal provides the stitched, working whole—crafted as a unified lesson.

The file is organized into 8 linear sections. Each section opens with an understated, didactic commentary explaining the theoretical foundation and the implementation choices. Reading the file top-to-bottom gives you a complete picture of modern engine design without navigating deep directory hierarchies or deciphering complex abstractions.

No classes, no polymorphism, no template metaprogramming. Just plain functions, well-chosen data representations, and standard C.

## Engine at a glance

### Board & Representation (S1, S5)
- 64-bit bitboards for piece sets, side occupancies, and ray masking
- 64-byte mailbox array for instant square-lookup without bitboard scanning
- Dual representation kept synchronized across `make_move` and `undo_move`
- Zobrist hashing with 64-bit xorshift pseudo-random numbers; incremental XOR updates
- Transposition table (16 MB default) with packed 16-byte entries and ply-normalized mate scores

### Attack Detection & Magic Tables (S2)
- Precomputed attack lookup tables for pawns, knights, and kings
- Fancy Magic Bitboards for bishops, rooks, and queens
- Carry-Rippler subset iteration for rapid startup table initialization

### NNUE Evaluation (S4)
- Architecture: 768 &rarr; 32 &rarr; 1 (HalfKP style, dual perspective)
- SCReLU activation (squared clipped ReLU: clamp to [0, 255], square)
- Incremental accumulator stack updated during `make_move` / `undo_move`
- Portable 256-bit SIMD vectorization via compiler vector extensions (`v16 __attribute__((vector_size(32)))`), emitting AVX2 `vpaddw`/`vpsubw` on supported hardware with automatic fallback on generic x86-64
- Self-contained: weights embedded directly in `net.h`, producing standalone binaries

### Move Generation (S6)
- Unified pseudo-legal move generator with 3 operating modes:
  - Mode 0: All legal candidates (search)
  - Mode 1: Captures and promotions only (quiescence search)
  - Mode 2: Quiet moves only
- Bitwise target masks (`~pos->occ[us]` vs `pos->occ[them]`)
- Special handling for pawn pushes, double pushes, promotions, and en passant
- Castling verified via path emptiness and 3-square safety checks

### Search & Heuristics (S7)
- Negamax formulation with alpha-beta pruning
- Quiescence search (`qsearch`) with stand-pat cutoff to prevent horizon effects
- Iterative deepening with adaptive aspiration windows
- Move ordering:
  1. Transposition table (TT) move
  2. Winning and equal captures (MVV-LVA checked by SEE)
  3. Killer moves (2 slots per ply)
  4. History heuristic (butterfly table refined with tactical threat context)
- Pruning & Reductions:
  - Reverse Futility Pruning (RFP)
  - Razoring
  - Null Move Pruning (NMP) with non-pawn zugzwang guard
  - Late Move Reductions (LMR) with bonus for moves escaping enemy threats
  - Futility Pruning & Late Move Pruning (LMP)
  - Static Exchange Evaluation (SEE) pruning

### Time Management & Protocol (S8)
- Standard Universal Chess Interface (UCI) over stdin/stdout
- Dual-boundary clock manager: volatility-scaled soft limit and hard emergency ceiling
- Built-in `perft` suite for mathematical move generation and make/undo validation

## Code Map

`src/chal.c` runs strictly top-to-bottom without forward declarations:

| Section | Title | Description |
|---|---|---|
| **S1** | Constants & Types | Square coordinates, bitboard macros, 16-bit move layout, move list |
| **S2** | Attacks & Magic Tables | Leaper lookups, slider ray masks, magic bitboards, carry-rippler setup |
| **S3** | Zobrist Hashing & TT | Hash seeds, position fingerprints, 16-byte TT entry, probe & store |
| **S4** | NNUE Evaluation | Dual-perspective accumulator, SCReLU, SIMD vector math, embedded net |
| **S5** | Board Representation | Dual mailbox/bitboard state, outward check detection, make/undo |
| **S6** | Move Generation | Pseudo-legal move generator (all, captures/promotions, quiets) |
| **S7** | Search & Heuristics | Negamax, alpha-beta, quiescence, move ordering, pruning & reductions |
| **S8** | UCI & Benchmarks | Line protocol parser, time allocation, perft suite, search benchmark |

## Building

Requires a standard C99/GNU99 compiler.

### Building with Make / MinGW

```bash
make          # Native host architecture (default, fastest) -> bin/chal.exe
make avx2     # Modern CPUs with AVX2/BMI2 (x86-64-v3)      -> bin/chal-avx2.exe
make general  # Compatible with all x86-64 (SSE4.2)         -> bin/chal-general.exe
make bench    # Run the verification test suite
```

*(On Windows without MSYS2/make, use `mingw32-make`)*

### Direct Compiler Command

```bash
# Native build
gcc -O3 -march=native -Wall -Wextra -pedantic -std=gnu99 src/chal.c -lm -o chal.exe
```

## UCI Usage

Chal communicates with any standard chess GUI (Arena, Cutechess, Banksia, Nibbler) or from the terminal:

```
uci
isready
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000 movestogo 40
```

**Commands:**
- `perft <depth>`: Runs move-path verification to depth $N$. `perft 5` from startpos returns `4,865,609` nodes.
- `bench`: Executes the 6-position perft verification suite and depth-4 search benchmark.
- `eval`: Prints the static NNUE evaluation for the current position in centipawns.

## Acknowledgements

- **Pawel Koziol** ([nescitus](https://github.com/nescitus)) &mdash; Co-author, mentor, and steadfast collaborator from the very beginning. His architectural guidance, deep testing, and feedback have shaped Chal's evolution at every step.
- **Anik Patel** ([Bobingstern](https://github.com/Bobingstern)) &mdash; For guiding the automated SPRT testing framework with [fastchess](https://github.com/Disservin/fastchess).
- **Gediminas Masaitis** ([GediminasMasaitis](https://github.com/GediminasMasaitis)) &mdash; For invaluable tuning insights and contributions to Chal's evaluation history.
- **The Chess Programming Community & CCRL Testers** &mdash; Sincere thanks to Gabor Szots, Andres Valverde, and Graham Banks for their dedication to testing and tracking Chal on the rating lists.
