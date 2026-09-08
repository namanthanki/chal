<div align="center">
<img src="logo.png" alt="Chal logo" width="200"/>

# Chal
</div>

**Chal** is a complete, FIDE-rules-compliant chess engine in **934 lines of C**.  
It uses a single source file, embeds its neural network, and has zero external dependencies.

The name is Gujarati for "move" or "tactic". The goal is not merely to write an engine, but a clean and readable one. Every subsystem in a single file written to be read top-to-bottom like a short book in a single sitting.

## Why under 1,000 lines?

It's a personal challenge.

I wanted to see how far a complete, readable chess engine could go within roughly a thousand lines of C. The limit forces trade-offs, keeps the code tight, and makes every part of the engine easier to follow in one pass.

With version 2.0, the engine was rewritten from scratch, replacing the original 0x88 board and handcrafted evaluation with bitboards, magic attack tables, and an NNUE trained on selfplay data of Chal 1.4.1's 35M positions. Most engines with this architecture span 8,000 to 16,000 lines of code. Chal fits the entire engine into **934 lines of C**—fewer lines than the 999 LOC of the original 0x88 version—while jumping from ~800k NPS to **~4 MNPS** (a 5x speedup) and gaining over 360 Elo.

## Ratings

| Version    | CCRL 40/15 | CCRL Blitz | Architecture |
|------------|------------|------------|--------------|
| Chal 1.3.0 | —          | 2284       | 0x88 + HCE   |
| Chal 1.3.2 | 2505       | 2466       | 0x88 + HCE   |
| Chal 1.4.0 | 2713       | —          | 0x88 + HCE   |
| Chal 1.4.1 | 2756       | 2764       | 0x88 + HCE   |
| Chal 2.0.0 | —          | ~3100 (est.)* | Bitboards + NNUE |

## What "complete" means here

Minimalist engines often cut corners, but Chal does not. It strictly implements:

- En passant and all three underpromotions (knight, bishop, rook)
- Full castling rules and path-safety checks
- Threefold repetition detection
- 50-move rule
- Insufficient material draw adjudication
- Stalemate and checkmate reported correctly to the GUI

## Philosophy

Most chess engine tutorials introduce concepts in isolation: here is a move generator, here is alpha-beta. You then spend weeks stitching them together into something that actually plays. Chal is the stitched result, the whole thing, readable as a book.

The file is split into 8 linear sections. Each section opens with a short comment explaining why the technique exists and how the implementation works. Reading top-to-bottom gives you a complete picture of how a modern engine is built: from bitboard operations through magic hashing, an embedded neural network, iterative deepening, pruning heuristics, and a time manager.

There are no abstractions introduced for their own sake. No classes, no polymorphism, no templated containers. Just standard C, functions calling functions, documented as clearly as possible.

## Features

- Move Generation
    - Unified pseudo-legal generator with 3 modes (all moves, captures/promotions, quiets)
- NNUE `(768 -> 32) x 2 -> 1`
    - Dual-perspective architecture (768 input features per perspective)
    - Clipped Squared ReLU (SCReLU) activation ($[0, 255]$ clamped, squared)
    - Trained on 35M positions of selfplay data from Chal 1.4.1 HCE
- Search
    - Negamax with alpha-beta pruning
    - Principle Variation Search (PVS)
    - Quiescence Search (`qsearch`) with stand-pat cutoffs and SEE filtering
    - Iterative Deepening
    - Adaptive Aspiration Windows
    - Transposition Table (16 MB default, 16-byte packed entries, TT aging, ply-normalized mate scores)
    - Move Ordering
        - Transposition Table (TT) Move
        - MVV-LVA for captures and promotions
        - Static Exchange Evaluation (SEE >= 0) verification
        - Killer Move Heuristic (2 slots per ply)
        - Butterfly History Heuristic indexed by `[side][from][to][src_threat][dst_threat]` (context-aware threat escapes and attacks)
    - Selectivity & Pruning
        - TT Cutoffs & Static Evaluation Bound Refinement
        - Check Extensions
        - Reverse Futility Pruning (RFP)
        - Razoring
        - Null Move Pruning (NMP) with dynamic reduction and non-pawn zugzwang verification
        - Internal Iterative Reductions (IIR)
        - History Pruning
        - Late Move Pruning (LMP)
        - Futility Pruning (FP)
        - Static Exchange Evaluation (SEE) Pruning
        - Late Move Reductions (LMR) based on logarithmic tables, history scores, and tactical threat escape bonus

## Code Map

`src/chal.c` runs strictly top-to-bottom without forward declarations:

| Section | Title | Content |
|---|---|---|
| **S1** | Constants & Types | Square coordinates, bitboard macros, move encoding |
| **S2** | Attacks & Magic Tables | Leaper lookups, slider ray masks, magic bitboards |
| **S3** | Zobrist Hashing & TT | Hash keys, position fingerprints, 16-byte TT entry |
| **S4** | NNUE Evaluation | Accumulator updates, SCReLU, forward pass, embedded net |
| **S5** | Board Representation | Dual mailbox/bitboard state, check detection, make/undo |
| **S6** | Move Generation | Pseudo-legal move generator (all, captures/promotions, quiets) |
| **S7** | Search & Heuristics | Negamax, alpha-beta, quiescence, move ordering, pruning |
| **S8** | UCI & Benchmarks | UCI protocol loop, time management, perft, and benchmark |

## Building

Requires a C99/GNU99 compiler.

### Using Make / MinGW

```bash
make          # Native host build (fastest on current CPU) -> bin/chal.exe
make avx2     # AVX2 + BMI2 build                          -> bin/chal-avx2.exe
make general  # Baseline x86-64 build                      -> bin/chal-general.exe
make bench    # Run the built-in benchmark
```

*(On Windows without MSYS2/make, use `mingw32-make`)*

### Direct Compiler Command

```bash
gcc -O3 -march=native -Wall -Wextra -pedantic -std=gnu99 src/chal.c -lm -o chal.exe
```

## UCI Usage

Chal communicates with any UCI-compatible GUI (Arena, Cutechess, Banksia, Nibbler) or from the terminal:

```
uci
isready
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000 movestogo 40
```

**Commands:**
- `perft <depth>`: Runs move-path verification. `perft 5` returns `4,865,609` nodes.
- `bench`: Runs the 6-position perft verification suite and depth-4 search benchmark.
- `eval`: Prints the static NNUE evaluation for the current position in centipawns.

## Acknowledgements

- **Pawel Koziol** ([nescitus](https://github.com/nescitus)) &mdash; Co-author, mentor, and steadfast collaborator from the very beginning. His architectural guidance, deep testing, and feedback have shaped Chal's evolution at every step.
- **Anik Patel** ([Bobingstern](https://github.com/Bobingstern)) &mdash; For guiding the automated SPRT testing setup using [fastchess](https://github.com/Disservin/fastchess).
- **Gediminas Masaitis** ([GediminasMasaitis](https://github.com/GediminasMasaitis)) &mdash; For tuning the evaluation in earlier versions of Chal.
- **The Chess Programming Community & CCRL Testers** &mdash; Sincere thanks to Gabor Szots, Andres Valverde, and Graham Banks for testing and tracking Chal on the rating lists.
