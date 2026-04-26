/*
================================================================
                          C H A L
================================================================
   Gujarati for "move." A minimal chess engine in C99.

   Author : Naman Thanki
   Date   : 2026

   This file is meant to be read as a book, not just run.
   Every subsystem is a short lesson in engine design.

   Compile:  gcc chal.c -O2 -Wall -Wextra -pedantic -std=c99 -o chal
   Protocol: Universal Chess Interface (UCI)
================================================================

   TABLE OF CONTENTS
   -----------------
   S1  Constants & Types         - pieces, moves, TT, PV, state
   S2  Board State               - 0x88 grid, global telemetry
   S3  Direction & Castling Data - geometric move vectors
   S4  Zobrist Hashing           - position fingerprints
   S5  Attack Detection          - sonar-ping ray scanning
   S6  Make / Undo               - incremental board updates
   S7  Move Generation           - unified move generation
   S8  FEN Parser                - reading position strings
   S9  Evaluation                - material, geometry, and structure
   S10 Move Ordering             - MVV-LVA, killers, and history
   S11 Search                    - qsearch, negamax alpha-beta, PVS
   S12 Perft                     - correctness testing
   S13 UCI Loop                  - GUI communication

   ================================================================

   NAMING CONVENTIONS
   ------------------
   fr, to    = from-square, to-square of a move
   pt        = piece type  (PAWN..KING, integer 1-6)
   p         = piece       (encoded as color<<3 | type)
   c, color  = piece color (WHITE=0, BLACK=1)
   sq        = board square (0x88 index, valid if !(sq & 0x88))
   cap       = captured piece (0 if none)
   pr        = promotion piece type (0 if none)
   sply      = search ply from root (0 at root)
   ply       = game ply (incremented by make_move)
   depth     = remaining search depth (counts down to 0)
   mg, eg    = midgame score, endgame score
   ac        = attacking color (used in attack detection)

   SQUARE HELPERS (prefer these over raw board[] access)
   ------------------------------------------------------
   piece_on(sq)        -> board[sq]          (safe on any square)
   is_empty(sq)        -> board[sq] == EMPTY (safe on any square)
   piece_is(sq, c, t)  -> board[sq] == make_piece(c, t) (safe on any square)
   ptype_on(sq)        -> piece type         (call only on non-empty square)
   color_on(sq)        -> piece color        (call only on non-empty square)

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#define CHAL_VERSION "1.4.1"

/* ===============================================================
   S1  CONSTANTS & TYPES
   ===============================================================

   PIECE ENCODING
   --------------
   One byte per piece.  Bit 3 = colour (0=White, 1=Black).
   Bits 2..0 = type (1=Pawn .. 6=King, 0=Empty).

       make_piece(c,t)  ->  (c<<3)|t
       piece_type(p)    ->  p & 7
       piece_color(p)   ->  p >> 3
*/

enum { EMPTY = 0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum { WHITE = 0, BLACK = 1 };
enum { SQ_NONE = -1 };
static inline int piece_type(int p) { return p & 7; }
static inline int piece_color(int p) { return p >> 3; }
static inline int make_piece(int c, int t) { return (c << 3) | t; }
enum { INF = 50000, MATE = 30000 };
const int WP = 1, BP = 9;

// globals for see
short see_cleared[128];
short see_sentinel = 1;

/* ---------------------------------------------------------------
   MOVE ENCODING
   ---------------------------------------------------------------
   Idea
   A chess move requires a source square, a target square, and
   optional promotion information.

   Implementation
   We pack this data into a single 32-bit integer for performance.
   Passing a scalar integer by value is extremely fast and natively
   supports register placement.

   Bits  0.. 6  ->  from-square  (0-127)
   Bits  7..13  ->  to-square    (0-127)
   Bits 14..17  ->  promotion piece type (0 = none)

       move_from(m)          ->  m & 0x7F
       move_to(m)            ->  (m >> 7) & 0x7F
       move_promo(m)         ->  (m >> 14) & 0xF
       make_move_enc(f,t,p)  ->  f|(t<<7)|(p<<14)
*/

typedef int Move;
static inline int  move_from(Move m) { return m & 0x7F; }
static inline int  move_to(Move m) { return (m >> 7) & 0x7F; }
static inline int  move_promo(Move m) { return (m >> 14) & 0xF; }
static inline Move make_move_enc(int fr, int to, int p) { return fr | (to << 7) | (p << 14); }

/* ---------------------------------------------------------------
   TRANSPOSITION TABLE (TT)
   ---------------------------------------------------------------
   Idea
   Different move orders can reach the identical board position.
   Evaluating the same node logic multiple times wastes time. A TT
   caches search results, preventing redundant sub-tree exploration.

   Implementation
   We allocate a global hash map of `TTEntry` structures, indexed
   by Zobrist keys. To minimize memory footprint and avoid cache
   thrashing, the scalar values `depth` and `flag` are bitwise packed
   into a single `unsigned char`.

   Pack:   depth_flag = (depth << 2) | flag
   Unpack: depth = depth_flag >> 2
           flag  = depth_flag & 3
*/

enum { TT_EXACT = 0, TT_ALPHA = 1, TT_BETA = 2 };
typedef uint64_t HASH;  /* 64-bit Zobrist key -- halves collision rate vs 32-bit */
typedef struct { HASH key; int score; Move best_move; unsigned int depth_flag; } TTEntry;
TTEntry* tt = NULL;
int64_t tt_size = 1 << 20;                                          /* default 1M entries = 16 MB */
static inline unsigned int tt_depth(const TTEntry* e) { return e->depth_flag >> 2; }  /* bits 7..2 */
static inline unsigned int tt_flag(const TTEntry* e)  { return e->depth_flag & 3; }   /* bits 1..0 */
static inline unsigned int tt_pack(int d, int f)      { return (unsigned int)((d << 2) | f); }

/* ---------------------------------------------------------------
   UNDO HISTORY & KILLERS
   ---------------------------------------------------------------
   Idea
   When un-making a move, we must restore the exact state prior to
   its execution. However, some state transformations are destructive
   and cannot be deduced natively (e.g., losing castling rights or
   removing an en-passant square).

   Implementation
   We push destructive state factors onto a monotonic `history` stack
   prior to every move. The Zobrist hash is also vaulted, enabling
   O(1) hash restoration without scanning the board array backward.
*/

typedef struct {
    Move move; int piece_captured; int capt_slot;  int ep_square_prev; unsigned int castle_rights_prev; int halfmove_clock_prev; HASH hash_prev; int in_check;
} State;

State history[1024];

enum { MAX_PLY = 64 };
Move killers[MAX_PLY][2];

/* ---------------------------------------------------------------
   PRINCIPAL VARIATION TABLE
   ---------------------------------------------------------------
   Idea
   The PV establishes the engine's "planned game." It tracks the best
   expected continuous line of moves from the root down to the leaf.
   This offers heuristic intuition for move ordering and exposes the
   engine's internal contemplation to the external GUI.

   Implementation
   We utilize a triangular scalar array layout. At search ply P, the PV
   covers subsequent positions strictly belonging to P through the leaf.

   When a new best move is found at ply P:
       pv[P][P] = best_move
       copy pv[P+1][P+1..] into pv[P][P+1..]

   By ascending through the recursion stack, the deepest best-move
   evaluations automatically write themselves to the 0th element array.
*/

Move pv[MAX_PLY][MAX_PLY];
int  pv_length[MAX_PLY];

/* HISTORY TABLE
   hist[from][to] holds a score in [-16000, 16000]:
     +bonus (depth^2) each time from->to causes a beta cutoff (bonus).
     -bonus for every other quiet move searched before that cutoff (malus).
   Indexed by both squares so Nf3 and Bf3 never share a bucket.
   Reset at the start of each search_root call.                          */
int hist[128][128];

/* LMR REDUCTION TABLE
   lmr_table[depth][move_number] = reduction R, precomputed from
   R = round(ln(depth) * ln(move_number) / 1.6), clamped to [1,5].
   Captures, promotions, and check-giving moves bypass LMR entirely. */
int lmr_table[32][64];

/* TIME MANAGEMENT GLOBALS */
clock_t t_start;
int time_over_flag = 0;

/* ===============================================================
   S2  BOARD STATE
   ===============================================================

   Idea
   The physical chessboard implies boundary limitations. Mapping an 8x8
   chessboard to a 1D array requires bounds checking to prevent pieces
   from sliding off the board horizontally or vertically.

   Implementation (The 0x88 Method)
   Instead of an 8x8 array (64 indices), we allocate a 16x8 array
   (128 indices). The left 8 columns belong to the actual board. The
   right 8 columns serve as phantom padding.

   Any valid square has rank 0..7 and file 0..7, so bits 3 and 7
   (the 0x88 mask) are always clear. Any out-of-range index will
   have at least one of those bits set:

       (sq & 0x88) != 0  ->  off the board
*/

static inline int sq_is_off(int sq) { return sq & 0x88; }
#define FOR_EACH_SQ(sq) for(sq=0; sq<128; sq++) if(sq_is_off(sq)) sq+=7; else
int board[128], side, xside, ep_square, ply, halfmove_clock;
unsigned int castle_rights;    /* bits: 1=WO-O  2=WO-O-O  4=BO-O  8=BO-O-O */
int count[2][7];               /* count[color][piece_type], piece_type 1..6  */
HASH hash_key;

int64_t nodes_searched; int root_depth; int best_root_move; /* search telemetry, reported in UCI info lines */

/* Time control -- set by the go command handler before calling search_root.
   time_budget_ms = milliseconds we are allowed to spend on this move.
   0 means no time limit: search_root respects only max_depth.
   search_root checks the clock after each completed depth iteration and
   stops early if the elapsed time exceeds the budget. */
int64_t time_budget_ms;

/* root_ply: value of ply when search_root() was called.
   Used by the repetition detector to distinguish in-tree positions
   (where a single prior occurrence is sufficient to claim draw) from
   game-history positions (which require two prior occurrences).     */
int root_ply;

/* ---------------------------------------------------------------
   PIECE LIST
   ---------------------------------------------------------------
   Idea
   Iterating the full 128-square board to find pieces is wasteful
   since at most 32 squares are ever occupied. A compact piece list
   lets the evaluator and other routines visit only live pieces.

   Implementation
   Two parallel arrays (list_piece[], list_square[]) hold piece type
   and square for each of the up to 32 pieces. White occupies slots
   0-15 and Black occupies slots 16-31. list_index[sq] maps a board
   square back to its slot in O(1), enabling fast removal on captures.
   list_count[color] tracks how many pieces each side has.

   set_list() rebuilds the list from scratch (called after parse_fen).
   make_move / undo_move update the list incrementally via list_index.
*/

enum { LIST_OFF = 0x88 };

int list_piece[32];
int list_square[32];
int list_index[128];
int list_count[2];

/* Square helpers -- defined here so set_list and all later code can use them */
static inline int piece_on(int sq)  { return board[sq]; }
static inline int is_empty(int sq)  { return board[sq] == EMPTY; }
static inline int piece_is(int sq, int c, int t) { return board[sq] == make_piece(c, t); }
/* WARNING: color_on and ptype_on assume a non-empty square -- always
   guard with !is_empty(sq) or piece_on(sq) before calling these.    */
static inline int color_on(int sq)  { return piece_color(board[sq]); }
static inline int ptype_on(int sq)  { return piece_type(board[sq]); }
static inline int king_sq(int color) { return list_square[(color == WHITE ? 0 : 16) + list_count[color] - 1]; }
static inline int list_slot_color(int i) { return (i < 16) ? WHITE : BLACK; }
static inline void list_set_sq(int slot, int sq) { list_square[slot] = sq; list_index[sq] = slot; }
static inline void list_remove(int slot, int sq) { list_square[slot] = LIST_OFF; list_index[sq] = -1; }

static void set_list(void) {
    int pt, sq;
    for (int i = 0; i < 32; i++) { list_piece[i] = EMPTY; list_square[i] = LIST_OFF; }
    for (sq = 0; sq < 128; sq++) list_index[sq] = -1;
    list_count[WHITE] = list_count[BLACK] = 0;
    for (pt = PAWN; pt <= KING; pt++) {
        FOR_EACH_SQ(sq) {
            int p = piece_on(sq); if (!p || piece_type(p) != pt) continue;
            int color = piece_color(p);
            int slot = (color == WHITE ? 0 : 16) + list_count[color]++;
            list_piece[slot] = pt; list_square[slot] = sq; list_index[sq] = slot;
        }
    }
}

/* ===============================================================
   S3  DIRECTION & CASTLING DATA
   ===============================================================

   Idea
   Hard-coding piece direction vectors as a flat array means the move
   generator and attack detector can share the same data without any
   per-call coordinate arithmetic.

   Implementation
   One rank step = +/-16, one file step = +/-1 on the 0x88 grid.
   Knights, bishops, rooks, and the king occupy contiguous slices of
   `step_dir`, delimited by `piece_offsets[]` and `piece_limits[]`.

   Castling data lives in four parallel arrays indexed 0-3
   (White O-O, White O-O-O, Black O-O, Black O-O-O). The move
   generator checks all four entries against the current rights bits.
*/

int step_dir[] = {
    0,0,0,0,                        /* padding: aligns with piece enum       */
    -33,-31,-18,-14,14,18,31,33,    /* Knight  (idx 4-11)                    */
    -17,-15, 15, 17,                /* Bishop  (idx 12-15)                   */
    -16, -1,  1, 16,                /* Rook    (idx 16-19)                   */
    -17,-16,-15,-1,1,15,16,17       /* King    (idx 20-27)                   */
};
int piece_offsets[] = { 0,0, 4,12,16,12,20 };
int piece_limits[] = { 0,0,12,16,20,20,28 };

/* Castling move data: index 0-1 = White, 2-3 = Black */
static const int castle_kf[] = { 4, 4, 116, 116 }, castle_kt[] = { 6, 2, 118, 114 };
static const int castle_rf[] = { 7, 0, 119, 112 }, castle_rt[] = { 5, 3, 117, 115 };
static const int castle_col[] = { WHITE, WHITE, BLACK, BLACK };
static const unsigned int castle_kmask[] = { ~3u, ~3u, ~12u, ~12u }; /* Rights stripped when king moves */
static const int cr_sq[] = { 0, 7, 112, 119 };
static const unsigned int cr_mask[] = { ~2u, ~1u, ~8u, ~4u }; /* Corner squares */

/* ===============================================================
   S4  ZOBRIST HASHING
   ===============================================================

   Idea
   Fast position comparison requires a mathematical fingerprint. By
   assigning a random 64-bit integer to every possible piece-square
   combination (along with side-to-move, en-passant, and castling
   rights), we can XOR all active elements together to generate a
   near-unique position key.

   Implementation
   Because XOR is self-inverse (A ^ B ^ B = A), adding or removing a
   piece uses the identical bitwise operation:
       hash ^= zobrist_piece[color][type][sq]

   The hash incrementally updates during `make_move`. Restoring the
   hash in `undo_move` requires O(1) complexity using the historical
   record.
*/

HASH         zobrist_piece[2][7][128];
HASH         zobrist_side;
HASH         zobrist_ep[128];
HASH         zobrist_castle[16];

/* xorshift64* PRNG */
static HASH rand64(void) {
    static HASH s = 1070372631ULL;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

void init_zobrist(void) {
    for (int c = 0; c < 2; c++) for (int p = 0; p < 7; p++) for (int s = 0; s < 128; s++)
        zobrist_piece[c][p][s] = rand64();
    zobrist_side = rand64();
    for (int s = 0; s < 128; s++) zobrist_ep[s] = rand64();
    for (int s = 0; s < 16; s++) zobrist_castle[s] = rand64();
}

/* Populate lmr_table once at startup; called from uci_init.
   Index 0 is unused (depth=0 or move=0 never reach LMR). */

HASH generate_hash(void) {
    HASH h = 0; int sq;
    FOR_EACH_SQ(sq) if (piece_on(sq)) h ^= zobrist_piece[color_on(sq)][ptype_on(sq)][sq];
    if (side == BLACK)        h ^= zobrist_side;
    if (ep_square != SQ_NONE) h ^= zobrist_ep[ep_square];
    return h ^ zobrist_castle[castle_rights];
}

/* ===============================================================
   S5  ATTACK DETECTION
   ===============================================================

   Idea
   To determine if a square is attacked, iterating through every
   enemy piece and generating their moves is wildly inefficient.
   Instead, we reverse the perspective: fire ray-traces outward
   from the target square and check if a capable enemy intercepts it.

   Implementation
   Using the `step_dir` array (S3), we fire ray-traces outward from `sq`.
   Non-sliding pieces (pawns, knights, king) are checked with direct square
   lookups. Sliders (bishop, rook, queen) use a while-loop that walks the
   ray until it hits a piece or the board edge.  Separating leapers from
   sliders removes the leaper-break branch from the hot slider inner loop.
*/

static inline int is_square_attacked(int sq, int ac) {
    int tgt;
    /* Pawn attacks */
    if (ac == WHITE) {
        tgt = sq - 17; if (!sq_is_off(tgt) && board[tgt] == WP) return 1;
        tgt = sq - 15; if (!sq_is_off(tgt) && board[tgt] == WP) return 1;
    } else {
        tgt = sq + 15; if (!sq_is_off(tgt) && board[tgt] == BP) return 1;
        tgt = sq + 17; if (!sq_is_off(tgt) && board[tgt] == BP) return 1;
    }
    /* Knight attacks: eight L-shaped jumps */
    for (int i = piece_offsets[KNIGHT]; i < piece_limits[KNIGHT]; i++)
        if (!sq_is_off(sq + step_dir[i]) && board[sq + step_dir[i]] == make_piece(ac, KNIGHT)) return 1;
    /* Slider attacks: bishops, rooks, queens.
       piece_offsets[BISHOP]..piece_limits[QUEEN] covers all 8 ray directions.
       Diagonal rays [BISHOP range] match bishops/queens; orthogonal [ROOK range] match rooks/queens. */
    for (int i = piece_offsets[BISHOP]; i < piece_limits[QUEEN]; i++) {
        int step = step_dir[i], tgt = sq + step;
        while (!sq_is_off(tgt)) {
            int p = piece_on(tgt);
            if (p) {
                if (piece_color(p) == ac) {
                    int pt = piece_type(p);
                    /* The direction index i tells us which piece types
                       can attack along this particular ray*/
                    if (i >= piece_offsets[BISHOP] && pt == QUEEN) return 1;
                    if (i >= piece_offsets[BISHOP] && i < piece_limits[BISHOP] && pt == BISHOP) return 1;
                    if (i >= piece_offsets[ROOK] && i < piece_limits[ROOK] && pt == ROOK) return 1;
                }
                break; /* A piece blocked the ray */
            }
            tgt += step;
        }
    }
    /* King attacks: eight one-step directions */
    for (int i = piece_offsets[KING]; i < piece_limits[KING]; i++)
        if (!sq_is_off(sq + step_dir[i]) && board[sq + step_dir[i]] == make_piece(ac, KING)) return 1;
    return 0;
}

/* ===============================================================
   S6  MAKE / UNDO MOVE
   ===============================================================

   Idea
   Moving a piece alters the board state incrementally. To prevent
   expensive full-board copies, `make_move` executes the move in-place
   while caching irreversible details (castling rights, en-passant)
   onto the `history` stack.

   Implementation
   1. Snapshot irreversible state.
   2. Execute the primary piece transfer (from-square to to-square).
   3. Update the Zobrist hash sequentially.
   4. Process special cases (en-passant, promotions, and table-driven castling).

   The `undo_move` function identically reverses this process, reading
   the `history` stack to repair the destructive state perfectly.
*/

/* Convenience attack functions.
   in_check(s)  -- is side s's king currently in check?
   is_illegal() -- after make_move (side/xside swapped), did the mover
                   leave their own king in check? */
static inline int in_check(int s) { return is_square_attacked(king_sq(s), s ^ 1); }
static inline int is_illegal(void) { return is_square_attacked(king_sq(xside), side); }

static inline void add_move(Move* list, int* n, int fr, int to, int pr) { list[(*n)++] = make_move_enc(fr, to, pr); }

/* Add all four promotion possibilities for a pawn move from f to t. */
static void add_promo(Move* list, int* n, int fr, int to) {
    for (int pr = QUEEN; pr >= KNIGHT; pr--) add_move(list, n, fr, to, pr);
}

/* XOR piece (color c, type p) at sq in or out of the running Zobrist hash.
   Because XOR is self-inverse, toggling the same value twice cancels out,
   which is exactly what make_move uses when it moves a piece from f to t:
       toggle(side, pt, f)  -- remove from source
       toggle(side, pt, t)  -- place at destination                        */
static inline void toggle(int c, int p, int sq) { hash_key ^= zobrist_piece[c][p][sq]; }

void make_move(Move m) {

    /* Setting the variables */
    int fr = move_from(m), to = move_to(m), pr = move_promo(m), p = piece_on(fr), pt = piece_type(p), cap = piece_on(to);

    /* Saving move data for unmaking */
    history[ply].move = m; history[ply].piece_captured = cap; history[ply].ep_square_prev = ep_square;
    history[ply].castle_rights_prev = castle_rights; history[ply].halfmove_clock_prev = halfmove_clock; history[ply].hash_prev = hash_key;
    halfmove_clock = (pt == PAWN || cap) ? 0 : halfmove_clock + 1;
    history[ply].capt_slot = -1;

    /* En passant capture */
    if (pt == PAWN && to == ep_square) {
        int ep_pawn = to + (side == WHITE ? -16 : 16);
        history[ply].piece_captured = piece_on(ep_pawn);
        history[ply].capt_slot = list_index[ep_pawn];
        list_square[history[ply].capt_slot] = LIST_OFF; list_index[ep_pawn] = -1;
        board[ep_pawn] = EMPTY;
        toggle(xside, PAWN, ep_pawn);
        count[xside][PAWN]--;
    }

    /* Normal capture */
    if (cap) {
        history[ply].capt_slot = list_index[to];
        list_remove(list_index[to], to);
        toggle(xside, piece_type(cap), to); count[xside][piece_type(cap)]--;
    }

    /* Move piece */
    list_set_sq(list_index[fr], to); list_index[fr] = -1;
    board[to] = p; board[fr] = EMPTY;
    toggle(side, pt, fr); toggle(side, pt, to);

    /* Promotion */
    if (pr) {
        int slot = list_index[to]; list_piece[slot] = pr;
        board[to] = make_piece(side, pr); toggle(side, pt, to); toggle(side, pr, to);
        count[side][PAWN]--; count[side][pr]++;
    }

    /* Castling */
    hash_key ^= zobrist_castle[castle_rights];
    if (pt == KING) {
        for (int ci = 0; ci < 4; ci++) {
            if (fr == castle_kf[ci] && to == castle_kt[ci]) {
                int rook_from = castle_rf[ci]; int rook_to = castle_rt[ci]; int rook_slot = list_index[rook_from];
                board[rook_from] = EMPTY; board[rook_to] = make_piece(castle_col[ci], ROOK);
                list_set_sq(rook_slot, rook_to); list_index[rook_from] = -1;
                toggle(castle_col[ci], ROOK, rook_from);  toggle(castle_col[ci], ROOK, rook_to);
                break;
            }
        }
        castle_rights &= castle_kmask[side * 2]; /* WHITE=0->index 0, BLACK=1->index 2 */
    }
    for (int ci = 0; ci < 4; ci++) if (fr == cr_sq[ci] || to == cr_sq[ci]) castle_rights &= cr_mask[ci]; /* Strip castling */
    hash_key ^= zobrist_castle[castle_rights];

    /* Setting en passant square */
    if (ep_square != SQ_NONE) hash_key ^= zobrist_ep[ep_square];
    ep_square = SQ_NONE;
    if (pt == PAWN && ((to - fr) == 32 || (fr - to) == 32)) { ep_square = fr + (side == WHITE ? 16 : -16); hash_key ^= zobrist_ep[ep_square]; }

    /* Changing side to move */
    hash_key ^= zobrist_side; side ^= 1; xside ^= 1;
    history[ply].in_check = is_square_attacked(king_sq(side), xside);
    ply++;
}

void undo_move(void) {
    ply--; side ^= 1; xside ^= 1;
    Move m = history[ply].move; int fr = move_from(m), to = move_to(m), pr = move_promo(m);
    int cap = history[ply].piece_captured;

    /* Move piece back: to -> fr */
    list_set_sq(list_index[to], fr); list_index[to] = -1;
    board[fr] = board[to]; board[to] = cap;

    /* Undo promotion */
    if (pr) {
        int slot = list_index[fr]; list_piece[slot] = PAWN;
        board[fr] = make_piece(side, PAWN); count[side][pr]--; count[side][PAWN]++;
    }
    int pt = ptype_on(fr);

    /* Undo en passant */
    if (pt == PAWN && to == history[ply].ep_square_prev) {
        int ep_pawn = to + (side == WHITE ? -16 : 16);
        board[to] = EMPTY;
        board[ep_pawn] = cap;
        if (cap) list_set_sq(history[ply].capt_slot, ep_pawn);
        count[xside][PAWN]++;
    /* Undo normal capture */
    } else if (cap) {
        list_set_sq(history[ply].capt_slot, to);
        count[xside][piece_type(cap)]++;
    }

    /* Undo castling */
    if (pt == KING) {
        for (int ci = 0; ci < 4; ci++) {
            if (fr == castle_kf[ci] && to == castle_kt[ci]) {
                int rook_from = castle_rt[ci]; int rook_to = castle_rf[ci]; int rook_slot = list_index[rook_from];
                board[rook_from] = EMPTY; board[rook_to] = make_piece(castle_col[ci], ROOK);
                list_set_sq(rook_slot, rook_to); list_index[rook_from] = -1;
                break;
            }
        }
    }

    /* Restore irreversible state */
    ep_square = history[ply].ep_square_prev; castle_rights = history[ply].castle_rights_prev;
    halfmove_clock = history[ply].halfmove_clock_prev;
    hash_key = history[ply].hash_prev;
}

/* ===============================================================
   S7  MOVE GENERATION
   ===============================================================

   Idea
   Full legal-move generation requires a check test for every candidate,
   which is expensive.  Instead, we generate pseudo-legal moves
   (geometrically valid but possibly leaving the king in check) and
   discard illegal ones inside the search loop after `make_move`.

   Implementation
   The generator is unified: `caps_only=1` restricts output to captures
   and promotions, which is exactly what quiescence search needs.

   1. Pawns: handled separately -- direction, double-push, and
      en-passant all depend on colour.
   2. Sliders & leapers: iterated via `step_dir` ray traces.
   3. Castling: only generated when `caps_only=0`; verified by
      checking that the path is clear and unattacked.
*/

int generate_moves(Move* moves, int caps_only) {
    int cnt = 0;
    int d_pawn = (side == WHITE) ? 16 : -16;
    int pawn_start = (side == WHITE) ? 1 : 6;
    int pawn_promo = (side == WHITE) ? 6 : 1;

    for (int slot = (side == WHITE ? 0 : 16); slot < (side == WHITE ? 16 : 32); slot++) {
        int sq = list_square[slot];
        if (sq == LIST_OFF) continue;
        int pt = list_piece[slot];

        /* -- Pawns ------------------------------------------------ */
        if (pt == PAWN) {
            int tgt = sq + d_pawn;
            if (!sq_is_off(tgt) && is_empty(tgt)) {
                if ((sq >> 4) == pawn_promo) add_promo(moves, &cnt, sq, tgt);
                else if (!caps_only) {
                    add_move(moves, &cnt, sq, tgt, 0);
                    if ((sq >> 4) == pawn_start && is_empty(tgt + d_pawn)) add_move(moves, &cnt, sq, tgt + d_pawn, 0);
                }
            }
            for (int i = -1; i <= 1; i += 2) {           /* diagonal captures + ep */
                tgt = sq + d_pawn + i;
                if (!sq_is_off(tgt) && ((!is_empty(tgt) && color_on(tgt) == xside) || tgt == ep_square)) {
                    if ((sq >> 4) == pawn_promo) add_promo(moves, &cnt, sq, tgt);
                    else add_move(moves, &cnt, sq, tgt, 0);
                }
            }
            continue;
        }

        /* -- Leapers (knight / king): single step, no ray continuation -- */
        if (pt == KNIGHT || pt == KING) {
            for (int i = piece_offsets[pt]; i < piece_limits[pt]; i++) {
                int tgt = sq + step_dir[i]; if (sq_is_off(tgt)) continue;
                if (is_empty(tgt)) { if (!caps_only) add_move(moves, &cnt, sq, tgt, 0); }
                else if (color_on(tgt) == xside) add_move(moves, &cnt, sq, tgt, 0);
            }
        } else {
        /* -- Sliders (bishop / rook / queen): ray trace ----------- */
            for (int i = piece_offsets[pt]; i < piece_limits[pt]; i++) {
                int step = step_dir[i], tgt = sq + step;
                while (!sq_is_off(tgt)) {
                    if (is_empty(tgt)) {
                        if (!caps_only) add_move(moves, &cnt, sq, tgt, 0);
                    } else {
                        if (color_on(tgt) == xside) add_move(moves, &cnt, sq, tgt, 0);
                        break;
                    }
                    tgt += step;
                }
            }
        }

        /* -- Castling (king only, never in caps_only mode) -------- */
        if (pt == KING && !caps_only) {
            int kf, kt, rf, bit, ac, clear_ok;
            for (int ci = 0; ci < 4; ci++) {
                kf = castle_kf[ci]; kt = castle_kt[ci]; rf = castle_rf[ci];
                bit = (ci == 0) ? 1 : (ci == 1) ? 2 : (ci == 2) ? 4 : 8;
                ac = (castle_col[ci] == WHITE) ? BLACK : WHITE;

                if (sq != kf || castle_col[ci] != side) continue;
                if (!(castle_rights & (unsigned int)bit)) continue;
                if (piece_on(rf) != make_piece(side, ROOK)) continue;

                /* Every square between king and rook must be empty */
                int sq1 = (kf < rf) ? kf + 1 : rf + 1, sq2 = (kf < rf) ? rf : kf;
                clear_ok = 1;
                for (int sq3 = sq1; sq3 < sq2; sq3++)
                    if (!is_empty(sq3)) { clear_ok = 0; break; }
                if (!clear_ok) continue;

                /* King's path must not traverse attacked squares */
                int step2 = (kt > kf) ? 1 : -1;
                clear_ok = 1;
                for (int sq3 = kf; sq3 != (kt + step2); sq3 += step2)
                    if (is_square_attacked(sq3, ac)) { clear_ok = 0; break; }
                if (clear_ok) add_move(moves, &cnt, kf, kt, 0);
            }
        }
    }
    return cnt;
}

/* Capture-only move generator for quiescence search.
   Avoids all quiet-move branches and the entire castling block. */
int generate_captures(Move* moves) {
    int cnt = 0;
    int d_pawn = (side == WHITE) ? 16 : -16;
    int pawn_promo = (side == WHITE) ? 6 : 1;

    for (int slot = (side == WHITE ? 0 : 16); slot < (side == WHITE ? 16 : 32); slot++) {
        int sq = list_square[slot];
        if (sq == LIST_OFF) continue;
        int pt = list_piece[slot];

        if (pt == PAWN) {
            int tgt = sq + d_pawn;                                  /* quiet promo */
            if ((sq >> 4) == pawn_promo && !sq_is_off(tgt) && is_empty(tgt))
                add_promo(moves, &cnt, sq, tgt);
            for (int i = -1; i <= 1; i += 2) {                     /* diagonal captures + ep */
                tgt = sq + d_pawn + i;
                if (!sq_is_off(tgt) && ((!is_empty(tgt) && color_on(tgt) == xside) || tgt == ep_square)) {
                    if ((sq >> 4) == pawn_promo) add_promo(moves, &cnt, sq, tgt);
                    else add_move(moves, &cnt, sq, tgt, 0);
                }
            }
            continue;
        }

        if (pt == KNIGHT || pt == KING) {
            for (int i = piece_offsets[pt]; i < piece_limits[pt]; i++) {
                int tgt = sq + step_dir[i];
                if (!sq_is_off(tgt) && !is_empty(tgt) && color_on(tgt) == xside) add_move(moves, &cnt, sq, tgt, 0);
            }
        } else {
            for (int i = piece_offsets[pt]; i < piece_limits[pt]; i++) {
                int step = step_dir[i], tgt = sq + step;
                while (!sq_is_off(tgt) && is_empty(tgt)) tgt += step;
                if (!sq_is_off(tgt) && color_on(tgt) == xside) add_move(moves, &cnt, sq, tgt, 0);
            }
        }
    }
    return cnt;
}

/* ===============================================================
   S8  FEN PARSER
   ===============================================================

   Idea
   Forsyth-Edwards Notation (FEN) is the standard ASCII string format
   for representing a distinct board state.

   Implementation
   The parser reads the space-delimited fields sequentially:
   1. Piece placement (ranks 8 down to 1).
   2. Side to move ('w' or 'b').
   3. Castling rights ('K', 'Q', 'k', 'q').
   4. En-passant target square.
*/

static int char_to_piece(char lo) {
    const char* m = "pnbrqk"; /* PAWN=1..KING=6 match i+1 */
    for (int i = 0; i < 6; i++) if (lo == m[i]) return i + 1;
    return EMPTY;
}

void parse_fen(const char* fen) {
    int rank = 7, file = 0;

    for (int i = 0; i < 128; i++) board[i] = EMPTY;
    castle_rights = 0; ep_square = SQ_NONE; ply = 0; hash_key = 0;
    memset(count, 0, sizeof(count));
    memset(killers, 0, sizeof(killers)); memset(pv, 0, sizeof(pv));
    memset(pv_length, 0, sizeof(pv_length)); memset(hist, 0, sizeof(hist));

    while (*fen && *fen != ' ') {
        if (*fen == '/') { file = 0; rank--; }
        else if (isdigit(*fen)) { file += *fen - '0'; }
        else {
            int sq = rank * 16 + file, color = isupper(*fen) ? WHITE : BLACK; char lo = (char)tolower(*fen);
            int piece = char_to_piece(lo);
            if (piece == EMPTY) { fen++; continue; }
            board[sq] = make_piece(color, piece);
            count[color][piece]++;
            file++;
        }
        fen++;
    }
    if (*fen) fen++;

    side = (*fen == 'w') ? WHITE : BLACK; xside = side ^ 1;
    if (*fen) fen++;
    if (*fen) fen++;

    while (*fen && *fen != ' ') {
        if (*fen == 'K') { castle_rights |= 1; } if (*fen == 'Q') { castle_rights |= 2; }
        if (*fen == 'k') { castle_rights |= 4; } if (*fen == 'q') { castle_rights |= 8; }
        fen++;
    }
    if (*fen) fen++;

    if (*fen != '-' && *fen && fen[1])
        ep_square = (fen[1] - '1') * 16 + (fen[0] - 'a');

    /* advance past ep field, then read halfmove clock */
    while (*fen && *fen != ' ') fen++;
    halfmove_clock = 0;
    if (*fen == ' ') { fen++; halfmove_clock = atoi(fen); }

    set_list();
}

/* ===============================================================
   S9  EVALUATION
   ===============================================================

   Idea
   A static evaluator scores the position in centipawns from the
   side-to-move's perspective.  Raw material counting ignores piece
   activity, so positional bonuses and penalties are layered on top.

   Implementation (PeSTO tapered evaluation)
   1. Material + PST: separate middlegame (MG) and endgame (EG) values
      from Rofchade's Texel-tuned PeSTO tables.  Each piece accumulates
      into mg[color] and eg[color] arrays independently.
   2. Phase: each non-pawn piece type contributes to a 0-24 phase
      counter (knight=1, bishop=1, rook=2, queen=4, max=24).
      phase=24 is a full middlegame; phase=0 is a pure endgame.
   3. Taper: the final score blends MG and EG smoothly:
         (mg_score * phase + eg_score * (24 - phase)) / 24
      This replaces the old single-score + MAX_PHASE approach and
      correctly handles all pieces (including the king) in one pass.
   4. Mobility: centered around typical values so inactive pieces are
      penalised rather than all pieces receiving a flat bonus.
   5. Pawn structure: doubled and isolated pawns penalised in both
      MG and EG.  Passed pawns are NOT added explicitly -- PeSTO's EG
      pawn table already encodes their value; double-counting hurts.
   6. Pawn shield: MG-only, since king centralisation in the endgame
      is handled by the EG king PST directly.
   7. Rook activity: open/semi-open file and 7th-rank bonuses applied
      to both MG and EG.
*/

/*
   PeSTO / Rofchade Texel-tuned piece-square tables.
   Indexed [piece-1][sq] where piece: 0=pawn..5=king.
   Square index: rank*8+file, rank 0 = White's back rank (rank 1),
   rank 7 = rank 8.  CPW tables (rank 8 first) are vertically flipped.
   Black uses (7-rank)*8+file to mirror vertically.
*/

/* mg_pst[piece-1][sq]: middlegame, 16 vals/line = one rank pair, rank 1 first */
/* Readable version kept as comments */
// static int mg_pst[6][64] = {
//  {  82,   82,   82,   82,   82,   82,   82,   82,      47,  76,    57,   60,   67,  100,  107,   56,  /* pawn   r1-r2 */
//     56,   71,   78,   74,   87,   87,  104,   70,      53,  77,    78,   96,   99,   88,   90,   55,  /*        r3-r4 */
//     66,   94,   90,  105,  107,   96,  100,   57,      76,  90,   101,  105,  120,  141,  108,   63,  /*        r5-r6 */
//    162,  158,  131,  136,  132,  139,  108,   79,      82,  82,    82,   82,   82,   82,   82,   82}, /*        r7-r8 */

//  { 231,  318,  281,  306,  322,  311,  317,  315,     310,  286,  327,  336,  338,  357,  325,  320,  /* knight r1-r2 */
//    312,  330,  347,  349,  358,  356,  364,  319,     325,  343,  353,  349,  367,  357,  360,  330,  /*        r3-r4 */
//    330,  355,  355,  392,  372,  407,  354,  360,     290,  398,  374,  400,  422,  465,  411,  379,  /*        r5-r6 */
//    266,  297,  411,  374,  360,  401,  342,  322,     172,  250,  305,  290,  400,  241,  322,  232}, /*       r7-r8 */

//  { 333,  364,  353,  346,  354,  351,  326,  344,     370,  384,  382,  365,  374,  388,  401,  367,  /* bishop r1-r2 */
//    363,  382,  380,  378,  377,  393,  384,  373,     361,  380,  376,  392,  398,  375,  374,  370,  /*        r3-r4 */
//    362,  368,  384,  417,  400,  400,  370,  362,     347,  403,  408,  403,  400,  417,  402,  361,  /*        r5-r6 */
//    339,  382,  348,  353,  397,  425,  385,  318,     338,  370,  283,  329,  342,  325,  373,  358}, /*        r7-r8 */

//  { 460,  466,  479,  492,  491,  486,  438,  452,     434,  462,  459,  467,  476,  490,  473,  405,  /* rook   r1-r2 */
//    433,  454,  461,  460,  478,  479,  474,  443,     439,  453,  464,  474,  486,  471,  483,  452,  /*        r3-r4 */
//    455,  467,  482,  502,  499,  512,  469,  457,     471,  496,  501,  511,  492,  523,  538,  493,  /*        r5-r6 */
//    502,  507,  533,  537,  555,  542,  501,  519,     510,  519,  508,  526,  539,  488,  510,  522}, /*        r7-r8 */
 
//  {1023, 1008, 1018, 1037, 1012, 1002,  996,  976,     991, 1016, 1036, 1029, 1035, 1042, 1024, 1028,  /* queen  r1-r2 */
//   1009, 1025, 1012, 1021, 1018, 1025, 1038, 1030,    1014,  997, 1014, 1013, 1021, 1019, 1026, 1020,  /*        r3-r4 */
//    996,  996, 1007, 1007, 1022, 1040, 1022, 1024,    1014, 1006, 1030, 1031, 1054, 1083, 1072, 1082,  /*        r5-r6 */
//   1002,  984, 1020, 1028, 1008, 1084, 1054, 1081,     999, 1026, 1056, 1038, 1086, 1071, 1070, 1072}, /*        r7-r8 */

//  { -17,   36,   14,  -56,    6,  -26,   26,   12,       1,    8,   -6,  -66,  -45,  -14,   11,    7,  /* king   r1-r2 */
//    -13,  -12,  -20,  -48,  -46,  -28,  -13,  -25,     -48,    1,  -25,  -41,  -48,  -42,  -32,  -53,  /*        r3-r4 */
//    -16,  -18,  -10,  -29,  -31,  -25,  -13,  -35,      -7,   26,    4,  -17,  -22,    8,   24,  -24,  /*        r5-r6 */
//     30,    1,  -18,   -5,  -10,   -2,  -36,  -28,     -66,   24,   18,  -14,  -58,  -32,    3,   13}  /*        r7-r8 */
// };

/* eg_pst[piece-1][sq]: endgame, same layout */
// static int eg_pst[6][64] = {
//   { 94,   94,   94,   94,   94,   94,   94,   94,    108,  100,  102,  102,  106,   92,   94,   85,  /* pawn   r1-r2 */
//     96,   99,   86,   94,   94,   89,   91,   84,    105,  101,   89,   85,   85,   84,   95,   91,  /*        r3-r4 */
//    124,  116,  105,   97,   90,   96,  109,  109,    166,  163,  140,  119,  118,  125,  146,  156,  /*        r5-r6 */
//    189,  186,  180,  156,  159,  182,  187,  218,     94,   94,   94,   94,   94,   94,   94,   94}, /*        r7-r8 */

//   {254,  232,  260,  268,  261,  265,  233,  219,    241,  263,  273,  278,  281,  263,  260,  239,  /* knight r1-r2 */
//    260,  279,  280,  297,  293,  279,  263,  261,    265,  277,  299,  308,  298,  298,  287,  265,  /*        r3-r4 */
//    266,  286,  305,  305,  305,  293,  291,  261,    259,  263,  290,  291,  278,  270,  262,  239,  /*        r5-r6 */
//    258,  275,  257,  281,  272,  254,  255,  231,    225,  245,  270,  255,  251,  256,  220,  183}, /*        r7-r8 */

//   {276,  290,  276,  294,  290,  283,  294,  282,    285,  281,  292,  298,  302,  290,  284,  271,  /* bishop r1-r2 */
//    287,  296,  307,  307,  312,  299,  292,  284,    293,  301,  311,  317,  304,  306,  295,  290,  /*        r3-r4 */
//    296,  308,  310,  306,  311,  305,  301,  301,    301,  290,  297,  297,  297,  303,  299,  303,  /*        r5-r6 */
//    291,  295,  306,  287,  295,  286,  293,  285,    285,  278,  288,  291,  292,  290,  282,  275}, /*        r7-r8 */

//   {506,  517,  518,  512,  510,  502,  519,  495,    509,  509,  515,  517,  506,  506,  504,  512,  /* rook   r1-r2 */
//    511,  515,  510,  514,  508,  503,  507,  499,    518,  520,  523,  516,  509,  509,  507,  504,  /*        r3-r4 */
//    519,  518,  528,  513,  513,  516,  513,  517,    522,  522,  520,  517,  517,  512,  510,  512,  /*        r5-r6 */
//    522,  524,  524,  522,  508,  514,  519,  514,    528,  524,  533,  526,  525,  527,  523,  520}, /*        r7-r8 */

//   {904,  910,  916,  896,  934,  906,  919,  897,    917,  916,  909,  922,  922,  916,  903,  906,  /* queen  r1-r2 */
//    923,  911,  952,  944,  947,  955,  949,  944,    920,  967,  956,  985,  967,  973,  976,  962,  /*        r3-r4 */
//    941,  960,  960,  983,  996,  977,  996,  975,    918,  943,  946,  987,  986,  974,  956,  948,  /*        r5-r6 */
//    921,  959,  970,  980,  997,  964,  969,  938,    930,  961,  960,  966,  966,  958,  949,  959}, /*        r7-r8 */
  
//   {-55,  -36,  -19,  -12,  -30,  -12,  -26,  -45,    -28,   -9,    6,   11,   12,    6,   -3,  -19,  /* king   r1-r2 */
//    -21,   -1,   13,   19,   21,   18,    9,  -10,    -19,   -3,   23,   22,   25,   25,   11,  -12,  /*        r3-r4 */
//    -10,   24,   26,   25,   24,   35,   28,    1,     10,   18,   24,   13,   18,   46,   45,   11,  /*        r5-r6 */
//    -12,   19,   16,   16,   15,   40,   25,   12,    -74,  -34,  -18,  -20,  -13,   17,    5,  -19}  /*        r7-r8 */
// };

static int mg_pst[6][64] = { {82,82,82,82,82,82,82,82, 71,72,75,75,87,100,107,63, 75,75,85,90,106,100,105,81, 75,81,94,109,113,109,97,77, 84,95,103,122,132,131,111,94, 94,112,140,140,160,177,149,103, 234,211,183,200,185,164,66,63, 82,82,82,82,82,82,82,82}, {265,320,302,322,326,338,321,293, 305,312,339,353,349,354,336,339, 319,342,358,371,380,365,363,325, 334,343,375,371,384,375,383,346, 347,363,391,415,386,412,369,379, 318,379,407,420,443,474,410,371, 313,318,385,394,387,434,335,356, 169,259,256,310,384,187,361,241}, {360,365,362,346,351,362,366,353, 375,376,382,364,374,381,398,373, 362,379,373,376,377,378,376,378, 355,366,372,394,390,365,365,373, 348,374,380,410,390,389,373,364, 355,386,380,389,416,415,412,388, 340,359,362,336,362,383,362,354, 318,299,276,267,277,252,364,341}, {461,459,463,472,476,481,475,472, 433,439,449,457,466,481,486,426, 432,434,436,450,454,458,484,458, 430,430,433,444,449,459,466,452, 440,447,455,475,475,489,485,482, 451,473,476,484,510,534,569,492, 469,467,488,503,499,545,521,533, 505,494,484,490,501,486,551,559}, {1076,1066,1083,1095,1089,1071,1059,1060, 1067,1079,1092,1092,1093,1102,1105,1085, 1061,1080,1079,1078,1085,1085,1088,1090, 1061,1058,1069,1063,1071,1075,1084,1082, 1056,1053,1053,1066,1067,1073,1084,1089, 1051,1050,1056,1050,1100,1146,1150,1106, 1045,1017,1043,1037,1046,1117,1074,1133, 1038,1051,1058,1099,1097,1138,1166,1159}, {-25,12,8,-91,12,-72,6,10, 7,-31,-38,-72,-50,-54,-10,4, -43,-38,-55,-91,-75,-68,-46,-52, -77,-26,-39,-107,-97,-72,-63,-84, -33,7,-15,-74,-74,-6,-43,-79, -30,44,30,-38,2,99,74,-21, 21,42,87,-8,22,56,24,61, 117,133,103,19,-7,22,164,146} };
static int eg_pst[6][64] = { {94,94,94,94,94,94,94,94, 125,121,120,116,119,121,109,109, 116,114,107,107,106,109,104,103, 123,118,103,94,95,104,108,108, 135,127,115,98,100,103,118,116, 168,162,143,125,115,121,134,146, 225,236,236,212,212,209,236,241, 94,94,94,94,94,94,94,94}, {276,297,340,345,342,332,313,284, 310,348,350,354,359,345,344,320, 328,358,363,379,378,360,355,337, 357,374,389,397,395,390,370,353, 359,375,391,397,401,395,382,354, 355,363,379,385,374,370,361,341, 339,369,362,378,373,342,356,330, 314,351,384,363,356,387,332,274}, {350,363,359,374,373,368,355,355, 358,357,363,370,370,362,361,346, 369,378,380,374,385,375,373,361, 376,384,385,381,376,385,381,375, 381,387,382,381,385,383,391,384, 380,382,380,379,376,384,381,382, 376,380,383,386,385,377,377,369, 382,391,391,397,395,393,381,376}, {638,639,644,639,631,636,631,620, 638,642,646,638,635,624,621,634, 644,653,651,645,643,644,631,627, 665,668,668,664,659,655,651,648, 676,673,673,667,663,663,661,657, 680,674,676,668,658,655,644,663, 680,683,678,674,674,656,665,655, 657,668,672,666,667,675,658,655}, {1132,1134,1125,1119,1117,1115,1113,1124, 1140,1141,1127,1137,1135,1117,1103,1103, 1152,1145,1169,1159,1166,1170,1166,1155, 1160,1194,1182,1211,1197,1209,1196,1203, 1177,1205,1216,1220,1236,1245,1230,1222, 1188,1204,1211,1243,1231,1224,1207,1238, 1202,1236,1245,1261,1277,1227,1240,1193, 1220,1227,1234,1214,1232,1215,1189,1189}, {-40,-27,-18,-10,-48,-4,-30,-63, -21,1,10,3,2,11,-6,-24, -12,11,22,21,20,22,8,-10, -6,11,28,35,34,35,21,2, 2,24,36,30,36,37,40,16, 15,24,28,21,10,33,41,21, -13,20,9,-3,7,29,44,-15, -111,-44,-34,-35,-22,12,-14,-70} };

static void init_lmr(void) {
    for (int d = 1; d < 32; d++)
        for (int m = 1; m < 64; m++) {
            int r = (int)round(log(d) * log(m) / 1.6);
            lmr_table[d][m] = r < 1 ? 1 : r > 5 ? 5 : r;
        }
}

/* Phase contribution per piece type (indexed by TYPE(): 1=pawn..6=king).
   knight=1, bishop=1, rook=2, queen=4; max total = 24. */
static const int phase_inc[7] = { 0, 0, 1, 1, 2, 4, 0 };

/* Piece value table for MVV-LVA move ordering (unchanged) */
static const int piece_val[7] = { 0,100,320,330,500,900,20000 };

/* Mobility centering offsets: subtract typical reachable-square count so
   inactive pieces are penalised rather than all pieces getting a flat bonus.
   Indexed by TYPE(): 0=empty,1=pawn,2=knight,3=bishop,4=rook,5=queen,6=king */
static const int mob_center[7] = { 0, 0, 4, 6, 6, 13, 0 };
static const int mob_step_mg[7] = { 0, 0, 0, 5, 3, 2, 0 };
static const int mob_step_eg[7] = { 0, 0, 1, 5, 3, 5, 0 };

static inline int max(int a, int b) { return a > b ? a : b; }
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int distance(int s1, int s2) { return max(abs((s1 & 7) - (s2 & 7)), abs((s1 >> 4) - (s2 >> 4))); }
static inline void add_score(int* mg, int* eg, int color, int mg_v, int eg_v) { mg[color] += mg_v; eg[color] += eg_v; }

int evaluate(void) {
    int mg[2], eg[2], phase;
    int lowest_pawn_rank[2][8];

    /* Combined list of pawns and rooks -- the only pieces needing a second pass.
       Pawns need full pawn structure info; rooks need open-file data.
       All other pieces are fully scored in the first pass.                       */
    int pr_list[32], pr_index = 0, i;

    mg[WHITE] = mg[BLACK] = eg[WHITE] = eg[BLACK] = phase = 0;
    for (int i = 0; i < 8; i++) lowest_pawn_rank[WHITE][i] = lowest_pawn_rank[BLACK][i] = 7;

    /* First pass: material, PST, phase, mobility.
       Pawns and rooks are also recorded into pr_list for the second pass. */
    for (int slot = 0; slot < 32; slot++) {
        int sq = list_square[slot];
        if (sq == LIST_OFF) continue;

        int pt = list_piece[slot], color = list_slot_color(slot);
        int rank = sq >> 4, f = sq & 7;

        if (pt == PAWN) {
            int own_rank = (color == WHITE) ? rank : (7 - rank);
            if (own_rank < lowest_pawn_rank[color][f])
                lowest_pawn_rank[color][f] = own_rank;
            pr_list[pr_index++] = sq;
        } else if (pt == ROOK) {
            pr_list[pr_index++] = sq;
        }

        /* Square index: rank 0 = White's back rank.
           Black mirrors vertically so its rank 0 is rank 7 in White terms. */
        int idx = (color == WHITE) ? rank * 8 + f : (7 - rank) * 8 + f;

        /* Material + PST: scored into MG and EG accumulators separately.
           pt-1 converts TYPE() (1-based) to the 0-based table index. */
        add_score(mg, eg, color, mg_pst[pt - 1][idx], eg_pst[pt - 1][idx]);
        phase += phase_inc[pt];

        /* Mobility: count pseudo-legal reachable squares, centered so that
           a piece with exactly mob_center[pt] squares scores zero.
           Pinned pieces appear more mobile than they are, but the
           approximation is cheap and consistently directional. */
        if (pt >= KNIGHT && pt <= QUEEN) {
            int mob = 0;
            for (i = piece_offsets[pt]; i < piece_limits[pt]; i++) {
                int step = step_dir[i], target = sq + step;
                while (!sq_is_off(target)) {
                    if (is_empty(target)) { mob++; }
                    else { if (color_on(target) != color) mob++; break; }
                    if (pt == KNIGHT) break;
                    target += step;
                }
            }
            mob -= mob_center[pt];
            add_score(mg, eg, color, mob_step_mg[pt] * mob, mob_step_eg[pt] * mob);
        }
    }

    /* Bishop pair bonus using incremental count */
    for (int c = 0; c < 2; c++)
        if (count[c][BISHOP] >= 2) add_score(mg, eg, c, 29, 60);

    /* King pawn shield -- MG only.
           In the endgame, king centralisation is already rewarded by the
           EG king PST; a pawn shield is irrelevant and would only hurt. */

    static const int shield_val[8] = { 0, 27, 16, 4, -5, 12, 51, -21 };
    for (int color = 0; color < 2; color++) {
        int ksq = king_sq(color), kf = ksq & 7;
        if (kf <= 2 || kf >= 5) {
            int shield = 0;
            for (int ft = kf - 1; ft <= kf + 1; ft++) if (ft >= 0 && ft <= 7) {
                shield += shield_val[lowest_pawn_rank[color][ft]];
                if (lowest_pawn_rank[color][ft] == 7) shield -= 5 * (lowest_pawn_rank[color ^ 1][ft] == 7);
            }
            mg[color] += shield;
        }
    }

    /* Second pass: iterate pr_list (pawns and rooks only).
       Pawns: doubled, isolated, passed pawn scoring.
       Rooks: open/semi-open file bonus.                  */
    static const int pp_mg[8] = { 0, 7, -6, -7, 8, 40, 106, 0 };
    static const int pp_eg[8] = { 0, -14, 1, 34, 62, 98, 57, 0 };
    static const int pp_blocked_mg[8] = { 0, -7, -20, -7, 14, 61, 117, 0 };
    static const int pp_blocked_eg[8] = { 0, -4, 2, 9, 2, -7, -60, 0 };

    for (i = 0; i < pr_index; i++) {
        int sq = pr_list[i], p = piece_on(sq);
        int pt = piece_type(p), color = piece_color(p), f = sq & 7;

        if (pt == ROOK) {
            if (lowest_pawn_rank[color][f] == 7) {
                if (lowest_pawn_rank[color ^ 1][f] == 7)    add_score(mg, eg, color, 44, -3);
                else    add_score(mg, eg, color, 21, 6);
            }
            continue;
        }

        /* PAWN: doubled, isolated, passed */
        int rank = sq >> 4;
        int own_rank = (color == WHITE) ? rank : (7 - rank);
        int enemy = color ^ 1;

        /* Doubled */
        if (own_rank != lowest_pawn_rank[color][f])
            add_score(mg, eg, color, -12, -20);

        /* Passed and isolated detection */
        int passed = 1, isolated = 1;
        for (int df = -1; df <= 1; df++) {
            int ef = f + df;
            if (ef < 0 || ef > 7) continue;

            /* Passed-pawn test: enemy pawn ahead on same/adjacent file */
            if (lowest_pawn_rank[enemy][ef] != 7) {
                int enemy_front_rank = 7 - lowest_pawn_rank[enemy][ef];
                if (enemy_front_rank >= own_rank) passed = 0;
            }
            /* Isolated-pawn test: any friendly pawn on adjacent file cancels isolation */
            if (df != 0 && lowest_pawn_rank[color][ef] != 7) isolated = 0;
        }

        if (isolated) add_score(mg, eg, color, -15, -12);
        if (!passed) continue;

        /* Passed pawn bonus -- separate tables for blocked/unblocked */
        int front = sq + (color == WHITE ? 16 : -16);
        int blocked = !sq_is_off(front) && !is_empty(front) && color_on(front) == enemy;
        int bonus_mg = blocked ? pp_blocked_mg[own_rank] : pp_mg[own_rank];
        int bonus_eg = blocked ? pp_blocked_eg[own_rank] : pp_eg[own_rank];
        bonus_eg += 12 * (distance(sq, king_sq(enemy)) - distance(sq, king_sq(color)));
        add_score(mg, eg, color, bonus_mg, bonus_eg);
    }

    /* Tapered blend.
       phase clamps to [0,24]: values beyond 24 (e.g. at game start) are
       treated as full middlegame.  The interpolation formula:
           (mg_score * phase + eg_score * (24 - phase)) / 24
       gives pure MG at phase=24 and pure EG at phase=0. */
    if (phase > 24) phase = 24;
    int mg_score = mg[side] - mg[side ^ 1];
    int eg_score = eg[side] - eg[side ^ 1];
    int score = (mg_score * phase + eg_score * (24 - phase)) / 24;

    /* DRAWISH ENDGAME SCALING -- disabled, retest at 2600+ Elo
    int winner = (score > 0) ? side : (side ^ 1);
    if (count[winner][PAWN] == 0
        && count[winner][QUEEN] + count[winner][ROOK] < 2
        && count[winner][KNIGHT] + count[winner][BISHOP] < 3) {
        int bal = 3 * (count[winner][KNIGHT] + count[winner][BISHOP] - count[winner^1][KNIGHT] - count[winner^1][BISHOP])
                + 5 * (count[winner][ROOK]   - count[winner^1][ROOK])
                + 9 * (count[winner][QUEEN]  - count[winner^1][QUEEN]);
        if (bal < 5) score /= 3;
    } */
    return score;
}

/* ===============================================================
   S10  MOVE ORDERING
   ===============================================================

   Idea
   Alpha-Beta pruning performs optimally when the best move is examined
   first. Perfect ordering theoretically reduces the search space from
   O(b^d) to O(b^(d/2)), doubling search depth effectively at zero cost.

   Implementation
   We score every generated move before searching, then pick the best
   lazily (one selection-sort step per move searched).
   1. Hash move      (30000):  TT best move from a prior search.
   2. MVV-LVA        (20000+): 20000 + 10*cap_val - atk_val.
   3. Promotion      (19999):  Queen underpromotion.
   4. Killer slot 0  (19998):  Most recent quiet beta-cutoff at this ply.
   5. Killer slot 1  (19997):  Older quiet beta-cutoff at this ply.
   6. History   (-16000..16000): Bonus/malus from beta-cutoff tracking.
      Negative scores are intentional: they push failing moves to the bottom
      of the ordering without ever skipping them entirely.
*/

static inline int pawn_defended_by_pawn(int s) {
    int p = piece_on(s); if (!p || piece_type(p) != PAWN) return 0;
    int c = piece_color(p), a1 = s+(c==WHITE?-17:15), a2 = s+(c==WHITE?-15:17);
    return (!sq_is_off(a1) && piece_is(a1,c,PAWN)) || (!sq_is_off(a2) && piece_is(a2,c,PAWN));
}

static inline int score_move(Move m, Move hash_move, int sply) {
    int cap, sc = 0;
    if (m == hash_move) return 30000;

    int fr = move_from(m); int to = move_to(m);
    cap = piece_on(move_to(m));

    /* EP captures land on an empty square; treat them as pawn captures for ordering. */
    if (!cap && ptype_on(move_from(m)) == PAWN && move_to(m) == ep_square)
        cap = make_piece(xside, PAWN);

    if (cap) {
        int hunter_type = ptype_on(fr); int prey_type = piece_type(cap);

        if (prey_type == PAWN && hunter_type != PAWN && pawn_defended_by_pawn(to))
            sc = -17000 + 10 * piece_val[prey_type] - piece_val[hunter_type];
        else
            sc = 20000 + 10 * piece_val[prey_type] - piece_val[hunter_type];
    }
    else if (move_promo(m)) sc = 19999;
    else if (sply < MAX_PLY && m == killers[sply][0]) sc = 19998;
    else if (sply < MAX_PLY && m == killers[sply][1]) sc = 19997;
    else                    sc = hist[fr][to];  /* [-16000, 16000] */
    return sc;
}

/* Score all moves into a parallel array. Called once before the move loop. */
static void score_moves(Move* moves, int* scores, int n, Move hash_move, int sply) {
    for (int i = 0; i < n; i++) scores[i] = score_move(moves[i], hash_move, sply);
}

/* Partial sort: swap the best remaining move to position idx.
   Called once per move inside the loop -- O(n) per pick vs O(n^2) total
   for selection sort, but we only pay for moves we actually search.      */
static void pick_move(Move* moves, int* scores, int n, int idx) {
    int best = idx;
    for (int i = idx + 1; i < n; i++)
        if (scores[i] > scores[best]) best = i;
    if (best != idx) {
        int ts = scores[idx]; scores[idx] = scores[best]; scores[best] = ts;
        Move tm = moves[idx];  moves[idx] = moves[best];  moves[best] = tm;
    }
}

/* ===============================================================
   S11  SEARCH
   ===============================================================

   Idea
   The engine looks ahead by recursively exploring all replies to all
   moves.  The tree grows exponentially with depth, so we prune
   branches that cannot change the final result.

   Implementation (Negamax Alpha-Beta)
   Negamax reformulates minimax as a single recursive function: the
   score for the current side is the negation of the best score the
   opponent achieves.  Alpha-beta cuts branches where the opponent
   already has a refutation.

   Extra heuristics layered on top:
   1. Quiescence Search (QS): at depth 0, keep searching captures
      until the position is "quiet" to avoid the horizon effect.
   2. Reverse Futility Pruning (RFP): at shallow depths, if the static
      eval minus a margin already beats beta, skip the search entirely.
      Zero nodes spent -- much cheaper than NMP.
   3. Null Move Pruning (NMP): pass our turn; if the opponent still
      can't beat beta at reduced depth, prune without searching.
   4. Principal Variation Search + LMR: search the first legal move with
      a full window. Every subsequent move is probed with a null window
      (-alpha-1, -alpha); late quiet moves also get depth-2 (LMR probe).
      A null-window beat that escapes the alpha bound triggers a full
      re-search. This is the standard PVS/LMR architecture.
   5. Transposition Table (TT): cache each sub-tree result by Zobrist
      key so the same position via different move orders is only
      searched once.
   6. Aspiration Windows: start each iterative-deepening depth with a
      narrow window around the previous score; widen on failure.
   7. Repetition detection: 2-fold within the search tree returns draw
      immediately; 2 prior occurrences in game history (3-fold total) also
      returns draw. Bounded by halfmove_clock to skip irreversible positions.
*/

/* ---------------------------------------------------------------
   print_move / print_pv  -- formatting helpers
   ---------------------------------------------------------------
   A chess move in UCI format: <from><to>[promo], e.g. "e2e4", "a7a8q".
   print_pv prints the full principal variation for the info line.
*/

void print_move(Move m) {
    static const char promo_ch[7] = { 0,0,'n','b','r','q',0 }; /* KNIGHT=2..QUEEN=5 */
    int fr = move_from(m), to = move_to(m), pr = move_promo(m);
    printf("%c%c%c%c", 'a' + (fr & 7), '1' + (fr >> 4), 'a' + (to & 7), '1' + (to >> 4));
    if (pr && pr < 7) putchar(promo_ch[pr]);
}

static void print_pv(void) {
    for (int k = 0; k < pv_length[0]; k++) { putchar(' '); print_move(pv[0][k]); }
}

void print_result(int best_sc) {
    /* UCI info line: depth, score, nodes, time, pv
   MATE SCORE FORMAT
   -----------------
   The UCI spec requires two distinct score tokens:
     "score cp X"    -- normal centipawn score
     "score mate N"  -- N moves to checkmate
                       positive N: we deliver mate
                       negative N: we are being mated
   We detect a mate score by testing abs(score) > MATE - MAX_PLY.
   The move-count formula:
     mating:  N =  (MATE - score + 1) / 2
     mated:   N = -(MATE + score + 1) / 2
     */
    int64_t ms = (int64_t)(((int64_t)(clock() - t_start) * 1000) / CLOCKS_PER_SEC);
    int64_t nps = ms ? 1000 * nodes_searched / (clock() - t_start) : 0;
    if (best_sc > MATE - MAX_PLY) printf("info depth %d score mate %d nodes %" PRId64 " time %" PRId64 " nps %" PRId64 " pv", root_depth, (MATE - best_sc + 1) / 2, nodes_searched, ms, nps);
    else if (best_sc < -(MATE - MAX_PLY)) printf("info depth %d score mate %d nodes %" PRId64 " time %" PRId64 " nps %" PRId64 " pv", root_depth, -(MATE + best_sc + 1) / 2, nodes_searched, ms, nps);
    else    printf("info depth %d score cp %d nodes %" PRId64 " time %" PRId64 " nps %" PRId64 " pv", root_depth, best_sc, nodes_searched, ms, nps);
    print_pv(); printf("\n"); fflush(stdout);
}

/* ---------------------------------------------------------------
   qsearch  -- quiescence search (captures only)
   ---------------------------------------------------------------
   At depth 0 the horizon effect can cause tactical blindness: the
   engine stops thinking just as a piece is hanging.  Quiescence
   search extends the tree with captures until the position is
   "quiet," eliminating this horizon distortion.

   Stand-pat: the side to move can always decline further captures,
   so we initialise best_sc = static eval.  If that already exceeds
   beta we prune immediately (standing pat is good enough).

   Delta pruning: if even capturing the most valuable piece on the
   board plus a safety margin cannot raise alpha, skip the subtree
   entirely -- no capture can possibly help.

   SEE pruning replaces the old pawn-defended-pawn heuristic with a
   full exchange evaluation: any capture where SEE < 0 is skipped.
--------------------------------------------------------------- */

static inline int line_step(int from, int to) {
    int diff = to - from;

    if ((from & 7) == (to & 7))         return (diff > 0) ? 16 : -16;  /* same file */
    if ((from >> 4) == (to >> 4))       return (diff > 0) ? 1 : -1;  /* same rank */
    if (diff % 17 == 0)                 return (diff > 0) ? 17 : -17;  /* main diagonal */
    if (diff % 15 == 0)                 return (diff > 0) ? 15 : -15;  /* anti-diagonal */

    return 0; /* not aligned */
}

/* ---------------------------------------------------------------
   piece_attacks_sq  --  does the piece on `from` attack `to`?
   Squares in cleared[] are treated as empty for slider ray tracing,
   which lets sliding pieces "see through" captured pieces (x-rays).
--------------------------------------------------------------- */
static int piece_attacks_sq(int from, int to) {
    int diff = to - from;
    int pt  = ptype_on(from);
    int col = color_on(from);

    if (pt == PAWN) {
        return diff == ((col == WHITE) ? 15 : -17)
            || diff == ((col == WHITE) ? 17 : -15);
    }

    if (pt == KNIGHT) {
        return diff == -33 || diff == -31 || diff == -18 || diff == -14 ||
               diff == 14 || diff == 18 || diff == 31 || diff == 33;
    }

    if (pt == KING) {
        return diff == -17 || diff == -16 || diff == -15 || diff == -1 ||
               diff == 1 || diff == 15 || diff == 16 || diff == 17;
    }

    /* Sliders */
    int step = line_step(from, to);
    if (!step) return 0;

    if (pt == BISHOP && step != -17 && step != -15 && step != 15 && step != 17) return 0;
    if (pt == ROOK && step != -16 && step != -1 && step != 1 && step != 16) return 0;

    for (int sq = from + step; !(sq & 0x88); sq += step) {
        if (sq == to) return 1;
        if (!is_empty(sq) && see_cleared[sq] != see_sentinel) break;
    }

    return 0;
}

/* ---------------------------------------------------------------
   see  --  Static Exchange Evaluation
   Simulates all captures on `to`, starting with the piece on `from`.
   Returns net material gain (cp) for the side making the first capture.
   Negative = losing exchange.  X-rays are handled naturally: when a
   piece is marked cleared[], sliders behind it join the exchange.
--------------------------------------------------------------- */
static int see(int from, int to) {
   
    if (see_sentinel >= 16384) {
        see_sentinel = 1;
        memset(see_cleared, 0, sizeof(see_cleared));
    }
    else see_sentinel++;

    int cap_type = ptype_on(to);
    if (!cap_type) return 0;   /* en passant: captured pawn not on `to`, treat as even */

    see_cleared[from] = see_sentinel;

    /* target_seq[d] = type of piece sitting on `to` that gets captured at step d.
       Step 0 = initial capture; step d+1 = recapture of what step d placed. */
    int target_seq[32];
    int nsteps = 0;
    target_seq[0] = cap_type;

    int piece_on_to = ptype_on(from); /* attacker now sits on `to` */
    int cur_side    = color_on(from) ^ 1;

    while (nsteps < 31) {
        /* Find least-valuable attacker of `to` for cur_side, skipping cleared squares */
        int lva_sq = -1, lva_type = 0, lva_val = INF;
        int base = (cur_side == WHITE) ? 0 : 16;
        int top  = base + list_count[cur_side];
        for (int i = base; i < top; i++) {
            int psq = list_square[i];
            if (psq == LIST_OFF) continue; /* captured in game */
            if (see_cleared[psq] == see_sentinel) continue;    /* already used in this exchange */
            int pv = piece_val[list_piece[i]];
            if (pv < lva_val && piece_attacks_sq(psq, to)) {
                lva_val  = pv;
                lva_sq   = psq;
                lva_type = list_piece[i];
            }
        }
        if (lva_sq < 0) break;

        target_seq[nsteps + 1] = piece_on_to;
        see_cleared[lva_sq] = see_sentinel;
        piece_on_to = lva_type;
        cur_side ^= 1;
        nsteps++;
    }

    /* Walk back: each player can stop (gain 0) rather than continue losing.
       result = opponent's maximum gain from step d onward. */
    int result = 0;
    for (int d = nsteps - 1; d >= 0; d--) {
        int gain = piece_val[target_seq[d + 1]] - result;
        result = gain > 0 ? gain : 0;
    }
    return piece_val[cap_type] - result;
}

/* Returns 1 if the capture from->to is obviously or probably losing.
   Fast path: attacker <= victim means SEE >= 0, so skip the full exchange. */
static inline int is_bad_capture(int from, int to) {
    if (piece_val[ptype_on(from)] <= piece_val[ptype_on(to)]) return 0;
    return see(from, to) < 0;
}

static int qsearch(int alpha, int beta, int sply) {
    Move moves[256]; int best_sc, sc;

    pv_length[sply] = sply;

    /* Time check -- same cadence as main search */
    if ((nodes_searched & 1023) == 0 && time_budget_ms > 0) {
        int64_t ms = (int64_t)(((int64_t)(clock() - t_start) * 1000) / CLOCKS_PER_SEC);
        if (ms >= time_budget_ms) { time_over_flag = 1; return 0; }
    }
    if (time_over_flag) return 0;

    /* Stand-pat: static eval as lower bound (we can always stop capturing) */
    best_sc = evaluate();
    if (best_sc >= beta) return best_sc;
    if (best_sc > alpha) alpha = best_sc;

    nodes_searched++;

    int cnt = generate_captures(moves);
    int scores[256];
    score_moves(moves, scores, cnt, 0, sply);

    for (int i = 0; i < cnt; i++) {
        pick_move(moves, scores, cnt, i);

        /* DELTA PRUNING: skip if even the captured piece + margin cannot raise alpha */
        int dp_cap = piece_on(move_to(moves[i]));
        int dp_ep  = (!dp_cap && ptype_on(move_from(moves[i])) == PAWN
                      && move_to(moves[i]) == ep_square);
        if (dp_cap || dp_ep) {
            int cap_val = dp_cap ? piece_val[piece_type(dp_cap)] : piece_val[PAWN];
            if (best_sc + cap_val + 200 < alpha) continue;
        }

        /* QS SEE PRUNING: skip captures that lose material in the full exchange.
           Promotions are always searched (large material swing).
           En passant returns SEE == 0 (ptype_on(to) == 0) so is kept. */
        if (!move_promo(moves[i]) && is_bad_capture(move_from(moves[i]), move_to(moves[i])))
            continue;

        make_move(moves[i]);
        if (is_illegal()) { undo_move(); continue; }

        sc = -qsearch(-beta, -alpha, sply + 1);
        undo_move();

        if (sc > best_sc) best_sc = sc;
        if (sc > alpha) {
            alpha = sc;
            if (!time_over_flag && moves[i] != 0) {
                pv[sply][sply] = moves[i];
                for (int k_ = sply + 1; k_ < pv_length[sply + 1]; k_++) pv[sply][k_] = pv[sply + 1][k_];
                pv_length[sply] = pv_length[sply + 1];
            }
        }
        if (alpha >= beta) break;
    }
    return best_sc;
}

/* ---------------------------------------------------------------
   search  -- negamax alpha-beta with TT, PVS, and killers
   ---------------------------------------------------------------
   Negamax: the score for the side to move equals the negation of
   the best score the opponent achieves.  One recursive function
   replaces the classical minimax pair.

   Alpha-beta: maintain a window [alpha, beta].  Alpha = best score
   the maximising side is guaranteed; beta = best the minimising
   side is guaranteed.  Any subtree that cannot improve on these
   bounds is pruned immediately.

   Principal Variation Search (PVS): the first legal move is
   searched with the full [alpha, beta] window.  Every subsequent
   move is probed with a null window (-alpha-1, -alpha) -- if our
   current best is truly best they fail low cheaply.  A null-window
   score that beats alpha triggers a full re-search on PV nodes.

   Check extension: if the move gives check, extend by 1 ply so the
   engine never horizon-drops while the opponent is in check.
--------------------------------------------------------------- */
int search(int depth, int alpha, int beta, int sply, int was_null) {
    Move moves[256], best = 0, hash_move = 0;
    int legal = 0, best_sc, old_alpha = alpha, sc;
    int is_pv = (beta - alpha > 1); /* PV node: wide window, not a null-window probe */
    TTEntry* e = &tt[hash_key % (HASH)tt_size];

    /* Clear PV at this ply before any early returns (TT hits, repetition).
       The parent reads pv_length[sply] to splice in the child continuation;
       it must equal sply (empty) rather than a stale value. */
    pv_length[sply] = sply;

    /* HARD TIME LIMIT CHECK
       Every 1024 nodes, check if we have exceeded our absolute time budget.
       If we have, abort the search tree immediately to prevent flagging. */
    if ((nodes_searched & 1023) == 0 && time_budget_ms > 0) {
        int64_t ms = (int64_t)(((int64_t)(clock() - t_start) * 1000) / CLOCKS_PER_SEC);
        if (ms >= time_budget_ms) { time_over_flag = 1; return 0; }
    }
    if (time_over_flag) return 0;

    /* Drop into quiescence search at the horizon */
    if (depth <= 0) return qsearch(alpha, beta, sply);

    /* REPETITION DETECTION
       Two rules apply, depending on whether the repeated position is inside
       the current search tree or in the game history before the search root.

       In-tree (ply >= root_ply): we are actively creating the repetition.
       One prior occurrence is enough to return draw -- the opponent can
       always force the third occurrence on the real board.

       In-history (ply < root_ply): the position was reached before the
       search started. That is only one prior occurrence; strict threefold
       requires two prior occurrences (three total) to be a forced draw.

       The halfmove_clock bound is exact: no repetition can cross an
       irreversible move (pawn advance or capture), so we need not look
       further back than ply - halfmove_clock. We step by 2 because
       repetitions require the same side to move. */
    if (ply > root_ply) {
        /* Repetition detection */
        for (int i = ply - 2; i >= root_ply; i -= 2)
            if (history[i].hash_prev == hash_key) return 0;
        int reps = 0;
        for (int i = ply - 2; i >= 0 && i >= ply - halfmove_clock; i -= 2)
            if (history[i].hash_prev == hash_key && ++reps >= 2) return 0;

        /* 50-move rule */
        if (halfmove_clock >= 100) return 0;

        /* INSUFFICIENT MATERIAL
           Only trigger when there is exactly one minor piece on the board total
           (KNK or KBK). With one minor per side the corner-checkmate edge case
           means we cannot safely claim a draw. */
        {
            int wm = count[WHITE][KNIGHT]+count[WHITE][BISHOP], bm = count[BLACK][KNIGHT]+count[BLACK][BISHOP];
            if (wm+bm==1 && !count[WHITE][PAWN] && !count[BLACK][PAWN]
                && !count[WHITE][ROOK] && !count[BLACK][ROOK] && !count[WHITE][QUEEN] && !count[BLACK][QUEEN])
                return 0;
        }
    }

    /* TT probe: always extract hash_move for ordering */
    if (e->key == hash_key) {
        hash_move = e->best_move;
        if ((int)tt_depth(e) >= depth) {
            int flag = tt_flag(e);
            /* Mate scores are stored relative to the node that proved them
               (+sply on write) so the same position compares correctly when
               retrieved via a transposition at a different search depth.
               Reverse that shift before using the score here.            */
            int tt_sc = e->score;
            if (tt_sc > MATE - MAX_PLY) tt_sc -= sply;
            if (tt_sc < -(MATE - MAX_PLY)) tt_sc += sply;
            if (sply > 0) {
                if (flag == TT_EXACT)                            return tt_sc;
                if (!is_pv && flag == TT_BETA && tt_sc >= beta) return tt_sc;
                if (!is_pv && flag == TT_ALPHA && tt_sc <= alpha) return tt_sc;
            }
        }
    }

    best_sc = -INF;
    nodes_searched++;

    /* Cache whether the side to move is currently in check.
       RFP, NMP, and IIR all guard on this -- compute once, reuse three times. */
    int node_in_check = (sply > 0) ? history[ply - 1].in_check : in_check(side);

    /* REVERSE FUTILITY PRUNING (RFP)
       If static eval is already well above beta at shallow depth, the
       position is unlikely to become worse after a quiet move -- prune
       immediately. Zero nodes spent per pruned node.
       Guards: not at root (sply>0), not in check, not a mate score. */
    /* Compute static eval once; shared by RFP, razoring, and NMP guard below.
       Skipped entirely when in check (no pruning applies). */
    int static_eval = (!is_pv && sply > 0 && !node_in_check && beta < MATE - MAX_PLY && depth <= 7)
                      ? evaluate() : -INF;

    if (static_eval != -INF) {
        /* REVERSE FUTILITY PRUNING (RFP) */
        if (depth <= 7 && static_eval - 70 * depth >= beta)
            return static_eval - 70 * depth;
        /* RAZORING: if eval is far below alpha even after the best capture,
           there is no point searching -- drop straight into qsearch. */
        if (depth <= 3 && static_eval + 300 + 60 * depth < alpha)
            return qsearch(alpha, beta, sply);
    }

    /* NULL MOVE PRUNING (NMP)
       Guard: static_eval >= beta ensures we're not in a losing position --
       passing the turn when already losing is pointless and wastes a search.
       R=3 normally, R=4 at depth >= 7.
       Guards: not a PV node, no consecutive null moves (was_null),
       not in check, and the side to move has non-pawn material
       (zugzwang guard -- in pure pawn endings passing is often worst). */
    if (!is_pv && sply > 0 && depth >= 3 && !was_null
        && !node_in_check && beta < MATE - MAX_PLY
        && (count[side][KNIGHT] + count[side][BISHOP]
            + count[side][ROOK]  + count[side][QUEEN] > 0)) {

        if (static_eval == -INF) static_eval = evaluate();

        if (static_eval >= beta) {
            int R = depth >= 7 ? 4 : 3;
            int ep_prev = ep_square;
            hash_key ^= zobrist_side;
            if (ep_square != SQ_NONE) hash_key ^= zobrist_ep[ep_square];
            ep_square = SQ_NONE;
            side ^= 1; xside ^= 1;
            history[ply].hash_prev = hash_key; ply++; /* push null move so repetition detection sees it */
            int null_sc = -search(depth - R - 1, -beta, -beta + 1, sply + 1, 1);
            ply--; side ^= 1; xside ^= 1;
            ep_square = ep_prev;
            if (ep_square != SQ_NONE) hash_key ^= zobrist_ep[ep_square];
            hash_key ^= zobrist_side;
            if (null_sc >= beta) return null_sc;
        }
    }

    /* INTERNAL ITERATIVE REDUCTIONS (IIR)
       If no hash move is available, the move ordering at this node is poor.
       Reduce depth by 1 to avoid spending too much time on a badly-ordered
       node; the resulting TT entry will guide a future full-depth search.
       Guards: depth >= 4 so we don't reduce already-shallow nodes;
       not in check -- evasions must be searched at full depth. */
    if (depth >= 4 && !hash_move && !node_in_check) depth--;

    int cnt = generate_moves(moves, 0);
    int scores[256];
    score_moves(moves, scores, cnt, hash_move, sply);
    /* quiet_moves tracks searched quiet moves in order so the history malus
       loop can penalise them without re-examining board[] after undo_move. */
    Move quiet_moves[256]; int nquiet = 0, quiet = 0;

    for (int i = 0; i < cnt; i++) {
        pick_move(moves, scores, cnt, i);

        /* Capture flag read before make_move: after the call board[to] holds
           a piece regardless, making a post-move test useless.
           En-passant has an empty destination, so check ep_square too. */
        int is_cap = !is_empty(move_to(moves[i]))
                  || (ptype_on(move_from(moves[i])) == PAWN
                      && move_to(moves[i]) == ep_square);

        /* PVS SEE PRUNING (captures)
           Skip captures whose full exchange value is below a depth-scaled
           threshold. Only applied once at least one legal move has been
           searched (legal > 0) so we never prune the first move.
           No make_move needed → no undo cost. */
        if (!is_pv && !node_in_check && is_cap && !move_promo(moves[i])
            && legal > 0
            && piece_val[ptype_on(move_from(moves[i]))] > piece_val[ptype_on(move_to(moves[i]))]
            && see(move_from(moves[i]), move_to(moves[i])) < -piece_val[PAWN] * depth)
            continue;

        make_move(moves[i]);
        if (is_illegal()) { undo_move(); continue; }
        legal++;
        if (!is_cap && !move_promo(moves[i])) quiet++;

        /* Cache whether this move gives check (opponent in check after make_move).
           Used by both LMP and check extension -- compute once, reuse twice. */
        int gives_check = history[ply - 1].in_check;

        /* LATE MOVE PRUNING (LMP)
           At shallow depths on non-PV nodes, skip quiet moves beyond the first few.
           Threshold: depth 1 allows 5, depth 2 allows 9, depth 3 allows 13.
           Moves that give check are exempted: they may be the only defence.
           Also skip entirely when the mover was in check (evasions must be fully searched). */
        if (!is_pv && depth < 4 && !node_in_check && quiet > 4 * depth + 1
            && !is_cap && !move_promo(moves[i])) {
            if (!gives_check) { undo_move(); continue; }
        }

        /* Push to quiet list only after LMP so pruned moves don't get malus */
        if (!is_cap && !move_promo(moves[i])) quiet_moves[nquiet++] = moves[i];
        /* CHECK EXTENSION: if the move gives check, extend by 1 ply.
           This ensures the engine never horizon-drops into QS while
           the opponent is in check -- the resolution is searched fully. */
        int ext = gives_check ? 1 : 0;

        /* PRINCIPAL VARIATION SEARCH + LMR
           First legal move: full window to establish the PV.
           All others: null window first; late quiet moves also get
           a depth reduction from lmr_table. Re-search at full depth
           if the reduced score beats alpha. */
        if (legal == 1) {
            sc = -search(depth - 1 + ext, -beta, -alpha, sply + 1, 0);
        } else {
            int lmr = 0;
            /* Apply LMR only at depth >= 3 (enough remaining depth to be meaningful)
               and after the 4th move (legal > 4) -- early moves are more likely
               to be strong so we search them fully before reducing later ones.
               Captures, promotions, and check-giving moves always get full depth. */
            if (depth >= 3 && legal > 4 && !is_cap && !move_promo(moves[i]) && !ext) {
                lmr = lmr_table[min(depth,31)][min(legal,63)];
                if (lmr > depth - 2) lmr = depth - 2; /* always leave at least 1 ply */
                if (lmr < 0) lmr = 0;
            }
            /* Three-level search:
               1. Reduced null window (LMR probe): fast refutation check.
               2. Full-depth null window: if reduced score beat alpha, get an exact
                  bound at the proper depth (cheap if move is truly bad).
               3. Full window on PV nodes only: if the score still exceeds alpha
                  here, this move is a new PV candidate and needs an exact score. */
            sc = -search(depth - 1 + ext - lmr, -alpha - 1, -alpha, sply + 1, 0);
            if (sc > alpha && lmr > 0)
                sc = -search(depth - 1 + ext, -alpha - 1, -alpha, sply + 1, 0);
            if (sc > alpha && is_pv)
                sc = -search(depth - 1 + ext, -beta, -alpha, sply + 1, 0);
        }

        undo_move();

        if (sc > best_sc) best_sc = sc;
        if (sc > alpha) {
            alpha = sc;
            best = moves[i];
            /* Triangular PV update: store this move, then copy the child
               ply's continuation into the current row of the table. */
            if (!time_over_flag && moves[i] != 0) {
                pv[sply][sply] = moves[i];
                for (int k_ = sply + 1; k_ < pv_length[sply + 1]; k_++) pv[sply][k_] = pv[sply + 1][k_];
                pv_length[sply] = pv_length[sply + 1];
                if (sply == 0) { best_root_move = moves[i]; print_result(best_sc); }
            }
        }
        if (alpha >= beta) {
            if (!is_cap && !move_promo(moves[i])) {
                int d = (sply < MAX_PLY) ? sply : MAX_PLY - 1;
                killers[d][1] = killers[d][0]; killers[d][0] = moves[i];
                /* History BONUS for the cutoff move, MALUS for quiets tried before it.
                   Gravity formula: self-corrects instead of saturating at ±16000. */
                int bonus = depth * depth;
                int h = hist[move_from(moves[i])][move_to(moves[i])];
                h += bonus - h * bonus / 16000;
                hist[move_from(moves[i])][move_to(moves[i])] = h > 16000 ? 16000 : h;
                for (int j = 0; j < nquiet - 1; j++) {
                    int hm = hist[move_from(quiet_moves[j])][move_to(quiet_moves[j])];
                    hm -= bonus + hm * bonus / 16000;
                    hist[move_from(quiet_moves[j])][move_to(quiet_moves[j])] = hm < -16000 ? -16000 : hm;
                }
            }
            break;
        }
    }

    /* Checkmate or stalemate: no legal moves found after full generation.
       MATE - sply encodes distance-to-mate so shorter mates score higher.
       Stalemate returns 0 (draw). */
    if (!legal) return node_in_check ? -(MATE - sply) : 0;

    /* TT store: skip if search was aborted mid-tree (score is meaningless) */
    if (!time_over_flag && (e->key != hash_key || depth >= (int)tt_depth(e))) {
        int flag = (best_sc <= old_alpha) ? TT_ALPHA :
            (best_sc >= beta) ? TT_BETA : TT_EXACT;
        /* Encode mate scores as distance-from-node (+sply) so the score
           stays valid when the position is retrieved via a transposition. */
        int sc_store = best_sc;
        if (sc_store > MATE - MAX_PLY) sc_store += sply;
        if (sc_store < -(MATE - MAX_PLY)) sc_store -= sply;
        /* For fail-low nodes, preserve the hash move from the probe for
           ordering on the next visit even if no move raised alpha. */
        Move store_move = best ? best : hash_move;
        e->key = hash_key; e->score = sc_store; e->best_move = store_move;
        e->depth_flag = tt_pack(depth > 0 ? depth : 0, flag);
    }
    return best_sc;
}

/* ---------------------------------------------------------------
   search_root  -- iterative deepening with UCI info output
   ---------------------------------------------------------------

   We search depth 1, then 2, ... up to max_depth. After each
   depth completes we emit a UCI info line:

       info depth N score cp X nodes N time N pv e2e4 e7e5 ...

   This lets the GUI display the engine's thinking in real time.
   The best move from depth D guides ordering for depth D+1 via the
   TT, making iterative deepening almost free compared to jumping
   straight to depth N.

   The root best move is placed at the front of the move list before
   each iteration so it is tried first (TT also achieves this, but
   explicit placement is a cheap belt-and-suspenders guarantee).
*/

void search_root(int max_depth) {
    int sc = 0, prev_sc = 0;
    time_over_flag = 0; best_root_move = 0; t_start = clock();
    memset(hist, 0, sizeof(hist)); memset(killers, 0, sizeof(killers));
    memset(pv, 0, sizeof(pv)); memset(pv_length, 0, sizeof(pv_length));
    nodes_searched = 0;
    root_ply = ply;   /* anchor sply=0 at the search root */

    for (root_depth = 1; root_depth <= max_depth; root_depth++) {
        if (root_depth < 5) {
            /* Full window for early depths: score too volatile for a narrow window. */
            sc = search(root_depth, -INF, INF, 0, 0);
        } else {
            /* ASPIRATION WINDOWS
               Initial delta scales with score magnitude: unbalanced positions
               (large |prev_sc|) are more volatile so they get a wider window.
               On fail-low: collapse beta to the midpoint before widening alpha,
               avoiding a needlessly large window on the high side.
               On fail-high: widen beta only.
               Proportional widening (delta += delta/2) is smoother than doubling. */
            int delta = 15 + prev_sc * prev_sc / 16384;
            int alpha = max(prev_sc - delta, -INF);
            int beta  = min(prev_sc + delta,  INF);
            while (1) {
                sc = search(root_depth, alpha, beta, 0, 0);
                if (time_over_flag) break;
                if (sc <= alpha) {
                    beta  = (alpha + beta) / 2; /* midpoint collapse */
                    alpha = max(alpha - delta, -INF);
                } else if (sc >= beta) {
                    beta = min(beta + delta, INF);
                } else {
                    break;                      /* exact score within window */
                }
                delta += delta / 2;             /* proportional widening */
            }
        }
        if (time_over_flag) break;
        prev_sc = sc;

        /* TIME CONTROL: stop iterating if we have used our budget.
           We check AFTER a depth completes, never mid-search, so
           the move we return is always from a fully searched depth. */
        {
            int64_t ms = (int64_t)(((int64_t)(clock() - t_start) * 1000) / CLOCKS_PER_SEC);
            if (time_budget_ms > 0 && ms >= time_budget_ms / 2) break;
        }
    }
    (void)sc;

    printf("bestmove ");
    if (best_root_move) print_move(best_root_move);
    else printf("0000");
    printf("\n");
    fflush(stdout);
}

/* ===============================================================
   S12  PERFT
   ===============================================================

   Idea
   Perft counts the exact number of leaf nodes reachable at a given
   depth from a given position.

   Implementation
   The counts are compared against published reference values.  Any
   discrepancy immediately pinpoints a bug in move generation or
   make/undo -- no evaluation or search heuristics are involved, so
   the numbers are fully deterministic.
*/

int64_t perft(int depth) {
    if (!depth) return 1;
    Move moves[256];
    int cnt = generate_moves(moves, 0);
    int64_t n = 0;
    for (int i = 0; i < cnt; i++) {
        make_move(moves[i]);
        if (!is_illegal()) n += perft(depth - 1);
        undo_move();
    }
    return n;
}

/* ===============================================================
   S13  UCI LOOP
   ===============================================================

   Idea
   UCI (Universal Chess Interface) is the standard text protocol
   between an engine and a GUI.  The engine reads commands from stdin
   and writes responses to stdout; the GUI drives the session.

   Implementation
   A simple readline loop dispatches on the first token of each
   command.  "position" sets up the board; "go" runs the search and
   streams info lines during it; "bestmove" closes the response.
   All output is flushed immediately so the GUI never blocks.
*/

static int parse_move(const char* s, Move* out) {
    int fr, to, pr = 0; Move list[256]; int cnt, i;
    if (!s || strlen(s) < 4) return 0;
    fr = (s[0] - 'a') + (s[1] - '1') * 16;
    to = (s[2] - 'a') + (s[3] - '1') * 16;
    if (strlen(s) > 4) pr = char_to_piece((char)tolower(s[4]));
    cnt = generate_moves(list, 0);
    for (i = 0; i < cnt; i++)
        if (move_from(list[i]) == fr && move_to(list[i]) == to && move_promo(list[i]) == pr) { *out = list[i]; return 1; }
    return 0;
}

/* Helpers for uci_loop */
static const char STARTPOS[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/* getval: find `key` in `line`, skip one space, then scan into `out` with `fmt`.
   Uses strlen so it works on any runtime string, not just string literals.
   e.g. getval(line, "depth", "%d", &depth) on "go depth 6" -> depth=6. */
static void getval(const char* line, const char* key, const char* fmt, void* out) {
    const char* t = strstr(line, key);
    if (t) sscanf(t + strlen(key) + 1, fmt, out);
}

void uci_loop(void) {
    char line[65536], * p;
    Move m;

    tt = calloc((size_t)tt_size, sizeof(TTEntry));
    if (!tt) {
        fprintf(stderr, "Error: failed to allocate TT (%lld entries)\n",
            (long long)tt_size); exit(1);
    }

    while (fgets(line, sizeof(line), stdin)) {
        if (!strncmp(line, "ucinewgame", 10)) {
            memset(tt, 0, (size_t)tt_size * sizeof(TTEntry)); memset(hist, 0, sizeof(hist));
            parse_fen(STARTPOS); hash_key = generate_hash();
        }
        else if (!strncmp(line, "uci", 3)) {
            puts("id name Chal " CHAL_VERSION "\nid author Naman Thanki\noption name Hash type spin default 16 min 1 max 4096\nuciok"); fflush(stdout);
        }
        else if (!strncmp(line, "setoption", 9)) {
            /* setoption name Hash value <N>  (N in megabytes) */
            char* nptr = strstr(line, "name Hash value ");
            if (nptr) {
                int mb = 1; sscanf(nptr + 16, "%d", &mb); if (mb < 1) mb = 1;
                int64_t new_size = max((int64_t)1, ((int64_t)mb * 1024 * 1024) / (int64_t)sizeof(TTEntry));
                TTEntry* new_tt = calloc((size_t)new_size, sizeof(TTEntry));
                if (new_tt) { free(tt); tt = new_tt; tt_size = new_size; }
            }
        }
        else if (!strncmp(line, "isready", 7)) { printf("readyok\n"); fflush(stdout); }
        else if (!strncmp(line, "perft", 5)) {
            int depth = 4; sscanf(line, "perft %d", &depth);
            printf("perft depth %d nodes %" PRId64 "\n", depth, perft(depth)); fflush(stdout);
        }
        else if (!strncmp(line, "position", 8)) {
            if (strlen(line) <= 9) continue;
            p = line + 9;
            if (!strncmp(p, "startpos", 8)) {
                parse_fen(STARTPOS);
                p += 8;
            }
            else if (!strncmp(p, "fen", 3)) {
                p += 4; parse_fen(p);
            }
            hash_key = generate_hash();
            p = strstr(line, "moves");
            if (p) {
                p += 6;
                while (*p) {
                    char mv[6];
                    int n = 0;
                    while (*p == ' ') p++;
                    if (*p == '\n' || !*p) break;
                    sscanf(p, "%5s%n", mv, &n);
                    if (n <= 0) break;
                    if (parse_move(mv, &m)) make_move(m);
                    p += n;
                }
            }
        }
        else if (!strncmp(line, "go", 2)) {
            /* TIME CONTROL
               ------------
               UCI sends one of two forms:

                 go depth N
                   Fixed-depth mode: ignore the clock entirely, search
                   exactly N plies.  Used by analysis GUIs and test suites.

                 go wtime W btime B [movestogo M] [winc I] [binc I]
                   Clock mode.  W and B are milliseconds remaining for
                   White and Black.  movestogo is how many moves remain
                   in the current time period (absent in increment-only
                   time controls).  winc/binc are per-move increments.

               BUDGET FORMULA
               ---------------
               Divide remaining time evenly across expected moves left:

                   budget = our_time / movestogo + our_increment

               If movestogo is not given we assume 20 moves remain --
               a safe estimate for sudden-death and increment games.
               search_root() iterates deeper until it has consumed more
               than half the budget for a single depth (at which point
               the next depth would almost certainly bust the limit), then
               returns the best move from the last fully searched depth. */

            int  depth = MAX_PLY;
            int64_t wtime = 0, btime = 0, movestogo = 20, winc = 0, binc = 0;

            getval(line, "depth", "%d", &depth);
            getval(line, "wtime", "%" SCNd64, &wtime);
            getval(line, "btime", "%" SCNd64, &btime);
            getval(line, "movestogo", "%" SCNd64, &movestogo);
            getval(line, "winc", "%" SCNd64, &winc);
            getval(line, "binc", "%" SCNd64, &binc);

            if (wtime || btime) {
                int64_t our_time = (side == WHITE) ? wtime : btime;
                int64_t our_inc = (side == WHITE) ? winc : binc;
                if (movestogo <= 0) movestogo = 20;
                time_budget_ms = (our_time / movestogo) + (our_inc * 3 / 4);
                if (time_budget_ms > our_time - 50) time_budget_ms = our_time - 50;
                if (time_budget_ms < 5) time_budget_ms = 5;
                depth = MAX_PLY;
            } else {
                time_budget_ms = 0;
            }
            search_root(depth);
        }
        else if (!strncmp(line, "quit", 4)) break;
    }
    free(tt);
}

/* ===============================================================
   ENTRY POINT
   =============================================================== */

int main(void) {
    setbuf(stdout, NULL);
    memset(see_cleared, 0, sizeof(see_cleared));
    init_zobrist();
    init_lmr();
    parse_fen(STARTPOS);
    hash_key = generate_hash();
    uci_loop();
    return 0;
}
