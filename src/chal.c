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
   S11 Search                    - negamax, alpha-beta, quiescence
   S12 Perft                     - correctness testing
   S13 UCI Loop                  - GUI communication
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

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

enum { EMPTY=0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum { WHITE=0, BLACK=1 };
enum { SQ_NONE = -1 };
static inline int piece_type(int p)  { return p & 7; }
static inline int piece_color(int p) { return p >> 3; }
static inline int make_piece(int c, int t) { return (c << 3) | t; }
enum { INF = 50000, MATE = 30000 };

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
static inline int  move_from(Move m)  { return m & 0x7F; }
static inline int  move_to(Move m)    { return (m >> 7) & 0x7F; }
static inline int  move_promo(Move m) { return (m >> 14) & 0xF; }
static inline Move make_move_enc(int f, int t, int p) { return f | (t << 7) | (p << 14); }

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

enum { TT_EXACT=0, TT_ALPHA=1, TT_BETA=2 };

/* 64-bit Zobrist key type -- halves collision rate vs 32-bit */
typedef uint64_t HASH;

typedef struct {
    HASH key; int score; Move best_move; unsigned int depth_flag;
} TTEntry;

/* TT size is configurable via UCI setoption name Hash (default 16 MB). */
TTEntry *tt = NULL;
int64_t tt_size = 1 << 20; /* default 1M entries = 16 MB */

static inline unsigned int tt_depth(const TTEntry *e) { return e->depth_flag >> 2; }          /* bits 7..2 */
static inline unsigned int tt_flag(const TTEntry *e)  { return e->depth_flag & 3; }            /* bits 1..0 */
static inline unsigned int tt_pack(int d, int f)      { return (unsigned int)((d << 2) | f); } /* write both */

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
    Move move; int piece_captured; int ep_square_prev; unsigned int castle_rights_prev; int halfmove_clock_prev; HASH hash_prev; int npc_prev[2];
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

int board[128];
int side, xside;
int ep_square;
unsigned int castle_rights;    /* bits: 1=WO-O  2=WO-O-O  4=BO-O  8=BO-O-O */
int king_sq[2];
int non_pawn_count[2];
int ply;
int halfmove_clock;   /* plies since last pawn move or capture; draw at 100 */
HASH hash_key;

/* Search telemetry -- reported in UCI info lines */
int64_t nodes_searched;

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
int piece_offsets[] = {0,0, 4,12,16,12,20};
int piece_limits[]  = {0,0,12,16,20,20,28};

/* Castling move data: index 0-1 = White, 2-3 = Black */
static const int castle_kf[] = {4, 4, 116, 116}, castle_kt[] = {6, 2, 118, 114};
static const int castle_rf[] = {7, 0, 119, 112}, castle_rt[] = {5, 3, 117, 115};
static const int castle_col[]= {WHITE, WHITE, BLACK, BLACK};
static const unsigned int castle_kmask[]= {~3u, ~3u, ~12u, ~12u}; /* Rights stripped when king moves */
static const int cr_sq[] = {0, 7, 112, 119};
static const unsigned int cr_mask[] = {~2u, ~1u, ~8u, ~4u}; /* Corner squares */

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
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

void init_zobrist(void) {
    for (int c=0;c<2;c++) for (int p=0;p<7;p++) for (int s=0;s<128;s++)
        zobrist_piece[c][p][s] = rand64();
    zobrist_side = rand64();
    for (int s=0;s<128;s++) zobrist_ep[s]     = rand64();
    for (int s=0;s<16; s++) zobrist_castle[s] = rand64();
}

HASH generate_hash(void) {
    HASH h = 0;
    int sq;
    FOR_EACH_SQ(sq) {
        if (board[sq]) h ^= zobrist_piece[piece_color(board[sq])][piece_type(board[sq])][sq];
    }
    if (side==BLACK)          h ^= zobrist_side;
    if (ep_square!=SQ_NONE)   h ^= zobrist_ep[ep_square];
    h ^= zobrist_castle[castle_rights];
    return h;
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
   Using the `step_dir` array (S3), we simulate piece movement originating
   from the target square. For example, to check for knight attacks, we 
   fire knight-rays; if they hit an enemy knight, the square is attacked.
*/

static inline int is_square_attacked(int sq, int ac) {
    /* Pawn check: two diagonal squares natively */
    for (int i=-1; i<=1; i+=2) {
        int tgt = sq + ((ac==WHITE) ? -16 : 16) + i;
        if (!sq_is_off(tgt) && board[tgt] && piece_color(board[tgt])==ac && piece_type(board[tgt])==PAWN) return 1;
    }
    /* Unified Ray-Tracing for Knights, Bishops, Rooks, Kings, and Queens */
    for (int i = piece_offsets[KNIGHT]; i < piece_limits[KING]; i++) {
        int step = step_dir[i], tgt = sq + step;
        while (!sq_is_off(tgt)) {
            int p = board[tgt];
            if (p) {
                if (piece_color(p) == ac) {
                    int pt = piece_type(p);
                    /* The direction index i tells us which piece types
                       can attack along this particular ray or jump. */
                    if (i < piece_limits[KNIGHT] && pt == KNIGHT) return 1;
                    if (i >= piece_offsets[BISHOP] && pt == QUEEN) return 1;
                    if (i >= piece_offsets[BISHOP] && i < piece_limits[BISHOP] && pt == BISHOP) return 1;
                    if (i >= piece_offsets[ROOK]   && i < piece_limits[ROOK]   && pt == ROOK) return 1;
                    if (i >= piece_offsets[KING]   && pt == KING) return 1;
                }
                break; /* A piece blocked the ray */
            }
            /* Leapers cannot slide: break after checking one square */
            if (i < piece_limits[KNIGHT] || i >= piece_offsets[KING]) break;
            tgt += step;
        }
    }
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
static inline int in_check(int s)   { return is_square_attacked(king_sq[s],    s ^ 1); }
static inline int is_illegal(void)  { return is_square_attacked(king_sq[xside], side); }

static inline void add_move(Move *list, int *n, int f, int t, int pr) { list[(*n)++] = make_move_enc(f,t,pr); }

/* Add all four promotion possibilities for a pawn move from f to t. */
static void add_promo(Move *list, int *n, int f, int t) {
    for (int pr = QUEEN; pr >= KNIGHT; pr--) {
        add_move(list, n, f, t, pr);
    }
}

/* XOR piece (color c, type p) at sq in or out of the running Zobrist hash.
   Because XOR is self-inverse, toggling the same value twice cancels out,
   which is exactly what make_move uses when it moves a piece from f to t:
       toggle(side, pt, f)  -- remove from source
       toggle(side, pt, t)  -- place at destination                        */
static inline void toggle(int c, int p, int sq) { hash_key ^= zobrist_piece[c][p][sq]; }

void make_move(Move m) {
    int f=move_from(m), t=move_to(m), pr=move_promo(m), p=board[f], pt=piece_type(p), cap=board[t];
    history[ply].move = m; history[ply].piece_captured = cap; history[ply].ep_square_prev = ep_square;
    history[ply].castle_rights_prev = castle_rights; history[ply].halfmove_clock_prev = halfmove_clock; history[ply].hash_prev = hash_key;
    history[ply].npc_prev[WHITE]=non_pawn_count[WHITE]; history[ply].npc_prev[BLACK]=non_pawn_count[BLACK];
    halfmove_clock = (pt == PAWN || cap) ? 0 : halfmove_clock + 1;

    if (pt==PAWN && t==ep_square) {
        int ep_pawn = t + (side==WHITE ? -16 : 16);
        history[ply].piece_captured = board[ep_pawn]; board[ep_pawn] = EMPTY;
        toggle(xside, PAWN, ep_pawn);
    }
    board[t]=p; board[f]=EMPTY;
    toggle(side, pt, f); toggle(side, pt, t);
    if (cap) { toggle(xside, piece_type(cap), t); if (piece_type(cap)>=KNIGHT && piece_type(cap)<=QUEEN) non_pawn_count[xside]--; }

    if (pr) { board[t] = make_piece(side,pr); toggle(side, pt, t); toggle(side, pr, t);
              non_pawn_count[side]++; } /* pawn promoted to piece */

    hash_key ^= zobrist_castle[castle_rights];
    if (pt==KING) {
        king_sq[side] = t;
        for (int ci=0; ci<4; ci++) {
            if (f==castle_kf[ci] && t==castle_kt[ci]) {
                board[castle_rf[ci]] = EMPTY; board[castle_rt[ci]] = make_piece(castle_col[ci], ROOK);
                toggle(castle_col[ci], ROOK, castle_rf[ci]); toggle(castle_col[ci], ROOK, castle_rt[ci]);
                break;
            }
        }
        castle_rights &= castle_kmask[side*2]; /* WHITE=0->index 0, BLACK=1->index 2 */
    }
    for (int ci=0; ci<4; ci++) if (f==cr_sq[ci] || t==cr_sq[ci]) castle_rights &= cr_mask[ci]; /* Strip castling */
    hash_key ^= zobrist_castle[castle_rights];

    if (ep_square!=SQ_NONE) hash_key ^= zobrist_ep[ep_square];
    ep_square = SQ_NONE;
    if (pt==PAWN && ((t-f)==32 || (f-t)==32)) { ep_square = f + (side==WHITE ? 16 : -16); hash_key ^= zobrist_ep[ep_square]; }

    hash_key ^= zobrist_side; side^=1; xside^=1; ply++;
}

void undo_move(void) {
    ply--; side^=1; xside^=1;
    Move m = history[ply].move; int f=move_from(m), t=move_to(m), pr=move_promo(m);
    board[f]=board[t]; board[t]=history[ply].piece_captured;
    if (pr) board[f]=make_piece(side,PAWN);
    int pt=piece_type(board[f]);

    if (pt==PAWN && t==history[ply].ep_square_prev) {
        board[t]=EMPTY; board[t+(side==WHITE?-16:16)] = history[ply].piece_captured;
    }
    if (pt==KING) {
        king_sq[side]=f;
        for (int ci=0; ci<4; ci++) {
            if (f==castle_kf[ci] && t==castle_kt[ci]) {
                board[castle_rt[ci]] = EMPTY; board[castle_rf[ci]] = make_piece(castle_col[ci], ROOK); break;
            }
        }
    }
    ep_square = history[ply].ep_square_prev; castle_rights = history[ply].castle_rights_prev;
    halfmove_clock = history[ply].halfmove_clock_prev;
    non_pawn_count[WHITE]=history[ply].npc_prev[WHITE]; non_pawn_count[BLACK]=history[ply].npc_prev[BLACK];
    hash_key = history[ply].hash_prev; /* O(1) restore */
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

int generate_moves(Move *moves, int caps_only) {
    int cnt=0, sq;
    int d_pawn     = (side==WHITE) ?  16 : -16;
    int pawn_start = (side==WHITE) ?   1 :  6;
    int pawn_promo = (side==WHITE) ?   6 :  1;

    FOR_EACH_SQ(sq) {
        int p=board[sq];
        if (!p || piece_color(p)!=side) continue;
        int pt=piece_type(p);

        /* -- Pawns ------------------------------------------------ */
        if (pt==PAWN) {
            int tgt=sq+d_pawn;
            if (!sq_is_off(tgt) && !board[tgt]) {
                if ((sq>>4)==pawn_promo) add_promo(moves, &cnt, sq, tgt);
                else if (!caps_only) {
                    add_move(moves,&cnt,sq,tgt,0);
                    if ((sq>>4)==pawn_start && !board[tgt+d_pawn]) add_move(moves,&cnt,sq,tgt+d_pawn,0);
                }
            }
            for (int i=-1; i<=1; i+=2) {           /* diagonal captures + ep */
                tgt=sq+d_pawn+i;
                if (!sq_is_off(tgt) && ((board[tgt] && piece_color(board[tgt])==xside) || tgt==ep_square)) {
                    if ((sq>>4)==pawn_promo) add_promo(moves, &cnt, sq, tgt);
                    else add_move(moves,&cnt,sq,tgt,0);
                }
            }
            continue;
        }

        /* -- Sliders & Leapers ------------------------------------ */
        for (int i=piece_offsets[pt]; i<piece_limits[pt]; i++) {
            int step=step_dir[i], tgt=sq+step;
            while (!sq_is_off(tgt)) {
                if (!board[tgt]) {
                    if (!caps_only) add_move(moves,&cnt,sq,tgt,0);
                } else {
                    if (piece_color(board[tgt])==xside) add_move(moves,&cnt,sq,tgt,0);
                    break;
                }
                if (pt==KNIGHT || pt==KING) break;
                tgt+=step;
            }
        }

        /* -- Castling (king only, never in caps_only mode) -------- */
        if (pt==KING && !caps_only) {
            int kf, kt, rf, bit, ac, clear_ok;
            for (int ci=0; ci<4; ci++) {
                kf=castle_kf[ci]; kt=castle_kt[ci]; rf=castle_rf[ci];
                bit = (ci==0)?1:(ci==1)?2:(ci==2)?4:8;
                ac  = (castle_col[ci]==WHITE) ? BLACK : WHITE;
                
                if (sq != kf || castle_col[ci]!=side) continue;
                if (!(castle_rights & (unsigned int)bit)) continue;
                if (board[rf] != make_piece(side,ROOK)) continue;

                /* Every square between king and rook must be empty */
                int sq1=(kf<rf)? kf+1 : rf+1, sq2=(kf<rf)? rf   : kf;
                clear_ok = 1;
                for (int sq3=sq1; sq3<sq2; sq3++)
                    if (board[sq3]) { clear_ok=0; break; }
                if (!clear_ok) continue;

                /* King's path must not traverse attacked squares */
                int step2 = (kt>kf) ? 1 : -1;
                clear_ok = 1;
                for (int sq3=kf; sq3!=(kt+step2); sq3+=step2)
                    if (is_square_attacked(sq3,ac)) { clear_ok=0; break; }
                if (clear_ok) add_move(moves,&cnt,kf,kt,0);
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
    switch (lo) {
        case 'p': return PAWN;
        case 'n': return KNIGHT;
        case 'b': return BISHOP;
        case 'r': return ROOK;
        case 'q': return QUEEN;
        case 'k': return KING;
        default:  return EMPTY;
    }
}

void parse_fen(const char *fen) {
    int rank=7, file=0;

    for (int i=0;i<128;i++) board[i]=EMPTY;
    king_sq[WHITE] = king_sq[BLACK] = SQ_NONE;
    castle_rights=0; ep_square=SQ_NONE; ply=0; hash_key=0;
    non_pawn_count[WHITE]=0; non_pawn_count[BLACK]=0;
    memset(killers,0,sizeof(killers)); memset(pv,0,sizeof(pv));
    memset(pv_length,0,sizeof(pv_length)); memset(hist,0,sizeof(hist));

    while (*fen && *fen!=' ') {
        if (*fen=='/') { file=0; rank--; }
        else if (isdigit(*fen)) { file += *fen-'0'; }
        else {
            int sq=rank*16+file, color=isupper(*fen)?WHITE:BLACK; char lo=(char)tolower(*fen);
            int piece = char_to_piece(lo);
            if (piece == EMPTY) { fen++; continue; }
            board[sq]=make_piece(color,piece); if (piece==KING) king_sq[color]=sq;
            if (piece>=KNIGHT && piece<=QUEEN) non_pawn_count[color]++;
            file++;
        }
        fen++;
    }
    if (*fen) fen++;

    side=(*fen=='w')?WHITE:BLACK; xside=side^1;
    if (*fen) fen++;
    if (*fen) fen++;

    while (*fen && *fen!=' ') {
        if (*fen=='K') { castle_rights|=1; } if (*fen=='Q') { castle_rights|=2; }
        if (*fen=='k') { castle_rights|=4; } if (*fen=='q') { castle_rights|=8; }
        fen++;
    }
    if (*fen) fen++;

    if (*fen != '-' && *fen && fen[1])
        ep_square = (fen[1]-'1')*16 + (fen[0]-'a');

    /* advance past ep field, then read halfmove clock */
    while (*fen && *fen != ' ') fen++;
    halfmove_clock = 0;
    if (*fen == ' ') { fen++; halfmove_clock = atoi(fen); }
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
static const int mg_pst[6][64] = {
  {   0,  0,  0,  0,  0,  0,  0,  0,  -35, -1,-20,-23,-15, 24, 38,-22,  /* pawn   r1-r2 */
    -26, -4, -4,-10,  3,  3, 33,-12,  -27, -2, -5, 12, 17,  6, 10,-25,  /*        r3-r4 */
    -14, 13,  6, 21, 23, 12, 17,-23,   -6,  7, 26, 31, 65, 56, 25,-20,  /*        r5-r6 */
     98,134, 61, 95, 68,126, 34,-11,    0,  0,  0,  0,  0,  0,  0,  0}, /*        r7-r8 */
  {-105,-21,-58,-33,-17,-28,-19,-23,  -29,-53,-12, -3, -1, 18,-14,-19,  /* knight r1-r2 */
    -23, -9, 12, 10, 19, 17, 25,-16,  -13,  4, 16, 13, 28, 19, 21, -8,  /*        r3-r4 */
     -9, 17, 19, 53, 37, 69, 18, 22,  -47, 60, 37, 65, 84,129, 73, 44,  /*        r5-r6 */
    -73,-41, 72, 36, 23, 62,  7,-17, -167,-89,-34,-49, 61,-97,-15,-107},/*        r7-r8 */
  { -33, -3,-14,-21,-13,-12,-39,-21,    4, 15, 16,  0,  7, 21, 33,  1,  /* bishop r1-r2 */
      0, 15, 15, 15, 14, 27, 18, 10,   -6, 13, 13, 26, 34, 12, 10,  4,  /*        r3-r4 */
     -4,  5, 19, 50, 37, 37,  7, -2,  -16, 37, 43, 40, 35, 50, 37, -2,  /*        r5-r6 */
    -26, 16,-18,-13, 30, 59, 18,-47,  -29,  4,-82,-37,-25,-42,  7, -8}, /*        r7-r8 */
  { -19,-13,  1, 17, 16,  7,-37,-26,  -44,-16,-20, -9, -1, 11, -6,-71,  /* rook   r1-r2 */
    -45,-25,-16,-17,  3,  0, -5,-33,  -36,-26,-12, -1,  9, -7,  6,-23,  /*        r3-r4 */
    -24,-11,  7, 26, 24, 35, -8,-20,   -5, 19, 26, 36, 17, 45, 61, 16,  /*        r5-r6 */
     27, 32, 58, 62, 80, 67, 26, 44,   32, 42, 32, 51, 63,  9, 31, 43}, /*        r7-r8 */
  {  -1,-18, -9, 10,-15,-25,-31,-50,  -35, -8, 11,  2,  8, 15, -3,  1,  /* queen  r1-r2 */
    -14,  2,-11, -2, -5,  2, 14,  5,   -9,-26, -9,-10, -2, -4,  3, -3,  /*        r3-r4 */
    -27,-27,-16,-16, -1, 17, -2,  1,  -13,-17,  7,  8, 29, 56, 47, 57,  /*        r5-r6 */
    -24,-39, -5,  1,-16, 57, 28, 54,  -28,  0, 29, 12, 59, 44, 43, 45}, /*        r7-r8 */
  { -15, 36, 12,-54,  8,-28, 24, 14,    1,  7, -8,-64,-43,-16,  9,  8,  /* king   r1-r2 */
    -14,-14,-22,-46,-44,-30,-15,-27,  -49, -1,-27,-39,-46,-44,-33,-51,  /*        r3-r4 */
    -17,-20,-12,-27,-30,-25,-14,-36,   -9, 24,  2,-16,-20,  6, 22,-22,  /*        r5-r6 */
     29, -1,-20, -7, -8, -4,-38,-29,  -65, 23, 16,-15,-56,-34,  2, 13}  /*        r7-r8 */
};

/* eg_pst[piece-1][sq]: endgame, same layout */
static const int eg_pst[6][64] = {
  {   0,  0,  0,  0,  0,  0,  0,  0,   13,  8,  8, 10, 13,  0,  2, -7,  /* pawn   r1-r2 */
      4,  7, -6,  1,  0, -5, -1, -8,   13,  9, -3, -7, -7, -8,  3, -1,  /*        r3-r4 */
     32, 24, 13,  5, -2,  4, 17, 17,   94,100, 85, 67, 56, 53, 82, 84,  /*        r5-r6 */
    178,173,158,134,147,132,165,187,    0,  0,  0,  0,  0,  0,  0,  0}, /*        r7-r8 */
  { -29,-51,-23,-15,-22,-18,-50,-64,  -42,-20,-10, -5, -2,-20,-23,-44,  /* knight r1-r2 */
    -23, -3, -1, 15, 10, -3,-20,-22,  -18, -6, 16, 25, 16, 17,  4,-18,  /*        r3-r4 */
    -17,  3, 22, 22, 22, 11,  8,-18,  -24,-20, 10,  9, -1, -9,-19,-41,  /*        r5-r6 */
    -25, -8,-25, -2, -9,-25,-24,-52,  -58,-38,-13,-28,-31,-27,-63,-99}, /*        r7-r8 */
  { -23, -9,-23, -5, -9,-16, -5,-17,  -14,-18, -7, -1,  4, -9,-15,-27,  /* bishop r1-r2 */
    -12, -3,  8, 10, 13,  3, -7,-15,   -6,  3, 13, 19,  7, 10, -3, -9,  /*        r3-r4 */
     -3,  9, 12,  9, 14, 10,  3,  2,    2, -8,  0, -1, -2,  6,  0,  4,  /*        r5-r6 */
     -8, -4,  7,-12, -3,-13, -4,-14,  -14,-21,-11, -8, -7, -9,-17,-24}, /*        r7-r8 */
  {  -9,  2,  3, -1, -5,-13,  4,-20,   -6, -6,  0,  2, -9, -9,-11, -3,  /* rook   r1-r2 */
     -4,  0, -5, -1, -7,-12, -8,-16,    3,  5,  8,  4, -5, -6, -8,-11,  /*        r3-r4 */
      4,  3, 13,  1,  2,  1, -1,  2,    7,  7,  7,  5,  4, -3, -5, -3,  /*        r5-r6 */
     11, 13, 13, 11, -3,  3,  8,  3,   13, 10, 18, 15, 12, 12,  8,  5}, /*        r7-r8 */
  { -33,-28,-22,-43, -5,-32,-20,-41,  -22,-23,-30,-16,-16,-23,-36,-32,  /* queen  r1-r2 */
    -16,-27, 15,  6,  9, 17, 10,  5,  -18, 28, 19, 47, 31, 34, 39, 23,  /*        r3-r4 */
      3, 22, 24, 45, 57, 40, 57, 36,  -20,  6,  9, 49, 47, 35, 19,  9,  /*        r5-r6 */
    -17, 20, 32, 41, 58, 25, 30,  0,   -9, 22, 22, 27, 27, 19, 10, 20}, /*        r7-r8 */
  { -53,-34,-21,-11,-28,-14,-24,-43,  -27,-11,  4, 13, 14,  4, -5,-17,  /* king   r1-r2 */
    -19, -3, 11, 21, 23, 16,  7, -9,  -18, -4, 21, 24, 27, 23,  9,-11,  /*        r3-r4 */
     -8, 22, 24, 27, 26, 33, 26,  3,   10, 17, 23, 15, 20, 45, 44, 13,  /*        r5-r6 */
    -12, 17, 14, 17, 17, 38, 23, 11,  -74,-35,-18,-18,-11, 15,  4,-17}  /*        r7-r8 */
};

/* Separate MG/EG material values (Rofchade).
   piece_val[] is kept unchanged for MVV-LVA move ordering. */
static const int mg_val[6] = {82, 337, 365, 477, 1025,    0};
static const int eg_val[6] = {94, 281, 297, 512,  936,    0};

/* Phase contribution per piece type (indexed by TYPE(): 1=pawn..6=king).
   knight=1, bishop=1, rook=2, queen=4; max total = 24. */
static const int phase_inc[7] = {0, 0, 1, 1, 2, 4, 0};

/* Piece value table for MVV-LVA move ordering (unchanged) */
static const int piece_val[7] = {0,100,320,330,500,900,20000};

/* Mobility centering offsets: subtract typical reachable-square count so
   inactive pieces are penalised rather than all pieces getting a flat bonus.
   Indexed by TYPE(): 0=empty,1=pawn,2=knight,3=bishop,4=rook,5=queen,6=king */
static const int mob_center[7] = {0, 0, 4, 6, 6, 13, 0};

int evaluate(void) {
    int mg[2], eg[2], phase;
    int bishops[2];
    int pawn_cnt[2][8];
    int pseudo_list[32]; /* occupied squares built during first pass (Pawel Koziol) */
    int index = 0, i;

    mg[WHITE] = mg[BLACK] = eg[WHITE] = eg[BLACK] = phase = 0;
    bishops[WHITE] = bishops[BLACK] = 0;
    memset(pawn_cnt, 0, sizeof(pawn_cnt));

    /* First pass: rank/file double loop visits exactly 64 valid squares
       instead of 128 (Pawel: FOR_EACH_SQ loops the empty half too).
       Occupied squares are recorded into pseudo_list as we go, so the
       rook pass below never touches a single empty square.              */
    for (int rank = 0; rank < 8; rank++) {
        for (int f = 0; f < 8; f++) {
            int sq = rank * 16 + f;
            int p = board[sq]; if (!p) continue;
            pseudo_list[index++] = sq;                    /* memorize (Pawel) */
            int pt = piece_type(p), color = piece_color(p);

            /* Square index: rank 0 = White's back rank.
               Black mirrors vertically so its rank 0 is rank 7 in White terms. */
            int idx = (color == WHITE) ? rank * 8 + f : (7 - rank) * 8 + f;

            /* Material + PST: scored into MG and EG accumulators separately.
               pt-1 converts TYPE() (1-based) to the 0-based table index. */
            mg[color] += mg_val[pt-1] + mg_pst[pt-1][idx];
            eg[color] += eg_val[pt-1] + eg_pst[pt-1][idx];
            phase     += phase_inc[pt];

            /* Mobility: count pseudo-legal reachable squares, centered so that
               a piece with exactly mob_center[pt] squares scores zero.
               Pinned pieces appear more mobile than they are, but the
               approximation is cheap and consistently directional. */
            if (pt >= KNIGHT && pt <= QUEEN) {
                int mob = 0;
                for (i = piece_offsets[pt]; i < piece_limits[pt]; i++) {
                    int step = step_dir[i], target = sq + step;
                    while (!sq_is_off(target)) {
                        if (board[target] == 0) { mob++; }
                        else { if (piece_color(board[target]) != color) mob++; break; }
                        if (pt == KNIGHT) break;
                        target += step;
                    }
                }
                mob -= mob_center[pt];
                mg[color] += (pt == QUEEN ? 2 : 3) * mob;
                eg[color] += (pt == QUEEN ? 2 : 3) * mob;
            }

            if (pt == BISHOP) {
                bishops[color]++;
                if (bishops[color] == 2) { mg[color] += 30; eg[color] += 30; } /* bishop pair */
            }
            if (pt == PAWN) pawn_cnt[color][f]++;
        }
    }

    /* Pawn structure and king safety (requires full pawn_cnt) */
    for (int color = 0; color < 2; color++) {
        for (int f = 0; f < 8; f++) {
            int cnt = pawn_cnt[color][f];
            if (cnt > 1) { mg[color] -= (cnt-1)*20; eg[color] -= (cnt-1)*20; } /* Doubled */
            if (cnt) {
                int left  = (f > 0) ? pawn_cnt[color][f-1] : 0;
                int right = (f < 7) ? pawn_cnt[color][f+1] : 0;
                if (!left && !right) { mg[color] -= 10; eg[color] -= 10; } /* Isolated */
            }
        }
        /* King pawn shield -- MG only.
           In the endgame, king centralisation is already rewarded by the
           EG king PST; a pawn shield is irrelevant and would only hurt. */
        {
            int ksq = king_sq[color], kf = ksq & 7;
            if (kf <= 2 || kf >= 5) {
                int penalty = 0;
                for (int f_test = kf-1; f_test <= kf+1; f_test++) {
                    if (f_test >= 0 && f_test <= 7 && pawn_cnt[color][f_test] == 0) {
                        penalty += 15;
                        penalty += (pawn_cnt[color^1][f_test] == 0) ? 25 : 10;
                    }
                }
                mg[color] -= penalty;
            }
        }
    }

    /* Rook activity: iterate pseudo_list -- only occupied squares, never
       the 32+ empty squares FOR_EACH_SQ would visit (Pawel Koziol).
       Applied to both MG and EG: an open rook is valuable in all phases. */
    for (i = 0; i < index; i++) {
        int sq = pseudo_list[i], p = board[sq];
        int pt = piece_type(p), color = piece_color(p), f = sq & 7;
        if (pt == ROOK) {
            int rank = sq >> 4, bonus = 0;
            if (pawn_cnt[color][f] == 0)
                bonus += (pawn_cnt[color^1][f] == 0) ? 20 : 10; /* open/semi-open file */
            if ((color == WHITE && rank == 6) || (color == BLACK && rank == 1))
                bonus += 20; /* 7th rank */
            mg[color] += bonus; eg[color] += bonus;
        }
    }

    /* Tapered blend.
       phase clamps to [0,24]: values beyond 24 (e.g. at game start) are
       treated as full middlegame.  The interpolation formula:
           (mg_score * phase + eg_score * (24 - phase)) / 24
       gives pure MG at phase=24 and pure EG at phase=0. */
    if (phase > 24) phase = 24;
    int mg_score = mg[side] - mg[side^1];
    int eg_score = eg[side] - eg[side^1];
    return (mg_score * phase + eg_score * (24 - phase)) / 24;
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

static inline int score_move(Move m, Move hash_move, int sply) {
    int cap, sc=0;
    if (m==hash_move) return 30000;
    cap=board[move_to(m)];
    /* EP captures land on an empty square; treat them as pawn captures for ordering. */
    if (!cap && piece_type(board[move_from(m)])==PAWN && move_to(m)==ep_square)
        cap=make_piece(xside,PAWN);
    if (cap)                sc=20000+10*piece_val[piece_type(cap)]-piece_val[piece_type(board[move_from(m)])];
    else if (move_promo(m)) sc=19999;
    else if (sply<MAX_PLY && m==killers[sply][0]) sc=19998;
    else if (sply<MAX_PLY && m==killers[sply][1]) sc=19997;
    else                    sc=hist[move_from(m)][move_to(m)];  /* [-16000, 16000] */
    return sc;
}

/* Score all moves into a parallel array. Called once before the move loop. */
static void score_moves(Move *moves, int *scores, int n, Move hash_move, int sply) {
    for (int i = 0; i < n; i++) scores[i] = score_move(moves[i], hash_move, sply);
}

/* Partial sort: swap the best remaining move to position idx.
   Called once per move inside the loop -- O(n) per pick vs O(n^2) total
   for selection sort, but we only pay for moves we actually search.      */
static void pick_move(Move *moves, int *scores, int n, int idx) {
    int best = idx;
    for (int i = idx+1; i < n; i++)
        if (scores[i] > scores[best]) best = i;
    if (best != idx) {
        int ts = scores[idx]; scores[idx] = scores[best]; scores[best] = ts;
        Move tm = moves[idx];  moves[idx]  = moves[best];  moves[best]  = tm;
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
   ADAPTIVE LMR REDUCTION TABLE
   Reduction R scales with both search depth and move number:
     R = round(ln(depth) * ln(move_number) / 1.6), clamped to [1, 5].
   No reduction for depth < 3 or move_number < 4 (threshold guard).
   "Adaptive" because R grows continuously rather than using fixed
   buckets (0 / 1 / 2), better matching the empirical cost-benefit
   curve of late-move pruning at increasing depths.
--------------------------------------------------------------- */
static int lmr_table[32][64];

static void init_lmr(void) {
    int d, m;
    for (d = 0; d < 32; d++)
        for (m = 0; m < 64; m++) {
            if (d < 3 || m < 4) { lmr_table[d][m] = 0; continue; }
            double r = log((double)d) * log((double)m) / 1.6;
            int ri = (int)(r + 0.5);   /* round */
            lmr_table[d][m] = ri < 1 ? 1 : ri > 5 ? 5 : ri;
        }
}

int search(int depth, int alpha, int beta, int was_null, int sply) {
    Move moves[256], best=0, hash_move=0;
    int legal=0, quiet=0, best_sc, old_alpha=alpha, sc;
    int is_pv = (beta - alpha > 1); /* PV node: wide window, not a null-window probe */
    TTEntry *e = &tt[hash_key % (HASH)tt_size];

    /* Clear PV at this ply before any early returns (TT hits, stand-pat, repetition).
       If we return early the parent reads pv_length[sply] to know how much of the
       child continuation to copy; it must equal sply (empty) not a stale value. */
    pv_length[sply] = sply;

    /* HARD TIME LIMIT CHECK
       Every 1024 nodes, check if we have exceeded our absolute time budget.
       If we have, abort the search tree immediately to prevent flagging. */
    if ((nodes_searched & 1023) == 0 && time_budget_ms > 0) {
        int64_t ms = (int64_t)(((int64_t)(clock() - t_start) * 1000) / CLOCKS_PER_SEC);
        if (ms >= time_budget_ms) { time_over_flag = 1; return 0; }
    }
    if (time_over_flag) return 0;

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
        for (int i = ply - 2; i >= root_ply; i -= 2)
            if (history[i].hash_prev == hash_key) return 0;
        {
            int reps = 0;
            for (int i = ply - 2; i >= 0 && i >= ply - halfmove_clock; i -= 2)
                if (history[i].hash_prev == hash_key && ++reps >= 2) return 0;
        }
    }

    /* 50-move rule */
    if (halfmove_clock >= 100) return 0;

    /* INSUFFICIENT MATERIAL
       KK, KNK, KBK -- no pawns and at most one minor piece.
       KRK and KQK are NOT draws -- rooks and queens can force checkmate.
       non_pawn_count is maintained incrementally so the common case is O(1);
       the board scan only runs when the count actually qualifies.            */
    if (non_pawn_count[WHITE] + non_pawn_count[BLACK] <= 1) {
        int s2, has_pawn = 0, has_major = 0;
        FOR_EACH_SQ(s2) {
            int pt = piece_type(board[s2]);
            if (pt == PAWN)                has_pawn  = 1;
            if (pt == ROOK || pt == QUEEN) has_major = 1;
        }
        if (!has_pawn && !has_major) return 0;
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
            if (tt_sc >  MATE - MAX_PLY) tt_sc -= sply;
            if (tt_sc < -(MATE - MAX_PLY)) tt_sc += sply;
            if (flag == TT_EXACT)                           return tt_sc;
            if (!is_pv && flag == TT_BETA  && tt_sc >= beta) return tt_sc;
            if (!is_pv && flag == TT_ALPHA && tt_sc <= alpha) return tt_sc;
        }
    }

    /* Quiescence: stand-pat evaluation when out of depth */
    int caps_only = (depth <= 0);
    if (caps_only) {
        best_sc = evaluate();
        if (best_sc >= beta) return best_sc;
        if (best_sc > alpha) alpha = best_sc;
        /* if (depth < -6) return best_sc;    max quiescence depth cap */
    } else {
        best_sc = -INF;
    }

    nodes_searched++;

    /* REVERSE FUTILITY PRUNING (RFP / Static Null Move Pruning)
       static_eval is computed once and reused; no second evaluate() call. */
    if (!caps_only && depth >= 1 && depth <= 7
        && beta < MATE - MAX_PLY
        && !in_check(side)) {
        int static_eval = evaluate();
        if (static_eval - 70 * depth >= beta)
            return static_eval - 70 * depth;
    }

    /* NULL MOVE PRUNING (NMP)
       Skip our turn; if the opponent still cannot beat beta at reduced
       depth, prune immediately. R=2 normally, R=3 at depth >= 6.

       ZUGZWANG GUARD: pure pawn endgames can be genuine zugzwang where
       passing really is the worst move. We skip NMP when the side to
       move has no non-pawn non-king piece, making the null move
       assumption safe in all normal middlegame and endgame positions.  */
    if (!caps_only && !is_pv && !was_null && depth >= 3 && non_pawn_count[side] > 0
        && !in_check(side)) {
        int R = (depth >= 7) ? 4 : 3;
        int ep_sq_prev = ep_square;
        hash_key ^= zobrist_side;
        if (ep_square != SQ_NONE) hash_key ^= zobrist_ep[ep_square];
        ep_square = SQ_NONE;
        side ^= 1; xside ^= 1;
        sc = -search(depth - R - 1, -beta, -beta + 1, 1, sply + 1);
        side ^= 1; xside ^= 1;
        ep_square = ep_sq_prev;
        if (ep_square != SQ_NONE) hash_key ^= zobrist_ep[ep_square];
        hash_key ^= zobrist_side;
        if (sc >= beta) return sc;  /* fail-soft: return actual score, not beta */
    }

    int cnt = generate_moves(moves, caps_only);
    int scores[256];
    score_moves(moves, scores, cnt, hash_move, sply);
    /* Track quiet moves in order so the malus loop has an explicit list
       (avoids re-checking board[] state after moves are undone). */
    Move quiet_moves[256]; int nquiet = 0;

    for (int i = 0; i < cnt; i++) {
        pick_move(moves, scores, cnt, i);
        /* DELTA PRUNING (Quiescence only)
           If capturing this piece plus a safety margin can't possibly
           raise alpha, skip generating the recursive tree. */
        if (caps_only && board[move_to(moves[i])]) {
            int cap_val = piece_val[piece_type(board[move_to(moves[i])])];
            if (best_sc + cap_val + 200 < alpha) continue;
        }

        /* Capture flag must be read before make_move: after the call
           board[TO] always holds a piece, making a post-move test useless.
           En-passant has an empty destination, so check ep_square too.    */
        int is_cap = board[move_to(moves[i])] != 0
                     || (piece_type(board[move_from(moves[i])]) == PAWN
                         && move_to(moves[i]) == ep_square);
        make_move(moves[i]);
        if (is_illegal()) { undo_move(); continue; }
        legal++;
        if (!is_cap) { quiet++; if (nquiet < 256) quiet_moves[nquiet++] = moves[i]; }

        /* LATE MOVE PRUNING (LMP)
           At shallow depths, skip quiet moves beyond the first few.
           The threshold scales with depth so we never prune at depth 1
           (4*1+1=5 quiet moves allowed) through depth 3 (13 allowed).
           Moves that give check are exempted: they may be the only defence. */
        if (!caps_only && !is_pv && depth < 4 && quiet > 4 * depth + 1) {
            if (!in_check(side)) { undo_move(); continue; }
        }

        /* PRINCIPAL VARIATION SEARCH + LMR
           First legal move searched with full window to establish the PV.
           All subsequent moves use a null window (-alpha-1,-alpha) since
           if our current best is truly best they should fail low cheaply.
           The `sc < beta` upper-bound guard is omitted intentionally: 
           in modern PVS the re-search will simply fail-high and that score 
           is still valid (it is >= beta, which the parent will cut off anyway). */
        if (caps_only || legal == 1) {
            sc = -search(depth - 1, -beta, -alpha, 0, sply + 1);
        } else {
            /* Adaptive LMR: late quiet moves are searched at a reduced depth
               R drawn from lmr_table (log-scaled by depth and move number).
               Any move whose reduced score beats alpha is re-searched at full
               depth to get an exact score before updating alpha/best. */

            int is_reduced = 0; // are we doing a reduced search?
            int gives_check = in_check(side);
            /* ADAPTIVE LMR: reduction R = lmr_table[depth][move_number],
               precomputed as round(ln(d)*ln(m)/1.6).  Captures, promotions,
               and check-giving moves skip reduction entirely.  R is capped so
               we never reduce past depth 1 (i.e. never drop into qsearch). */
            int d_clamped = depth < 32 ? depth : 31;
            int m_clamped = legal < 64 ? legal : 63;
            int lmr = (!is_cap && !move_promo(moves[i]) && !gives_check)
                      ? lmr_table[d_clamped][m_clamped] : 0;
            
            // Increase reduction in non-PV nodes (they're expected to fail low anyway)
            if (!is_pv && lmr > 0) lmr += 1;
            
            // Never drop to depth 0 or below
            if (lmr > depth - 2) lmr = depth - 2;
            if (lmr < 0) lmr = 0;
            
            if (lmr > 0) {
                // reduced depth + null window
                sc = -search(depth - 1 - lmr, -alpha - 1, -alpha, 0, sply + 1); 
                if (sc <= alpha) is_reduced = 1; // failed low, skip re-search
            }

            if (!is_reduced) {
                // full depth + full window
                sc = -search(depth - 1, -alpha - 1, -alpha, 0, sply + 1);
                if (sc > alpha && is_pv) {
                    // full window only on PV nodes
                    sc = -search(depth - 1, -beta, -alpha, 0, sply + 1);
                }
            }
        }

        undo_move();

        if (sc > best_sc) best_sc = sc;
        if (sc > alpha) {
            alpha = sc;
            best = moves[i];              /* moved here -- only update when alpha raised */
            /* Triangular PV update: store this move, then copy the child
               ply's continuation into the current row of the table. */
            pv[sply][sply] = moves[i];
            for (int k_ = sply+1; k_ < pv_length[sply+1]; k_++)
                pv[sply][k_] = pv[sply+1][k_];
            pv_length[sply] = pv_length[sply+1];
        }
        if (alpha >= beta) {
            if (!board[move_to(moves[i])]) {   /* quiet cutoff move */
                int d = (sply < MAX_PLY) ? sply : MAX_PLY - 1;
                int bonus = depth * depth;
                killers[d][1] = killers[d][0];
                killers[d][0] = moves[i];
                /* History BONUS: reward the cutoff move.
                   History MALUS:  penalise every quiet move tried before it.
                   Both use the gravity formula: base update +/- bonus, then
                   subtract a fraction of the current value (diminishing returns)
                   so entries self-correct instead of saturating at ±16000. */
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

    /* Checkmate or stalemate (only detectable in full search, not QS) */
    if (!caps_only && !legal)
        return in_check(side) ? -(MATE - sply) : 0;

    /* TT store: skip if search was aborted mid-tree (score is meaningless) */
    if (!time_over_flag && best && (e->key != hash_key || depth >= (int)tt_depth(e))) {
        int flag = (best_sc <= old_alpha) ? TT_ALPHA :
                   (best_sc >= beta)      ? TT_BETA  : TT_EXACT;
        /* Encode mate scores as distance-from-node (+sply) so the score
           stays valid when the position is retrieved via a transposition. */
        int sc_store = best_sc;
        if (sc_store >  MATE - MAX_PLY) sc_store += sply;
        if (sc_store < -(MATE - MAX_PLY)) sc_store -= sply;
        e->key = hash_key; e->score = sc_store; e->best_move = best;
        e->depth_flag = tt_pack(depth > 0 ? depth : 0, flag);
    }
    return best_sc;
}

/* ---------------------------------------------------------------
   print_move / print_pv  -- formatting helpers
   ---------------------------------------------------------------
   A chess move in UCI format: <from><to>[promo], e.g. "e2e4", "a7a8q".
   print_pv prints the full principal variation for the info line.
*/

void print_move(Move m) {
    static const char promo_ch[7] = {0,0,'n','b','r','q',0}; /* KNIGHT=2..QUEEN=5 */
    int f=move_from(m), t=move_to(m), pr=move_promo(m);
    printf("%c%c%c%c", 'a'+(f&7), '1'+(f>>4), 'a'+(t&7), '1'+(t>>4));
    if (pr && pr < 7) putchar(promo_ch[pr]);
}

static void print_pv(void) {
    for (int k=0; k<pv_length[0]; k++) {
        putchar(' ');
        print_move(pv[0][k]);
    }
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
    Move moves[256], global_best=0, iter_best;
    int cnt = generate_moves(moves, 0);
    int best_sc = -INF, legal_root = 0, sply = 0;
    int64_t total_nodes = 0;

    time_over_flag = 0;
    t_start = clock();
    memset(hist, 0, sizeof(hist)); memset(killers, 0, sizeof(killers));

    int root_scores[256];
    score_moves(moves, root_scores, cnt, 0, 0);
    root_ply = ply;   /* anchor sply=0 at the search root */

    for (int d=1; d<=max_depth; d++) {
        int asp_alpha, asp_beta, asp_delta, asp_failed;

        /* ASPIRATION WINDOWS
           Depths 1-3 always use a full [-INF, INF] window: scores are
           too volatile at shallow depth for a narrow window to help.
           Depth 4+: open with [prev-50, prev+50]. If the score escapes
           (fail-low or fail-high), double the delta and re-search.
           The sentinel asp_delta=INF disables retry logic for full-window
           depths so we never re-search depths 1-3 unnecessarily.         */
        if (d >= 4 && best_sc > -MATE && best_sc < MATE) {
            asp_delta = 50;
            asp_alpha = best_sc - asp_delta;
            asp_beta  = best_sc + asp_delta;
        } else {
            asp_delta = INF;   /* sentinel: full window, retry logic disabled */
            asp_alpha = -INF;
            asp_beta  =  INF;
        }

        do {
            int asp_lo = asp_alpha;
            int asp_hi = asp_beta;
            asp_failed     = 0;
            best_sc        = -INF;
            iter_best      = 0;
            nodes_searched = 0;
            memset(pv,        0, sizeof(pv));
            memset(pv_length, 0, sizeof(pv_length));

            if (global_best) {
                for (int i=0;i<cnt;i++)
                    if (moves[i]==global_best) {
                        Move tmp=moves[0]; moves[0]=moves[i]; moves[i]=tmp; break;
                    }
            }

            for (int i=0; i<cnt; i++) {
                pick_move(moves, root_scores, cnt, i);
                make_move(moves[i]);
                if (is_illegal()) { undo_move(); continue; }
                if (d==1) {
                    legal_root++;
                    if (!global_best) global_best = moves[i]; /* Immediate fallback */
                }
                /* PVS at root: first legal move gets the full aspiration window;
                   all subsequent moves get a null-window probe first and only
                   re-search with the full window on a fail-high. */
                int sc;
                if (i == 0 || best_sc == -INF) {
                    sc = -search(d-1, -asp_hi, -asp_lo, 0, sply + 1);
                } else {
                    sc = -search(d-1, -asp_lo-1, -asp_lo, 0, sply + 1);
                    if (!time_over_flag && sc > asp_lo && sc < asp_hi)
                        sc = -search(d-1, -asp_hi, -asp_lo, 0, sply + 1);
                }
                undo_move();
                if (time_over_flag) break; /* Abort this depth entirely */
                if (sc>best_sc) best_sc=sc;
                if (sc > asp_lo) {
                    asp_lo = sc;
                    iter_best = moves[i];         /* only when alpha raised */
                    /* Propagate PV from depth 1 up to the root (sply=0) */
                    pv[0][0]=moves[i];
                    for (int k=1; k<pv_length[1]; k++) pv[0][k]=pv[1][k];
                    pv_length[0]=pv_length[1];
                }
            }

            if (time_over_flag) break; /* Don't use incomplete depth data */

            /* Fail-low or fail-high: widen the relevant side and retry.
               The sentinel (asp_delta==INF) skips this for full windows. */
            if (asp_delta != INF) {
                if (best_sc < asp_alpha) {        /* asp_alpha unchanged = original floor */
                    asp_delta *= 2;
                    asp_alpha  = best_sc - asp_delta;
                    if (asp_alpha < -INF/2) asp_alpha = -INF;
                    asp_failed = 1;
                } else if (best_sc >= asp_beta) {
                    asp_delta *= 2;
                    asp_beta   = best_sc + asp_delta;
                    if (asp_beta > INF/2) asp_beta = INF;
                    asp_failed = 1;
                }
            }
        } while (asp_failed && !time_over_flag);

        total_nodes += nodes_searched;
        if (time_over_flag) break; /* Don't use incomplete depth data */

        if (iter_best) global_best=iter_best;

        /* No legal moves at root: terminal position.
           Skip the info line -- the mate/stalemate report below handles it. */
        if (!legal_root) break;

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
             mated:   N = -(MATE + score + 1) / 2    */
            int64_t ms = (int64_t)(((int64_t)(clock()-t_start)*1000) / CLOCKS_PER_SEC);
            if (best_sc > MATE - MAX_PLY)
                printf("info depth %d score mate %d nodes %" PRId64 " time %" PRId64 " pv",
                       d, (MATE - best_sc + 1)/2, total_nodes, ms);
            else if (best_sc < -(MATE - MAX_PLY))
                printf("info depth %d score mate %d nodes %" PRId64 " time %" PRId64 " pv",
                       d, -(MATE + best_sc + 1)/2, total_nodes, ms);
            else
                printf("info depth %d score cp %d nodes %" PRId64 " time %" PRId64 " pv",
                       d, best_sc, total_nodes, ms);
            print_pv();
            printf("\n");
            fflush(stdout);

            /* TIME CONTROL: stop iterating if we have used our budget.
               We check AFTER a depth completes, never mid-search, so
               the move we return is always from a fully searched depth.
               The heuristic: if this depth took more than half the
               budget, the next depth will almost certainly exceed it,
               so we stop now rather than risk overshoot. */
            if (time_budget_ms > 0 && ms >= time_budget_ms / 2) break;
    }

    /* ROOT MATE / STALEMATE DETECTION
       ---------------------------------
       If legal_root == 0, we have no legal moves at the root.
       We are either checkmated or stalemated. We must not return
       "bestmove 0000" silently -- report the correct terminal score
       and let the GUI handle the game-over condition.            */
    if (!legal_root) {
        int is_in_check = in_check(side);
        printf("info depth 0 score mate %d nodes 0 time 0 pv\n",
               is_in_check ? 0 : 0);   /* 0 = already mated / stalemated */
        fflush(stdout);
    }

    printf("bestmove ");
    if (legal_root && global_best) print_move(global_best);
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
        if (!is_illegal()) n += perft(depth-1);
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

static int parse_move(const char *s, Move *out) {
    int f,t,pr=0; Move list[256]; int cnt,i;
    if (!s||strlen(s)<4) return 0;
    f=(s[0]-'a')+(s[1]-'1')*16;
    t=(s[2]-'a')+(s[3]-'1')*16;
    if (strlen(s)>4) pr = char_to_piece((char)tolower(s[4]));
    cnt=generate_moves(list,0);
    for (i=0;i<cnt;i++)
        if (move_from(list[i])==f&&move_to(list[i])==t&&move_promo(list[i])==pr) { *out=list[i]; return 1; }
    return 0;
}

/* Helpers for uci_loop */
static const char STARTPOS[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/* getval: find `key` in `line`, skip one space, then scan into `out` with `fmt`.
   Uses strlen so it works on any runtime string, not just string literals.
   e.g. getval(line, "depth", "%d", &depth) on "go depth 6" -> depth=6. */
static void getval(const char *line, const char *key, const char *fmt, void *out) {
    const char *t = strstr(line, key);
    if (t) sscanf(t + strlen(key) + 1, fmt, out);
}

void uci_loop(void) {
    char line[65536], *p;
    Move m;

    init_zobrist();
    init_lmr();
    parse_fen(STARTPOS);
    hash_key=generate_hash();

    tt = calloc((size_t)tt_size, sizeof(TTEntry));
    if (!tt) { fprintf(stderr, "Error: failed to allocate TT (%lld entries)\n",
                       (long long)tt_size); exit(1); }

    while (fgets(line,sizeof(line),stdin)) {
        if (!strncmp(line,"uci",3)) {
            printf("id name Chal\nid author Naman Thanki\n");
            printf("option name Hash type spin default 16 min 1 max 4096\n");
            printf("uciok\n");
            fflush(stdout);
        }
        else if (!strncmp(line,"setoption",9)) {
            /* setoption name Hash value <N>  (N in megabytes) */
            char *nptr = strstr(line, "name Hash value ");
            if (nptr) {
                int mb = 0;
                sscanf(nptr + 16, "%d", &mb);
                if (mb < 1) mb = 1;
                int64_t new_size = ((int64_t)mb * 1024 * 1024) / (int64_t)sizeof(TTEntry);
                if (new_size < 1) new_size = 1;
                TTEntry *new_tt = calloc((size_t)new_size, sizeof(TTEntry));
                if (new_tt) {
                    free(tt);
                    tt = new_tt;
                    tt_size = new_size;
                }
            }
        }
        else if (!strncmp(line,"isready",7)) {
            printf("readyok\n"); fflush(stdout);
        }
        else if (!strncmp(line,"ucinewgame",10)) {
            memset(tt, 0, (size_t)tt_size * sizeof(TTEntry)); memset(hist, 0, sizeof(hist));
            parse_fen(STARTPOS);
            hash_key = generate_hash();
        }
        else if (!strncmp(line,"perft",5)) {
            int depth=4; int64_t n;
            sscanf(line,"perft %d",&depth);
            n=perft(depth);
            printf("perft depth %d nodes %" PRId64 "\n",depth,n);
            fflush(stdout);
        }
        else if (!strncmp(line,"position",8)) {
            if (strlen(line) <= 9) continue;
            p=line+9;
            if (!strncmp(p,"startpos",8)) {
                parse_fen(STARTPOS);
                p+=8;
            } else if (!strncmp(p,"fen",3)) {
                p+=4; parse_fen(p);
            }
            hash_key=generate_hash();
            p=strstr(line,"moves");
            if (p) {
                p+=6;
                while (*p) {
                    char mv[6];
                    int n = 0;
                    while (*p==' ') p++;
                    if (*p=='\n'||!*p) break;
                    sscanf(p,"%5s%n",mv,&n);
                    if (n <= 0) break;
                    if (parse_move(mv,&m)) make_move(m);
                    p+=n;
                }
            }
        }
        else if (!strncmp(line,"go",2)) {
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

               If movestogo is not given we assume 30 moves remain --
               a safe estimate for sudden-death and increment games.
               search_root() iterates deeper until it has consumed more
               than half the budget for a single depth (at which point
               the next depth would almost certainly bust the limit), then
               returns the best move from the last fully searched depth. */

            int  depth   = MAX_PLY;
            int64_t wtime=0, btime=0, movestogo=30, winc=0, binc=0;

            getval(line, "depth",     "%d",          &depth);
            getval(line, "wtime",     "%" SCNd64,    &wtime);
            getval(line, "btime",     "%" SCNd64,    &btime);
            getval(line, "movestogo", "%" SCNd64,    &movestogo);
            getval(line, "winc",      "%" SCNd64,    &winc);
            getval(line, "binc",      "%" SCNd64,    &binc);

            if (wtime || btime) {
                int64_t our_time = (side==WHITE) ? wtime  : btime;
                int64_t our_inc  = (side==WHITE) ? winc   : binc;
                if (movestogo <= 0) movestogo = 30;
                time_budget_ms = (our_time / movestogo) + (our_inc * 3 / 4);
                if (time_budget_ms > our_time - 50) time_budget_ms = our_time - 50;
                if (time_budget_ms < 5) time_budget_ms = 5;
                depth = MAX_PLY;
            } else {
                time_budget_ms = 0;
            }
            search_root(depth);
        }
        else if (!strncmp(line,"quit",4)) {
            break;
        }
    }
    free(tt);
}

/* ===============================================================
   ENTRY POINT
   =============================================================== */

int main(void) {
    setbuf(stdout,NULL);
    uci_loop();
    return 0;
}
