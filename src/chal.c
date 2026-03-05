/*
================================================================
                          C H A L
================================================================
   Gujarati for "move." A minimal chess engine in ANSI C90.

   Author : Naman Thanki
   Date   : 2026

   This file is meant to be read as a book, not just run.
   Every subsystem is a short lesson in engine design.

   Compile:  gcc chal.c -O2 -Wall -Wextra -pedantic -std=gnu90 -o chal
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

/* ===============================================================
   S1  CONSTANTS & TYPES
   ===============================================================

   PIECE ENCODING
   --------------
   One byte per piece.  Bit 3 = colour (0=White, 1=Black).
   Bits 2..0 = type (1=Pawn .. 6=King, 0=Empty).

       PIECE(c,t)  ->  (c<<3)|t
       TYPE(p)     ->  p & 7
       COLOR(p)    ->  p >> 3
*/

enum { EMPTY=0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum { WHITE=0, BLACK=1 };
#define SQ_NONE     (-1)
#define TYPE(p)     ((p) & 7)
#define COLOR(p)    ((p) >> 3)
#define PIECE(c,t)  (((c)<<3)|(t))
#define INF         50000
#define MATE        30000

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
*/

typedef int Move;
#define FROM(m)          ((m) & 0x7F)
#define TO(m)            (((m)>>7) & 0x7F)
#define PROMO(m)         (((m)>>14) & 0xF)
#define MAKE_MOVE(f,t,p) ((f)|((t)<<7)|((p)<<14))

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

#define TT_EXACT 0
#define TT_ALPHA 1
#define TT_BETA  2

typedef struct {
    unsigned int  key;        /* Zobrist hash of the position  */
    int            score;      /* evaluation in centipawns      */
    Move           best_move;  /* best move found at this node  */
    unsigned int   depth_flag; /* bits 7-2: depth; bits 1-0: flag          */
} TTEntry;

/* TT_SIZE must be a power of two (hash_key % TT_SIZE = fast bitwise AND).
   65536 entries (1 MB) caused ~500x overwrites per depth-9 search, meaning
   almost every probe past depth 6 was a miss. 1048576 entries (16 MB)
   brings that to ~31x -- the TT actually contributes at the depths that
   matter for a 40/5 time control.                                         */
#define TT_SIZE 1048576
TTEntry tt[TT_SIZE];

#define TT_DEPTH(e)  ((e)->depth_flag >> 2)
#define TT_FLAG(e)   ((e)->depth_flag & 3)
#define TT_PACK(d,f) (unsigned int)(((d)<<2)|(f))

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
    Move          move;
    int           piece_captured;
    int           ep_square_prev;
    int           castle_rights_prev;
    unsigned int hash_prev;
} State;

State history[1024];

#define MAX_PLY 64
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
   hist[from][to] accumulates depth^2 credit every time the quiet move
   from->to causes a beta cutoff. Indexed by both squares (not just the
   destination) so Nf3 and Bf3 never share a bucket. Reset at the start
   of each search_root call. Capped at 32000 to prevent overflow.        */
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

   Binary geometry dictates that any coordinate residing inside the
   phantom subspace will naturally have its 4th bit (0x08) or 8th bit
   (0x80) activated. 
   
   Off-board detection reduces to one bitwise AND: (sq & 0x88) != 0
*/

#define SQ_IS_OFF(sq) ((sq) & 0x88)

int board[128];
int side, xside;
int ep_square;
int castle_rights;    /* bits: 1=WO-O  2=WO-O-O  4=BO-O  8=BO-O-O */
int king_sq[2];
int ply;
unsigned int hash_key;

/* Search telemetry -- reported in UCI info lines */
long nodes_searched;

/* Time control -- set by the go command handler before calling search_root.
   time_budget_ms = milliseconds we are allowed to spend on this move.
   0 means no time limit: search_root respects only max_depth.
   search_root checks the clock after each completed depth iteration and
   stops early if the elapsed time exceeds the budget. */
long time_budget_ms;

/* root_ply: value of ply when search_root() was called.
   sply (search ply) = ply - root_ply.
   This separates the game half-move counter (ply, used by history[])
   from the search depth index (sply, used by PV table and killers).
   Without this, 'position startpos moves e2e4' sets ply=1, and the
   PV table would write to pv[1] but print_pv would read pv[0]. */
int root_ply;
#define sply (ply - root_ply)

/* ===============================================================
   S3  DIRECTION & CASTLING DATA
   ===============================================================

   Idea
   Piece movement follows fixed trajectories. Calculating movement 
   relative to coordinates at runtime is mathematically heavy. Likewise, 
   enforcing castling validation normally requires disparate logical branching.

   Implementation
   We encode piece trajectories into a static integer array. One rank 
   translation equates to +/- 16 (0x88 boundary shift), while one file 
   translation equates to +/- 1.

   For Castling, four hardcoded configurations map king and rook translation
   points natively. The iterative logic evaluates the validity index without 
   heavy block branching.
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
static const int castle_kf[]   = {  4,   4, 116, 116};
static const int castle_kt[]   = {  6,   2, 118, 114};
static const int castle_rf[]   = {  7,   0, 119, 112};
static const int castle_rt[]   = {  5,   3, 117, 115};
static const int castle_col[]  = {WHITE, WHITE, BLACK, BLACK};
/* Castle rights bit cleared when king moves: 0-1=White(~3), 2-3=Black(~12) */
static const int castle_kmask[]= {~3, ~3, ~12, ~12};

/* Rights stripped when any piece touches these corner squares */
static const int cr_sq[]   = {0,   7,   112,  119 };
static const int cr_mask[]  = {~2,  ~1,  ~8,   ~4  };

/* ===============================================================
   S4  ZOBRIST HASHING
   ===============================================================

   Idea
   Fast position comparison requires a mathematical fingerprint. By 
   assigning a random 32-bit integer to every possible piece-square 
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

unsigned int zobrist_piece[2][7][128];
unsigned int zobrist_side;
unsigned int zobrist_ep[128];
unsigned int zobrist_castle[16];

static unsigned int rand32(void) {
    return ((unsigned int)rand()<<16) | ((unsigned int)rand()&0xFFFF);
}

void init_zobrist(void) {
    int c,p,s;
    for (c=0;c<2;c++) for (p=0;p<7;p++) for (s=0;s<128;s++)
        zobrist_piece[c][p][s] = rand32();
    zobrist_side = rand32();
    for (s=0;s<128;s++) zobrist_ep[s]     = rand32();
    for (s=0;s<16; s++) zobrist_castle[s] = rand32();
}

unsigned int generate_hash(void) {
    unsigned int h = 0;
    int sq;
    for (sq=0;sq<128;sq++) {
        if (SQ_IS_OFF(sq)) { sq+=7; continue; }
        if (board[sq])
            h ^= zobrist_piece[COLOR(board[sq])][TYPE(board[sq])][sq];
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

int is_square_attacked(int sq, int ac) {
    int i, step, tgt, p, pt;
    int pd = (ac==WHITE) ? -16 : 16;   /* pawn-attack direction from sq */

    for (i=-1; i<=1; i+=2) {           /* pawn check: two diagonal squares */
        tgt = sq+pd+i;
        if (!SQ_IS_OFF(tgt) && board[tgt]
            && COLOR(board[tgt])==ac && TYPE(board[tgt])==PAWN) return 1;
    }
    for (i=piece_offsets[KNIGHT]; i<piece_limits[KNIGHT]; i++) {
        tgt = sq+step_dir[i];
        if (!SQ_IS_OFF(tgt) && board[tgt]
            && COLOR(board[tgt])==ac && TYPE(board[tgt])==KNIGHT) return 1;
    }
    for (i=piece_offsets[KING]; i<piece_limits[KING]; i++) {
        tgt = sq+step_dir[i];
        if (!SQ_IS_OFF(tgt) && board[tgt]
            && COLOR(board[tgt])==ac && TYPE(board[tgt])==KING) return 1;
    }
    for (i=piece_offsets[BISHOP]; i<piece_limits[BISHOP]; i++) {
        step=step_dir[i]; tgt=sq+step;
        while (!SQ_IS_OFF(tgt)) {
            p=board[tgt];
            if (p) { pt=TYPE(p); if (COLOR(p)==ac&&(pt==BISHOP||pt==QUEEN)) return 1; break; }
            tgt+=step;
        }
    }
    for (i=piece_offsets[ROOK]; i<piece_limits[ROOK]; i++) {
        step=step_dir[i]; tgt=sq+step;
        while (!SQ_IS_OFF(tgt)) {
            p=board[tgt];
            if (p) { pt=TYPE(p); if (COLOR(p)==ac&&(pt==ROOK||pt==QUEEN)) return 1; break; }
            tgt+=step;
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

static void add_move(Move *list, int *n, int f, int t, int pr) {
    list[(*n)++] = MAKE_MOVE(f,t,pr);
}

void make_move(Move m) {
    int f=FROM(m), t=TO(m), pr=PROMO(m);
    int p=board[f], pt=TYPE(p), cap=board[t], ci;

    history[ply].move              = m;
    history[ply].piece_captured    = cap;
    history[ply].ep_square_prev    = ep_square;
    history[ply].castle_rights_prev= castle_rights;
    history[ply].hash_prev         = hash_key;

    /* En-passant capture: pawn is one rank behind the target */
    if (pt==PAWN && t==ep_square) {
        int ep_pawn = t + (side==WHITE ? -16 : 16);
        history[ply].piece_captured = board[ep_pawn];
        board[ep_pawn] = EMPTY;
        hash_key ^= zobrist_piece[xside][PAWN][ep_pawn];
    }

    board[t]=p; board[f]=EMPTY;
    hash_key ^= zobrist_piece[side][pt][f];
    hash_key ^= zobrist_piece[side][pt][t];
    if (cap) hash_key ^= zobrist_piece[xside][TYPE(cap)][t];

    if (pr) {
        board[t] = PIECE(side,pr);
        hash_key ^= zobrist_piece[side][pt][t];   /* XOR out pawn   */
        hash_key ^= zobrist_piece[side][pr][t];   /* XOR in  promo  */
    }

    hash_key ^= zobrist_castle[castle_rights];

    /* Castling: table-driven rook teleport */
    if (pt==KING) {
        king_sq[side] = t;
        for (ci=0; ci<4; ci++) {
            if (f==castle_kf[ci] && t==castle_kt[ci]) {
                board[castle_rf[ci]] = EMPTY;
                board[castle_rt[ci]] = PIECE(castle_col[ci], ROOK);
                hash_key ^= zobrist_piece[castle_col[ci]][ROOK][castle_rf[ci]];
                hash_key ^= zobrist_piece[castle_col[ci]][ROOK][castle_rt[ci]];
                break;
            }
        }
        castle_rights &= castle_kmask[side==WHITE ? 0 : 2];
    }

    /* Strip castling rights when a corner square is touched */
    for (ci=0; ci<4; ci++)
        if (f==cr_sq[ci] || t==cr_sq[ci]) castle_rights &= cr_mask[ci];

    hash_key ^= zobrist_castle[castle_rights];

    if (ep_square!=SQ_NONE) hash_key ^= zobrist_ep[ep_square];
    ep_square = SQ_NONE;
    if (pt==PAWN && abs(t-f)==32) {
        ep_square = f + (side==WHITE ? 16 : -16);
        hash_key ^= zobrist_ep[ep_square];
    }

    hash_key ^= zobrist_side;
    side^=1; xside^=1; ply++;
}

void undo_move(void) {
    Move m; int f,t,pr,pt,ci;

    ply--; side^=1; xside^=1;
    m=history[ply].move;
    f=FROM(m); t=TO(m); pr=PROMO(m);

    board[f]=board[t];
    board[t]=history[ply].piece_captured;
    pt=TYPE(board[f]);

    if (pr) board[f]=PIECE(side,PAWN);

    if (pt==PAWN && t==history[ply].ep_square_prev) {
        board[t]=EMPTY;
        board[t+(side==WHITE?-16:16)] = history[ply].piece_captured;
    }

    if (pt==KING) {
        king_sq[side]=f;
        for (ci=0; ci<4; ci++) {
            if (f==castle_kf[ci] && t==castle_kt[ci]) {
                board[castle_rt[ci]] = EMPTY;
                board[castle_rf[ci]] = PIECE(castle_col[ci], ROOK);
                break;
            }
        }
    }

    ep_square     = history[ply].ep_square_prev;
    castle_rights = history[ply].castle_rights_prev;
    hash_key      = history[ply].hash_prev;   /* O(1) restore */
}

/* ===============================================================
   S7  MOVE GENERATION
   ===============================================================

   Idea
   A high-performance chess engine defers complex legality checks. 
   Instead of mathematically proving every generated candidate is legal 
   upfront, we generate "pseudo-legal" geometric moves natively. 
   Legality (verifying the king isn't in check) is resolved later 
   during the search recursion using `is_square_attacked`.

   Implementation
   The move generator is unified. Passing `caps_only=1` gates the
   `add_move` output to filter exclusively for captures and promotions, 
   powering Quiescence Search natively without redundant sub-routines.

   1. Pawns: Evaluated directionally based on color (+16 or -16).
   2. Sliders/Leapers: Iterated iteratively via `step_dir` ray-traces.
   3. Castling: Validated by rights, emptiness, and attack matrices.
*/

int generate_moves(Move *moves, int caps_only) {
    int cnt=0, sq, tgt, step, p, pt, i;
    int d_pawn     = (side==WHITE) ?  16 : -16;
    int pawn_start = (side==WHITE) ?   1 :  6;
    int pawn_promo = (side==WHITE) ?   6 :  1;

    for (sq=0; sq<128; sq++) {
        if (sq & 0x88) { sq+=7; continue; }
        p=board[sq];
        if (!p || COLOR(p)!=side) continue;
        pt=TYPE(p);

        /* -- Pawns ------------------------------------------------ */
        if (pt==PAWN) {
            tgt=sq+d_pawn;
            if (!SQ_IS_OFF(tgt) && !board[tgt]) {
                if ((sq>>4)==pawn_promo) {
                    add_move(moves,&cnt,sq,tgt,QUEEN);
                    add_move(moves,&cnt,sq,tgt,ROOK);
                    add_move(moves,&cnt,sq,tgt,BISHOP);
                    add_move(moves,&cnt,sq,tgt,KNIGHT);
                } else if (!caps_only) {
                    add_move(moves,&cnt,sq,tgt,0);
                    if ((sq>>4)==pawn_start && !board[tgt+d_pawn])
                        add_move(moves,&cnt,sq,tgt+d_pawn,0);
                }
            }
            for (i=-1; i<=1; i+=2) {           /* diagonal captures + ep */
                tgt=sq+d_pawn+i;
                if (!SQ_IS_OFF(tgt)
                    && ((board[tgt] && COLOR(board[tgt])==xside) || tgt==ep_square)) {
                    if ((sq>>4)==pawn_promo) {
                        add_move(moves,&cnt,sq,tgt,QUEEN);
                        add_move(moves,&cnt,sq,tgt,ROOK);
                        add_move(moves,&cnt,sq,tgt,BISHOP);
                        add_move(moves,&cnt,sq,tgt,KNIGHT);
                    } else {
                        add_move(moves,&cnt,sq,tgt,0);
                    }
                }
            }
            continue;
        }

        /* -- Sliders & Leapers ------------------------------------ */
        for (i=piece_offsets[pt]; i<piece_limits[pt]; i++) {
            step=step_dir[i]; tgt=sq+step;
            while (!SQ_IS_OFF(tgt)) {
                if (!board[tgt]) {
                    if (!caps_only) add_move(moves,&cnt,sq,tgt,0);
                } else {
                    if (COLOR(board[tgt])==xside) add_move(moves,&cnt,sq,tgt,0);
                    break;
                }
                if (pt==KNIGHT || pt==KING) break;
                tgt+=step;
            }
        }

        /* -- Castling (king only, never in caps_only mode) -------- */
        if (pt==KING && !caps_only) {
            int ci, kf, kt, rf, bit, ac, sq1, sq2, sq3, clear_ok;
            for (ci=0; ci<4; ci++) {
                kf=castle_kf[ci]; kt=castle_kt[ci]; rf=castle_rf[ci];
                bit = (ci==0)?1:(ci==1)?2:(ci==2)?4:8;
                ac  = (castle_col[ci]==WHITE) ? BLACK : WHITE;
                
                if (sq != kf || castle_col[ci]!=side) continue;
                if (!(castle_rights & bit)) continue;
                if (board[rf] != PIECE(side,ROOK)) continue;

                /* Every square between king and rook must be empty */
                clear_ok = 1;
                sq1=(kf<rf)? kf+1 : rf+1;
                sq2=(kf<rf)? rf   : kf;
                for (sq3=sq1; sq3<sq2; sq3++)
                    if (board[sq3]) { clear_ok=0; break; }
                if (!clear_ok) continue;

                /* King's path must not traverse attacked squares */
                step = (kt>kf) ? 1 : -1;
                clear_ok = 1;
                for (sq3=kf; sq3!=(kt+step); sq3+=step)
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

void parse_fen(const char *fen) {
    int rank=7, file=0, sq, color, piece, i;
    char lo;

    for (i=0;i<128;i++) board[i]=EMPTY;
    castle_rights=0; ep_square=SQ_NONE; ply=0;
    memset(killers,0,sizeof(killers));
    memset(pv,0,sizeof(pv));
    memset(pv_length,0,sizeof(pv_length));
    memset(hist,0,sizeof(hist));

    while (*fen && *fen!=' ') {
        if (*fen=='/') { file=0; rank--; }
        else if (isdigit(*fen)) { file += *fen-'0'; }
        else {
            sq=rank*16+file;
            color=isupper(*fen)?WHITE:BLACK;
            lo=(char)tolower(*fen);
            piece=(lo=='p')?PAWN:(lo=='n')?KNIGHT:(lo=='b')?BISHOP:
                  (lo=='r')?ROOK:(lo=='q')?QUEEN:KING;
            board[sq]=PIECE(color,piece);
            if (piece==KING) king_sq[color]=sq;
            file++;
        }
        fen++;
    }
    fen++;

    side=(*fen=='w')?WHITE:BLACK; xside=side^1;
    fen+=2;

    while (*fen && *fen!=' ') {
        if (*fen=='K') castle_rights|=1;
        if (*fen=='Q') castle_rights|=2;
        if (*fen=='k') castle_rights|=4;
        if (*fen=='q') castle_rights|=8;
        fen++;
    }
    fen++;

    if (*fen!='-') ep_square=(fen[1]-'1')*16+(fen[0]-'a');
}

/* ===============================================================
   S9  EVALUATION
   ===============================================================

   Idea
   A static evaluator mathematically scores the current position from 
   the perspective of the side to move. Calculations are measured in
   centipawns (100 cp = 1 pawn). Since raw material summation is blind 
   to geometry, we supplement point totals with strategic coefficients.

   Implementation
   1. Material Values: Base scores dynamically added per piece.
   2. Piece-Square Tables (PSTs): Static arrays mapped to the 0x88 
      grid providing scalar bonuses for optimal piece placement 
      (e.g., centralizing knights or pushing central pawns).
   3. Phase-Aware King Evaluation: Interpolates the king's safety 
      score smoothly between a safe castled middlegame position 
      and an aggressively active endgame posture.
   4. Positional Rules: Deducts penalties for doubled or isolated pawns,
      and strictly rewards passed pawns proportionally to their rank.
*/

/*
   Piece-square tables (signed char = 1 byte each, range -128..127).
   All values are white-perspective (rank 0 = rank 1, rank 7 = rank 8).
   For Black: pst_rank = 7 - rank  (vertical mirror).
*/
static const signed char pst_pawn[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
};
static const signed char pst_knight[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};
static const signed char pst_bishop[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};
static const signed char pst_rook[64] = {
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
static const signed char pst_queen[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};
/* King middlegame: castled king in the corner is safest */
static const signed char pst_king_mg[64] = {
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30
};
/* King endgame: centralize to support passed pawns */
static const signed char pst_king_eg[64] = {
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50
};

/* Piece value table for material counting */
static const int piece_val[7] = {0,100,320,330,500,900,20000};
/* Non-pawn material values used to compute game phase */
static const int phase_val[7] = {0,0,  320,330,500,900,0    };
/* Total non-pawn material at game start (both sides): */
/* 2*(2*320 + 2*330 + 2*500 + 900) = 6400              */
#define MAX_PHASE 6400

int evaluate(void) {
    int sq, score=0, p, pt, color;
    int phase=0;
    int bishops[2]={0,0};
    int pawn_cnt[2][8];   /* pawn_cnt[color][file] */
    int f, mg, eg;

    memset(pawn_cnt,0,sizeof(pawn_cnt));

    for (sq=0; sq<128; sq++) {
        if (sq & 0x88) { sq+=7; continue; }
        p=board[sq]; if (!p) continue;
        pt=TYPE(p); color=COLOR(p);

        /* Accumulate game phase from non-pawn material */
        phase += phase_val[pt];

        /* Piece-square lookup: pst_rank mirrors for Black */
        {
            int rank = sq>>4;
            int file_idx = sq & 7;
            int pr = (color==WHITE) ? rank : (7-rank);
            int idx = pr*8 + file_idx;
            int positional;

            if (pt==KING) {
                /* Blend MG and EG king tables by phase.
                   phase==MAX_PHASE -> pure MG; phase==0 -> pure EG. */
                int p_clamped = (phase>MAX_PHASE) ? MAX_PHASE : phase;
                mg = (int)pst_king_mg[idx];
                eg = (int)pst_king_eg[idx];
                positional = (mg*p_clamped + eg*(MAX_PHASE-p_clamped)) / MAX_PHASE;
            } else {
                static const signed char *tables[7] = {
                    NULL, pst_pawn, pst_knight, pst_bishop,
                    pst_rook, pst_queen, NULL
                };
                positional = tables[pt] ? (int)tables[pt][idx] : 0;
            }

            if (color==WHITE) score += piece_val[pt] + positional;
            else              score -= piece_val[pt] + positional;
        }

        /* Mobility Bonus (Pseudo-Legal Raytrace) */
        if (pt >= KNIGHT && pt <= QUEEN) {
            int i, step, target, mob = 0;
            int start = piece_offsets[pt];
            int end   = piece_limits[pt];
            for (i = start; i < end; i++) {
                step = step_dir[i];
                target = sq + step;
                while (!SQ_IS_OFF(target)) {
                    if (board[target] == 0) { mob++; }
                    else { if (COLOR(board[target]) != color) mob++; break; }
                    if (pt == KNIGHT || pt == KING) break;
                    target += step;
                }
            }
            if (color == WHITE) score += (pt==QUEEN ? 1 : 2) * mob;
            else                score -= (pt==QUEEN ? 1 : 2) * mob;
        }

        if (pt==BISHOP) bishops[color]++;
        if (pt==PAWN)   pawn_cnt[color][sq & 7]++;
    }

    /* Bishop pair */
    if (bishops[WHITE]>=2) score+=30;
    if (bishops[BLACK]>=2) score-=30;

    /* Pawn structure penalties & Rook Activity & King Safety */
    for (color=0; color<2; color++) {
        int sign = (color==WHITE) ? 1 : -1;
        for (f=0; f<8; f++) {
            int cnt = pawn_cnt[color][f];
            if (cnt > 1) score -= sign * (cnt-1) * 20; /* Doubled */
            
            if (cnt) {
                /* Isolated */
                int left  = (f>0) ? pawn_cnt[color][f-1] : 0;
                int right = (f<7) ? pawn_cnt[color][f+1] : 0;
                if (!left && !right) score -= sign * 10;
            }
        }

        /* King Safety & Pawn Shields (Evaluated mostly in Middlegame) */
        {
            int ksq = king_sq[color];
            int kf = ksq & 7;
            /* Only check shields if king is castled or tucked away */
            if (kf <= 2 || kf >= 5) {
                int def, f_test, p_clamped;
                int penalty = 0;
                for (f_test = kf - 1; f_test <= kf + 1; f_test++) {
                    if (f_test >= 0 && f_test <= 7) {
                        def = pawn_cnt[color][f_test];
                        if (def == 0) {
                            /* Missing shield pawn */
                            penalty += 15;
                            /* If the file is completely open (enemy has no pawns either), it is highly dangerous */
                            if (pawn_cnt[color^1][f_test] == 0) penalty += 25;
                            else penalty += 10; /* Semi-open file targeting the king */
                        }
                    }
                }
                /* Scale penalty down as phase moves toward Endgame */
                p_clamped = (phase > MAX_PHASE) ? MAX_PHASE : phase;
                penalty = (penalty * p_clamped) / MAX_PHASE;
                score -= sign * penalty;
            }
        }
    }

    /* Rook Activity */
    for (sq=0; sq<128; sq++) {
        if (sq & 0x88) { sq+=7; continue; }
        p = board[sq];
        if (p && TYPE(p) == ROOK) {
            int sign, rank;
            color = COLOR(p);
            f = sq & 7;
            sign = (color==WHITE) ? 1 : -1;
            rank = sq >> 4;

            /* Semi-open file (+10) or fully open file (+20) */
            if (pawn_cnt[color][f] == 0) {
                if (pawn_cnt[color^1][f] == 0) score += sign * 20;
                else score += sign * 10;
            }
            /* 7th rank bonus (+20) */
            if ((color==WHITE && rank==6) || (color==BLACK && rank==1)) {
                score += sign * 20;
            }
        }
    }

    /* Passed Pawns: no enemy pawns ahead on current or adjacent files */
    for (sq=0; sq<128; sq++) {
        if (sq & 0x88) { sq+=7; continue; }
        p=board[sq];
        if (p && TYPE(p)==PAWN) {
            color = COLOR(p);
            f = sq & 7;
            {
                int rank = sq >> 4;
                int step = (color==WHITE) ? 16 : -16;
                int pass = 1;
                int r, f_adj;
                for (r=rank+(step>>4); r>=0 && r<=7; r+=(step>>4)) {
                    for (f_adj=f-1; f_adj<=f+1; f_adj++) {
                        if (f_adj>=0 && f_adj<=7) {
                            if (board[(r<<4)+f_adj] == PIECE(color^1, PAWN)) { pass=0; break; }
                        }
                    }
                    if (!pass) break;
                }
                if (pass) {
                    /* Bonus accelerates as rank advances: 1->2, 2->8, 3->18, 4->32, 5->50, 6->72 */
                    int r_rel = (color==WHITE) ? rank : 7-rank;
                    int bonus = (r_rel * r_rel) * 2;
                    if (color==WHITE) score += bonus;
                    else              score -= bonus;
                }
            }
        }
    }

    return (side==WHITE) ? score : -score;
}

/* ===============================================================
   S10  MOVE ORDERING
   ===============================================================

   Idea
   Alpha-Beta pruning performs optimally when the best move is examined
   first. Perfect ordering theoretically reduces the search space from 
   O(b^d) to O(b^(d/2)), doubling search depth effectively at zero cost.

   Implementation
   We score every generated move before searching. A simple Selection
   Sort strictly orders the moves sequentially by these priorities:
   1. Hash Move (20000): The globally proven best move from the TT.
   2. MVV-LVA (1000+): Prioritizes capturing valuable pieces using cheap 
      attackers (e.g., Pawn takes Queen).
   3. Killers (800): High-performing quiet moves discovered earlier 
      at the same depth.
   4. History Heuristic (1..799): Historical reputation of quiet moves.
*/

static const int order_val[7] = {0,100,300,300,500,900,0};

static int score_move(Move m, Move hash_move, int depth) {
    int cap, sc=0;
    if (m==hash_move) return 20000;
    cap=board[TO(m)];
    if (cap)           sc=1000+order_val[TYPE(cap)]-order_val[TYPE(board[FROM(m)])];
    else if (PROMO(m)) sc=900;
    else if (depth<MAX_PLY && (m==killers[depth][0]||m==killers[depth][1])) sc=800;
    else               { int h=hist[FROM(m)][TO(m)]; sc=(h>799)?799:h; }
    return sc;
}

static void sort_moves(Move *moves, int n, Move hash_move, int depth) {
    int scores[256], i, j, ts; Move tm;
    for (i=0;i<n;i++) scores[i]=score_move(moves[i],hash_move,depth);
    for (i=0;i<n-1;i++)
        for (j=i+1;j<n;j++)
            if (scores[j]>scores[i]) {
                ts=scores[i]; tm=moves[i];
                scores[i]=scores[j]; moves[i]=moves[j];
                scores[j]=ts; moves[j]=tm;
            }
}

/* ===============================================================
   S11  SEARCH
   ===============================================================

   Idea
   The engine explores future game states sequentially to determine the 
   mathematically strongest move. Exploring every possible interaction 
   is computationally explosive, requiring aggressive pruning algorithms.

   Implementation (Negamax Alpha-Beta)
   The engine implements an Alpha-Beta framework using the Negamax 
   formulation: "My best score is my opponent's worst score." 
   We evaluate the tree iteratively, leveraging multiple heuristics:

   1. Quiescence Search (QS): Resolves unstable capture sequences 
      at the horizon to prevent tactical blindness.
   2. Null Move Pruning (NMP): Concedes a turn to prove a position 
      is so overwhelmingly strong that further search is mathematically unnecessary.
   3. Late Move Reductions (LMR): Shallowly checks unpromising, 
      low-priority quiet moves to skip wasted deep exploration.
   4. Transposition Table (TT): Validates previous sub-tree bounds 
      to eliminate redundant recursion globally.
   5. Aspiration Windows: Constrains the iterative root bounds, 
      improving Alpha-Beta cutoffs across the highest-value branches.
   6. Repetition Detection: Prevents evaluating a drawn cycle as 
      advantageous, safeguarding against perpetual-check recursion.
*/

int search(int depth, int alpha, int beta) {
    Move moves[256], best=0, hash_move=0;
    int cnt, i, legal=0, best_sc, old_alpha=alpha, sc, caps_only;
    TTEntry *e = &tt[hash_key % TT_SIZE];

    /* HARD TIME LIMIT CHECK
       Every 1024 nodes, check if we have exceeded our absolute time budget.
       If we have, abort the search tree immediately to prevent flagging. */
    if ((nodes_searched & 1023) == 0 && time_budget_ms > 0) {
        long ms = (long)((clock() - t_start) * 1000 / CLOCKS_PER_SEC);
        if (ms >= time_budget_ms) {
            time_over_flag = 1;
            return 0;
        }
    }
    if (time_over_flag) return 0;

    /* REPETITION DETECTION (Draw Safety)
       If this exact board hash has occurred previously in the current
       game sequence mathematically, it is a 3-fold repetition draw.
       Returning 0 forces the engine to abandon +3.00 evaluations that
       are locked behind perpetual checks, forcing it to push pawns safely.
       We step by 2 because repetitions strictly require same side-to-move. */
    if (ply > 0) {
        for (i = ply - 2; i >= 0; i -= 2) {
            if (history[i].hash_prev == hash_key) return 0; /* Draw */
        }
    }

    /* TT probe: always extract hash_move for ordering */
    if (e->key==hash_key) {
        hash_move=e->best_move;
        if ((int)TT_DEPTH(e) >= depth) {
            int flag=TT_FLAG(e);
            if (flag==TT_EXACT)                  return e->score;
            if (flag==TT_BETA  && e->score>=beta) return beta;
            if (flag==TT_ALPHA && e->score<=alpha) return alpha;
        }
    }

    /* Quiescence: stand-pat evaluation when out of depth */
    caps_only = (depth<=0);
    if (caps_only) {
        best_sc = evaluate();
        if (best_sc>=beta) return beta;
        if (best_sc>alpha) alpha=best_sc;
        if (depth < -6) return alpha;   /* max quiescence depth cap */
    } else {
        best_sc = -INF;
    }

    nodes_searched++;
    pv_length[sply] = sply;   /* default: PV ends here */

   /* NULL MOVE PRUNING (NMP)
       Skip our turn; if the opponent still cannot beat beta at reduced
       depth, prune immediately. R=2 normally, R=3 at depth >= 6.

       ZUGZWANG GUARD: pure pawn endgames can be genuine zugzwang where
       passing really is the worst move. We skip NMP when the side to
       move has no non-pawn non-king piece, making the null move
       assumption safe in all normal middlegame and endgame positions.  */
    if (!caps_only && depth >= 3 && !is_square_attacked(king_sq[side], xside)) {
        int sq_scan, has_piece = 0, ep_sq_prev, R;
        for (sq_scan = 0; sq_scan < 128 && !has_piece; sq_scan++) {
            int pt_scan;
            if (sq_scan & 0x88) { sq_scan += 7; continue; }
            pt_scan = TYPE(board[sq_scan]);
            if (board[sq_scan] && COLOR(board[sq_scan]) == side
                && pt_scan != PAWN && pt_scan != KING) has_piece = 1;
        }
        if (has_piece) {
            R = (depth >= 6) ? 3 : 2;
            ep_sq_prev = ep_square;
            hash_key ^= zobrist_side;
            if (ep_square != SQ_NONE) hash_key ^= zobrist_ep[ep_square];
            ep_square = SQ_NONE;
            side ^= 1; xside ^= 1; ply++;
            sc = -search(depth - R - 1, -beta, -beta + 1);
            side ^= 1; xside ^= 1; ply--;
            ep_square = ep_sq_prev;
            if (ep_square != SQ_NONE) hash_key ^= zobrist_ep[ep_square];
            hash_key ^= zobrist_side;
            if (sc >= beta) return beta;
        }
    }

    cnt=generate_moves(moves, caps_only);
    sort_moves(moves, cnt, hash_move, sply);

    for (i=0; i<cnt; i++) {
        /* DELTA PRUNING (Quiescence only)
           If capturing this piece plus a safety margin can't possibly
           raise alpha, skip generating the recursive tree. */
        if (caps_only && board[TO(moves[i])]) {
            int cap_val = piece_val[TYPE(board[TO(moves[i])])];
            if (best_sc + cap_val + 200 < alpha) continue;
        }

        make_move(moves[i]);
        if (is_square_attacked(king_sq[xside],side)) { undo_move(); continue; }
        legal++;

        /* LATE MOVE REDUCTIONS (LMR)
           Quiet moves sorted late are rarely best. Save time by
           searching them at shallower depth (depth-2). If it
           surprisingly scores well, research it cleanly. */
        if (!caps_only && depth >= 3 && legal >= 4 && !board[TO(moves[i])] && !PROMO(moves[i])) {
            int gives_check = is_square_attacked(king_sq[xside], side);
            if (!gives_check) {
                sc = -search(depth-2, -alpha-1, -alpha);
                if (sc > alpha && sc < beta) sc = -search(depth-1, -beta, -alpha);
            } else {
                sc = -search(depth-1, -beta, -alpha);
            }
        } else {
            sc = -search(depth-1, -beta, -alpha);
        }

        undo_move();

        if (sc>best_sc) { best_sc=sc; best=moves[i]; }
        if (sc>alpha) {
            alpha=sc;
            /* Update PV: this move improves our best line */
            pv[sply][sply]=moves[i];
            {
                int k;
                for (k=sply+1; k<pv_length[sply+1]; k++)
                    pv[sply][k]=pv[sply+1][k];
                pv_length[sply]=pv_length[sply+1];
            }
        }
       if (alpha>=beta) {
            if (!board[TO(moves[i])]) {   /* quiet cutoff move */
                int d = (sply < MAX_PLY) ? sply : MAX_PLY - 1;
                int bonus = depth * depth;
                int h;
                killers[d][1] = killers[d][0];
                killers[d][0] = moves[i];
                /* History: credit the (from,to) pair, not just the destination.
                   This keeps Nf3 and Bf3 separate. Cap at 32000 (never wraps). */
                h = hist[FROM(moves[i])][TO(moves[i])] + bonus;
                hist[FROM(moves[i])][TO(moves[i])] = (h > 32000) ? 32000 : h;
            }
            break;
        }
    }

    /* Checkmate or stalemate (only detectable in full search, not QS) */
    if (!caps_only && !legal)
        return is_square_attacked(king_sq[side],xside) ? -(MATE-sply) : 0;

    /* TT store with depth-preferred replacement */
    if (best && (e->key!=hash_key || depth>=(int)TT_DEPTH(e))) {
        int flag = (best_sc<=old_alpha)?TT_ALPHA:(best_sc>=beta)?TT_BETA:TT_EXACT;
        e->key       = hash_key;
        e->score     = best_sc;
        e->best_move = best;
        e->depth_flag= TT_PACK(depth>0?depth:0, flag);
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
    int f=FROM(m), t=TO(m), pr=PROMO(m);
    char pc=0;
    if (pr==QUEEN) pc='q'; else if (pr==ROOK) pc='r';
    else if (pr==BISHOP) pc='b'; else if (pr==KNIGHT) pc='n';
    if (pc) printf("%c%c%c%c%c",'a'+(f&7),'1'+(f>>4),'a'+(t&7),'1'+(t>>4),pc);
    else    printf("%c%c%c%c",  'a'+(f&7),'1'+(f>>4),'a'+(t&7),'1'+(t>>4));
}

static void print_pv(void) {
    int k;
    for (k=0; k<pv_length[0]; k++) {
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
    int cnt=generate_moves(moves,0), d, i, sc, best_sc=-INF, legal_root=0;

    time_over_flag = 0;
    t_start = clock();
    memset(hist, 0, sizeof(hist));
    memset(killers, 0, sizeof(killers));

    sort_moves(moves, cnt, 0, 0);
    root_ply = ply;   /* anchor sply=0 at the search root */

    for (d=1; d<=max_depth; d++) {
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
                for (i=0;i<cnt;i++)
                    if (moves[i]==global_best) {
                        Move tmp=moves[0]; moves[0]=moves[i]; moves[i]=tmp; break;
                    }
            }

            for (i=0; i<cnt; i++) {
                make_move(moves[i]);
                if (is_square_attacked(king_sq[xside],side)) { undo_move(); continue; }
                if (d==1) {
                    legal_root++;
                    if (!global_best) global_best = moves[i]; /* Immediate fallback */
                }
                sc = -search(d-1, -asp_hi, -asp_lo);
                undo_move();
                if (time_over_flag) break; /* Abort this depth entirely */
                if (sc>best_sc) {
                    best_sc=sc; iter_best=moves[i];
                    /* Propagate PV from depth 1 up to the root (sply=0) */
                    pv[0][0]=moves[i];
                    {
                        int k;
                        for (k=1; k<pv_length[1]; k++) pv[0][k]=pv[1][k];
                        pv_length[0]=pv_length[1];
                    }
                }
                if (sc > asp_lo) asp_lo = sc;
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
        {
            long ms=(long)((clock()-t_start)*1000/CLOCKS_PER_SEC);
            if (best_sc > MATE - MAX_PLY)
                printf("info depth %d score mate %d nodes %ld time %ld pv",
                       d, (MATE - best_sc + 1)/2, nodes_searched, ms);
            else if (best_sc < -(MATE - MAX_PLY))
                printf("info depth %d score mate %d nodes %ld time %ld pv",
                       d, -(MATE + best_sc + 1)/2, nodes_searched, ms);
            else
                printf("info depth %d score cp %d nodes %ld time %ld pv",
                       d, best_sc, nodes_searched, ms);
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
    }

    /* ROOT MATE / STALEMATE DETECTION
       ---------------------------------
       If legal_root == 0, we have no legal moves at the root.
       We are either checkmated or stalemated. We must not return
       "bestmove 0000" silently -- report the correct terminal score
       and let the GUI handle the game-over condition.            */
    if (!legal_root) {
        int in_check = is_square_attacked(king_sq[side], xside);
        printf("info depth 0 score mate %d nodes 0 time 0 pv\n",
               in_check ? 0 : 0);   /* 0 = already mated / stalemated */
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
   Performance Test (Perft) explores the move tree to a distinct depth
   strictly to count terminal leaf nodes.

   Implementation
   By mathematically matching generated node hierarchies against 
   universally proven engine results, we verify that the Move Generation
   and Make/Undo logic is natively flawless without subjective evaluation.
*/

long perft(int depth) {
    Move moves[256]; int cnt,i; long n=0;
    if (!depth) return 1;
    cnt=generate_moves(moves,0);
    for (i=0;i<cnt;i++) {
        make_move(moves[i]);
        if (!is_square_attacked(king_sq[xside],side)) n+=perft(depth-1);
        undo_move();
    }
    return n;
}

/* ===============================================================
   S13  UCI LOOP
   ===============================================================

   Idea
   A standardized bridge between abstract engine computation and the 
   human-readable Graphical User Interface natively controls game execution.

   Implementation
   A strictly stateless polling loop interprets ASCII strings passed 
   via `stdin`. It dynamically controls positional initialization, 
   time-boundary constraints, and concurrent telemetry reporting.
*/

static int parse_move(const char *s, Move *out) {
    int f,t,pr=0; Move list[256]; int cnt,i;
    if (!s||strlen(s)<4) return 0;
    f=(s[0]-'a')+(s[1]-'1')*16;
    t=(s[2]-'a')+(s[3]-'1')*16;
    if (strlen(s)>4) {
        char c=(char)tolower(s[4]);
        if (c=='q') pr=QUEEN; else if (c=='r') pr=ROOK;
        else if (c=='b') pr=BISHOP; else if (c=='n') pr=KNIGHT;
    }
    cnt=generate_moves(list,0);
    for (i=0;i<cnt;i++)
        if (FROM(list[i])==f&&TO(list[i])==t&&PROMO(list[i])==pr) { *out=list[i]; return 1; }
    return 0;
}

void uci_loop(void) {
    char line[65536], *p;
    Move m;

    srand((unsigned)time(NULL));
    init_zobrist();
    parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    hash_key=generate_hash();

    while (fgets(line,sizeof(line),stdin)) {
        if (!strncmp(line,"uci",3)) {
            printf("id name Chal\nid author Naman Thanki\nuciok\n");
            fflush(stdout);
        }
        else if (!strncmp(line,"isready",7)) {
            printf("readyok\n"); fflush(stdout);
        }
        else if (!strncmp(line,"ucinewgame",10)) {
            memset(tt,   0, sizeof(tt));
            memset(hist, 0, sizeof(hist));
            parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            hash_key = generate_hash();
        }
        else if (!strncmp(line,"perft",5)) {
            int depth=4; long n;
            sscanf(line,"perft %d",&depth);
            n=perft(depth);
            printf("perft depth %d nodes %ld\n",depth,n);
            fflush(stdout);
        }
        else if (!strncmp(line,"position",8)) {
            p=line+9;
            if (!strncmp(p,"startpos",8)) {
                parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
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
                    while (*p==' ') p++;
                    if (*p=='\n'||!*p) break;
                    sscanf(p,"%5s",mv);
                    if (parse_move(mv,&m)) make_move(m);
                    p+=strlen(mv);
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
               A 5% safety margin prevents us from ever flagging on lag:

                   budget = budget * 95 / 100

               search_root() iterates deeper until it has consumed more
               than half the budget for a single depth (at which point
               the next depth would almost certainly bust the limit), then
               returns the best move from the last fully searched depth. */

            int  depth      = MAX_PLY;
            long wtime=0, btime=0, movestogo=30, winc=0, binc=0;
            char *tok;

            tok = strstr(line,"depth");
            if (tok) { sscanf(tok,"depth %d",&depth); }

            tok = strstr(line,"wtime");    if (tok) sscanf(tok,"wtime %ld",&wtime);
            tok = strstr(line,"btime");    if (tok) sscanf(tok,"btime %ld",&btime);
            tok = strstr(line,"movestogo");if (tok) sscanf(tok,"movestogo %ld",&movestogo);
            tok = strstr(line,"winc");     if (tok) sscanf(tok,"winc %ld",&winc);
            tok = strstr(line,"binc");     if (tok) sscanf(tok,"binc %ld",&binc);

            if (wtime || btime) {
                long our_time = (side==WHITE) ? wtime  : btime;
                long our_inc  = (side==WHITE) ? winc   : binc;
                if (movestogo <= 0) movestogo = 30;
                
                time_budget_ms = (our_time / movestogo) + (our_inc * 3 / 4);
                
                /* Never consume all remaining time; reserve 50ms for network/GUI lag */
                if (time_budget_ms > our_time - 50) {
                    time_budget_ms = our_time - 50;
                }
                
                if (time_budget_ms < 5) time_budget_ms = 5; /* Absolute minimum */
                depth = MAX_PLY;   /* let time control decide when to stop */
            } else {
                time_budget_ms = 0;  /* fixed-depth: no clock check */
            }
            search_root(depth);
        }
        else if (!strncmp(line,"quit",4)) {
            break;
        }
    }
}

/* ===============================================================
   ENTRY POINT
   =============================================================== */

int main(void) {
    setbuf(stdout,NULL);
    uci_loop();
    return 0;
}