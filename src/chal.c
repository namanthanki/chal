/*
================================================================
                          C H A L
================================================================
   "Chal" is Gujarati for "move" or "tactic".

   Author : Naman Thanki
   Date   : 2026

   This program is written to be read from top to bottom, like a
   short book on chess programming. It shows how bitboards, a
   small neural network (NNUE), and an alpha-beta search fit
   together in standard C without excess ceremony.

   Compile:  gcc -O3 -march=native -Wall -Wextra src/chal.c -o chal.exe
   Protocol: Universal Chess Interface (UCI)
================================================================

   TABLE OF CONTENTS
   -----------------
   S1  Constants & Types       Bitboards, square indices, and moves
   S2  Attacks & Magic Tables  Leapers, rays, and slider lookups
   S3  Zobrist Hashing & TT    Position fingerprints and hash table
   S4  NNUE Evaluation         Accumulator updates and forward pass
   S5  Board Representation    Bitboards, mailbox, and make/undo
   S6  Move Generation         Pseudo-legal move generator
   S7  Search & Heuristics     Alpha-beta, quiescence, and pruning
   S8  UCI & Benchmarks        Command loop, perft, and benchmark
================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>

#define CHAL_VERSION "2.0.0"

/* ================================================================
   S1  CONSTANTS & TYPES
   ================================================================

   A chessboard has 64 squares. A 64-bit integer has 64 bits. A
   "bitboard" represents any set of squares (for example, all white
   pawns) as a single uint64_t where bit N is 1 if square N is in
   the set.

   Squares are numbered 0 to 63 from a1 to h8 (rank * 8 + file):

     8 | 56 57 58 59 60 61 62 63
     7 | 48 49 50 51 52 53 54 55
     6 | 40 41 42 43 44 45 46 47
     5 | 32 33 34 35 36 37 38 39
     4 | 24 25 26 27 28 29 30 31
     3 | 16 17 18 19 20 21 22 23
     2 |  8  9 10 11 12 13 14 15
     1 |  0  1  2  3  4  5  6  7
       +------------------------
          a  b  c  d  e  f  g  h

   Basic bit operations:
     bit(sq)        1ULL << sq
     popcount(bb)   number of set bits
     lsb(bb)        index of the least significant set bit (0..63)
     pop_lsb(&bb)   return lowest set bit, then clear it (bb &= bb - 1)
*/

typedef uint64_t u64;
typedef uint16_t Move;

enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE, WHITE = 0, BLACK = 1, BOTH = 2 };
enum { A1,B1,C1,D1,E1,F1,G1,H1, A2,B2,C2,D2,E2,F2,G2,H2, A3,B3,C3,D3,E3,F3,G3,H3, A4,B4,C4,D4,E4,F4,G4,H4, A5,B5,C5,D5,E5,F5,G5,H5, A6,B6,C6,D6,E6,F6,G6,H6, A7,B7,C7,D7,E7,F7,G7,H7, A8,B8,C8,D8,E8,F8,G8,H8, SQ_NONE };
enum { CASTLE_WK = 1, CASTLE_WQ = 2, CASTLE_BK = 4, CASTLE_BQ = 8, FLAG_QUIET = 0, FLAG_DOUBLE_PUSH, FLAG_CASTLE, FLAG_EP, FLAG_PROMO_N, FLAG_PROMO_B, FLAG_PROMO_R, FLAG_PROMO_Q };

#define C64(x) (x##ULL)
#define bit(sq) (C64(1) << (sq))
#define popcount(bb) __builtin_popcountll(bb)
#define lsb(bb) __builtin_ctzll(bb)

static inline int pop_lsb(u64 *bb) { int sq = __builtin_ctzll(*bb); *bb &= *bb - 1; return sq; }

/*
   Move encoding (16 bits):
     bits  0.. 5   from-square (0..63)
     bits  6..11   to-square   (0..63)
     bits 12..15   special move flags

     15..12   11..6      5..0
     [ flag ] [ to-sq ] [ from-sq ]

   Flags: 0=quiet, 1=double pawn push, 2=castle, 3=en passant,
          4=promo knight, 5=promo bishop, 6=promo rook, 7=promo queen.
*/

static inline Move encode_move(int f, int t, int fl) { return (Move)((f & 0x3F) | ((t & 0x3F) << 6) | ((fl & 0xF) << 12)); }
static inline int move_from(Move m) { return m & 0x3F; } 
static inline int move_to(Move m) { return (m >> 6) & 0x3F; }
static inline int move_flag(Move m) { return (m >> 12) & 0xF; } 
static inline int promo_piece(int fl) { return (fl - FLAG_PROMO_N) + KNIGHT; }

/* A simple move list buffer. A chess position has at most 218 moves. */
typedef struct { Move moves[256]; int count; } MoveList;
static inline void push_move(MoveList *list, Move m) { list->moves[list->count++] = m; }
static inline void push_promos(MoveList *list, int f, int t) { for (int fl = FLAG_PROMO_Q; fl >= FLAG_PROMO_N; fl--) push_move(list, encode_move(f, t, fl)); }

/* ================================================================
   S2  ATTACKS & MAGIC TABLES
   ================================================================

   Pawns, knights, and kings jump by fixed offsets regardless of
   other pieces. Their attacks are precomputed into lookup tables:
     pawn_attacks[color][sq]
     knight_attacks[sq]
     king_attacks[sq]

   Bishops, rooks, and queens slide until they reach a board edge
   or hit another piece. Tracing rays square by square during search
   is slow. Instead, magic bitboards map any blocker arrangement to
   a precomputed attack set in a single lookup:

     index = ((occ & mask) * magic) >> shift;
     attacks = table[index];

   1. The mask contains only the inner squares of each ray; pieces on
      the board rim stop the ray but cannot block beyond it.
   2. Multiplying by a 64-bit constant ("magic") packs the relevant
      blocker bits into the high bits of the product.
   3. Shifting right leaves a compact, collision-free index.

   During setup, the Carry-Rippler trick:
     occ = (occ - mask) & mask;
   iterates through every blocker subset of the mask without branching.
*/

u64 pawn_attacks[2][64], knight_attacks[64], king_attacks[64], bishop_table[64 * 512], rook_table[64 * 4096];
typedef struct { u64 mask, magic; int shift; u64 *table; } Magic;
Magic bishop_magics[64], rook_magics[64];

const u64 BISHOP_MAGICS_CONST[64] = { C64(0x0842021001020884), C64(0x0014131444048100), C64(0x0108080120204050), C64(0x3209040500010100),C64(0x0004042000020858), C64(0x0001016050000440), C64(0x0002082108080008), C64(0x2822004208050821),C64(0x1485404801044290), C64(0x0c012108411900a0), C64(0x81010808912e0040), C64(0x0018040411840512),C64(0x0800040420050012), C64(0x2840020250140020), C64(0x4000210818021804), C64(0x0000010413040a30),C64(0x00500004a0080120), C64(0x0008000210840090), C64(0x20100001888a0040), C64(0x1301000804110088),C64(0x0004020880a00000), C64(0x129c802100a01106), C64(0x14020c8400840406), C64(0x0442000343040100),C64(0x201010006820e100), C64(0x2401282260880100), C64(0x0004020010002240), C64(0x0068080008820102),C64(0x4a25001101004010), C64(0x0003010002008c80), C64(0x0201004004044404), C64(0x701401022c208209),C64(0x005002c800105042), C64(0x0202304405020801), C64(0x0240203000880680), C64(0x00840a0080080080),C64(0x0020008402048020), C64(0x4010100480084048), C64(0x0008009080010801), C64(0x4002840140010100),C64(0x0024022050081560), C64(0x0809008804012082), C64(0x0000220130000a00), C64(0x0004001c24024800),C64(0x0e21031020800c00), C64(0x0240900400c14020), C64(0x0906021444008900), C64(0x20c80a4082053268),C64(0x880e441414400002), C64(0x0021018804020000), C64(0xa010002908080100), C64(0x0480802220880030),C64(0x20010010202a0020), C64(0x0800200a02020012), C64(0x4460080208414802), C64(0x0003120811010002),C64(0x0000440400880421), C64(0x4003902401241012), C64(0x1002900100413020), C64(0x1005804020a0a800),C64(0x11404a02b0020600), C64(0x006c004818302420), C64(0x0000508c10008200), C64(0x1a0520140101e100) };
const u64 ROOK_MAGICS_CONST[64] = { C64(0x0080001924400080), C64(0x4040004010002000), C64(0x2100200100400811), C64(0x8b0004a008100100),C64(0x0200041020481200), C64(0x0100020100840018), C64(0x8300010004008200), C64(0x0180084100002c80),C64(0x0010800084a04002), C64(0x0013004004250082), C64(0x0011001101a000c2), C64(0x9001002009005000),C64(0x1001800400080181), C64(0x000a00100a000864), C64(0x0105000405000200), C64(0x400e000049029204),C64(0x0000818001c00060), C64(0x0290084000a00450), C64(0x0100808010006001), C64(0x2050010011002108),C64(0x2000818004002800), C64(0x0605808004009200), C64(0x0000040018100596), C64(0x4002020002e40081),C64(0x0241400080082080), C64(0x0040008080200040), C64(0xa200200480100083), C64(0x5002000a00201040),C64(0x2482001600081021), C64(0x80090400804a0080), C64(0x204010040048010a), C64(0x2068228600004401),C64(0x080a400022800484), C64(0xa062400080802001), C64(0x00008012020041a1), C64(0x0080b00081800800),C64(0x0040080011000d00), C64(0x0020804400801200), C64(0x0011000401008200), C64(0x2400042092000041),C64(0x2000400480a08000), C64(0x0000402010004000), C64(0x1021100020008080), C64(0x006030002101000a),C64(0x01050010c8010004), C64(0x0002000508420010), C64(0x000900020005000c), C64(0x6200006081060004),C64(0x8601008022044200), C64(0x2000c00280250100), C64(0x0100200080100880), C64(0x0121a00810010100),C64(0x1080820401180080), C64(0x1000020004008080), C64(0x0880010608100400), C64(0x000000a043040200),C64(0x00010080001020c1), C64(0x01101040008701a1), C64(0x3204300820820142), C64(0x4068210008443001),C64(0x0805000204180011), C64(0x1401000802040001), C64(0x0808081002408914), C64(0x1098118401082042) };

static const int s_dr[2][4] = {{1, -1, 0, 0}, {1, 1, -1, -1}}, s_df[2][4] = {{0, 0, 1, -1}, {1, -1, 1, -1}};

static u64 slider_mask(int sq, int is_b) {
    u64 m = 0; int r = sq / 8, f = sq % 8;
    for (int i = 0; i < 4; i++)
        for (int cr = r + s_dr[is_b][i], cf = f + s_df[is_b][i]; cr + s_dr[is_b][i] >= 0 && cr + s_dr[is_b][i] < 8 && cf + s_df[is_b][i] >= 0 && cf + s_df[is_b][i] < 8; cr += s_dr[is_b][i], cf += s_df[is_b][i])
            m |= bit(cr * 8 + cf);
    return m;
}

static u64 slider_attacks(int sq, u64 occ, int is_b) {
    u64 a = 0; int r = sq / 8, f = sq % 8;
    for (int i = 0; i < 4; i++)
        for (int cr = r + s_dr[is_b][i], cf = f + s_df[is_b][i]; cr >= 0 && cr < 8 && cf >= 0 && cf < 8; cr += s_dr[is_b][i], cf += s_df[is_b][i]) {
            a |= bit(cr * 8 + cf);
            if (occ & bit(cr * 8 + cf)) break;
        }
    return a;
}

static void init_magics(void) {
    for (int is_b = 0; is_b < 2; is_b++)
        for (int sq = 0; sq < 64; sq++) {
            Magic *m = is_b ? &bishop_magics[sq] : &rook_magics[sq];
            m->mask = slider_mask(sq, is_b);
            m->magic = is_b ? BISHOP_MAGICS_CONST[sq] : ROOK_MAGICS_CONST[sq];
            m->shift = 64 - popcount(m->mask);
            m->table = is_b ? &bishop_table[sq * 512] : &rook_table[sq * 4096];
            u64 occ = 0;
            do {
                m->table[(unsigned int)((occ * m->magic) >> m->shift)] = slider_attacks(sq, occ, is_b);
                occ = (occ - m->mask) & m->mask;
            } while (occ);
        }
}

void init_attacks(void) {
    int dn[8][2] = {{2,1},{2,-1},{1,2},{1,-2},{-1,2},{-1,-2},{-2,1},{-2,-1}}, dk[8][2] = {{1,1},{1,0},{1,-1},{0,1},{0,-1},{-1,1},{-1,0},{-1,-1}};
    for (int sq = 0; sq < 64; sq++) {
        u64 b = bit(sq); int r = sq / 8, f = sq % 8;
        pawn_attacks[WHITE][sq] = ((b << 9) & C64(0xFEFEFEFEFEFEFEFE)) | ((b << 7) & C64(0x7F7F7F7F7F7F7F7F));
        pawn_attacks[BLACK][sq] = ((b >> 7) & C64(0xFEFEFEFEFEFEFEFE)) | ((b >> 9) & C64(0x7F7F7F7F7F7F7F7F));
        knight_attacks[sq] = king_attacks[sq] = 0;
        for (int i = 0; i < 8; i++) {
            int nr = r + dn[i][0], nf = f + dn[i][1], kr = r + dk[i][0], kf = f + dk[i][1];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) knight_attacks[sq] |= bit(nr * 8 + nf);
            if (kr >= 0 && kr < 8 && kf >= 0 && kf < 8) king_attacks[sq] |= bit(kr * 8 + kf);
        }
    }
    init_magics();
}

static inline u64 get_bishop_attacks(int sq, u64 occ) { Magic *m = &bishop_magics[sq]; return m->table[(unsigned int)(((occ & m->mask) * m->magic) >> m->shift)]; }
static inline u64 get_rook_attacks(int sq, u64 occ) { Magic *m = &rook_magics[sq]; return m->table[(unsigned int)(((occ & m->mask) * m->magic) >> m->shift)]; }
static inline u64 get_queen_attacks(int sq, u64 occ) { return get_bishop_attacks(sq, occ) | get_rook_attacks(sq, occ); }

typedef struct { u64 pieces[2][6], occ[3], hash; int side, castling, ep, halfmove, fullmove; uint8_t mailbox[64]; } Position;
typedef struct { u64 hash; int castling, ep, halfmove, captured; } UndoInfo;

/* ================================================================
   S3  ZOBRIST HASHING & TRANSPOSITION TABLE
   ================================================================

   Different move orders often reach the same position (a transposition).
   Zobrist hashing assigns a 64-bit random number to each piece on each
   square (768 values), the side to move, castling rights, and en passant
   files.

   Because exclusive-or is its own inverse (x ^ y ^ y == x), making or
   undoing a move updates the hash with simple XOR operations:
     pos->hash ^= zobrist_piece[color][piece][from];
     pos->hash ^= zobrist_piece[color][piece][to];
   No board rescanning is needed. A 64-bit xorshift generator seeds
   these tables deterministically at startup.

   The transposition table (TT) caches previous search results. If a
   position is visited again at equal or greater depth, the stored
   score or bound can be returned immediately. Even without a cutoff,
   the stored best move guides move ordering.

   Each entry occupies 16 bytes:
     key    64-bit hash to verify exact match
     score  search score
     move   best move found
     depth  search depth
     bound  EXACT, ALPHA (upper bound), or BETA (lower bound)
     age    generation counter to retire stale entries

   Mate scores depend on distance from root; we add ply before storing
   and subtract ply after probing so cached values remain independent of
   search depth.
*/

u64 zobrist_piece[2][6][64], zobrist_side, zobrist_castle[16], zobrist_ep[64];

static u64 rand64(u64 *s) { *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27; return *s * C64(0x2545F4914F6CDD1D); }

void init_zobrist(void) {
    u64 s = C64(0x9E3779B97F4A7C15);
    for (int i = 0; i < 768; i++) ((u64*)zobrist_piece)[i] = rand64(&s);
    zobrist_side = rand64(&s);
    for (int i = 0; i < 16; i++) zobrist_castle[i] = rand64(&s);
    for (int i = 0; i < 64; i++) zobrist_ep[i] = rand64(&s);
}

u64 generate_hash(const Position *pos) {
    u64 h = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++)
            for (u64 bb = pos->pieces[c][pt]; bb; ) h ^= zobrist_piece[c][pt][pop_lsb(&bb)];
    if (pos->side == BLACK) h ^= zobrist_side;
    if (pos->ep != SQ_NONE) h ^= zobrist_ep[pos->ep];
    return h ^ zobrist_castle[pos->castling];
}

enum { TT_NONE = 0, TT_EXACT = 1, TT_ALPHA = 2, TT_BETA = 3 };
typedef struct { u64 key; int16_t score; Move move; uint8_t depth, bound, age, padding; } TTEntry;
TTEntry *tt = NULL; size_t tt_mask = 0; uint8_t tt_age = 0;

int tt_allocate(size_t mb) {
    if (tt) free(tt);
    size_t count = (mb * 1024 * 1024) / sizeof(TTEntry), pow2 = 1;
    while (pow2 * 2 <= count) pow2 *= 2;
    tt = (TTEntry*)calloc(pow2, sizeof(TTEntry));
    return tt ? (tt_mask = pow2 - 1, 1) : 0;
}

void tt_clear(void) { if (tt) memset(tt, 0, (tt_mask + 1) * sizeof(TTEntry)); tt_age = 0; }

static inline int tt_probe(u64 hash, int depth, int alpha, int beta, int ply, Move *tt_move, int *tt_score, int *tt_bound) {
    if (!tt) return 0;
    TTEntry *e = &tt[hash & tt_mask];
    if (e->key != hash) return 0;
    *tt_move = e->move;
    int sc = e->score;
    if (sc > 30000 - 512) sc -= ply; else if (sc < -30000 + 512) sc += ply;
    *tt_score = sc;
    if (tt_bound) *tt_bound = e->bound;
    if (e->depth >= depth) {
        if (e->bound == TT_EXACT || (e->bound == TT_ALPHA && sc <= alpha) || (e->bound == TT_BETA && sc >= beta)) return 1;
    }
    return 0;
}

static inline void tt_store(u64 hash, int depth, int score, int bound, Move m, int ply) {
    if (!tt) return;
    TTEntry *e = &tt[hash & tt_mask];
    if (score > 30000 - 512) score += ply; else if (score < -30000 + 512) score -= ply;
    if (e->key != hash || depth >= e->depth || e->age != tt_age) {
        e->key = hash; e->score = (int16_t)score; e->depth = (uint8_t)depth; e->bound = (uint8_t)bound;
        if (m || e->key != hash) e->move = m;
        e->age = tt_age;
    }
}

/* ================================================================
   S4  NNUE EVALUATION & ACCUMULATOR
   ================================================================

   NNUE evaluates positions with a small neural network. The first
   layer (the accumulator) is a linear sum of weight vectors for
   each piece on the board:
     768 inputs (2 sides * 6 piece types * 64 squares) -> 32 values.

   Because the accumulator is purely additive, moving a piece requires
   only subtracting the weights of the vacated square and adding the
   weights of the new square. The 32 16-bit values fit across two
   256-bit AVX2 registers (via GCC's v16 vector type).

   Each side maintains its own perspective; Black's view flips the
   board vertically (sq ^ 56). The hidden values pass through a
   squared clipped linear unit (SCReLU: clamp to [0, 255], square),
   and the output layer computes an inner product scaled to centipawns.
*/

typedef int16_t i16; typedef int32_t i32; typedef i16 v16 __attribute__((vector_size(32)));

enum {
    NNUE_INPUT_SIZE  = 768, NNUE_HIDDEN_SIZE = 32,
    NNUE_VECTORS     = NNUE_HIDDEN_SIZE / 16,
    NNUE_EVAL_SCALE  = 400, NNUE_L0_SCALE = 255, NNUE_L1_SCALE = 64
};

typedef struct {
    i16 input_weights[NNUE_INPUT_SIZE][NNUE_HIDDEN_SIZE], input_biases[NNUE_HIDDEN_SIZE];
    i16 output_weights[2][NNUE_HIDDEN_SIZE], output_bias;
} NNUEParameters;

typedef union __attribute__((aligned(64))) {
    i16 accumulation[2][NNUE_HIDDEN_SIZE];
    v16 vec[2][NNUE_VECTORS];
} Accumulator;

typedef struct { Accumulator stack[1024]; int loaded; } NNUEState;
static __attribute__((aligned(64))) NNUEParameters nnue_params; static NNUEState nnue; int acc_ply = 0;

#include "net.h"

static inline i32 nnue_screlu(i32 x) { i32 c = (x < 0) ? 0 : ((x > NNUE_L0_SCALE) ? NNUE_L0_SCALE : x); return c * c; }

static inline void nnue_feat(int c, int pt, int sq, const v16 **pw, const v16 **pb) {
    *pw = (const v16 *)nnue_params.input_weights[c * 384 + pt * 64 + sq];
    *pb = (const v16 *)nnue_params.input_weights[(c ^ 1) * 384 + pt * 64 + (sq ^ 56)];
}

static inline void nnue_add1sub1(Accumulator *dst, const Accumulator *src, int ac, int apt, int asq, int sc, int spt, int ssq) {
    const v16 *aw, *ab, *sw, *sb;
    nnue_feat(ac, apt, asq, &aw, &ab); nnue_feat(sc, spt, ssq, &sw, &sb);
    for (int i = 0; i < NNUE_VECTORS; i++) {
        dst->vec[0][i] = src->vec[0][i] + aw[i] - sw[i];
        dst->vec[1][i] = src->vec[1][i] + ab[i] - sb[i];
    }
}

static inline void nnue_add1sub2(Accumulator *dst, const Accumulator *src, int ac, int apt, int asq, int s1c, int s1pt, int s1sq, int s2c, int s2pt, int s2sq) {
    const v16 *aw, *ab, *s1w, *s1b, *s2w, *s2b;
    nnue_feat(ac, apt, asq, &aw, &ab); nnue_feat(s1c, s1pt, s1sq, &s1w, &s1b); nnue_feat(s2c, s2pt, s2sq, &s2w, &s2b);
    for (int i = 0; i < NNUE_VECTORS; i++) {
        dst->vec[0][i] = src->vec[0][i] + aw[i] - s1w[i] - s2w[i];
        dst->vec[1][i] = src->vec[1][i] + ab[i] - s1b[i] - s2b[i];
    }
}

static inline void nnue_add2sub2(Accumulator *dst, const Accumulator *src, int a1c, int a1pt, int a1sq, int a2c, int a2pt, int a2sq, int s1c, int s1pt, int s1sq, int s2c, int s2pt, int s2sq) {
    const v16 *a1w, *a1b, *a2w, *a2b, *s1w, *s1b, *s2w, *s2b;
    nnue_feat(a1c, a1pt, a1sq, &a1w, &a1b); nnue_feat(a2c, a2pt, a2sq, &a2w, &a2b);
    nnue_feat(s1c, s1pt, s1sq, &s1w, &s1b); nnue_feat(s2c, s2pt, s2sq, &s2w, &s2b);
    for (int i = 0; i < NNUE_VECTORS; i++) {
        dst->vec[0][i] = src->vec[0][i] + a1w[i] + a2w[i] - s1w[i] - s2w[i];
        dst->vec[1][i] = src->vec[1][i] + a1b[i] + a2b[i] - s1b[i] - s2b[i];
    }
}

void nnue_refresh(const Position *pos) {
    acc_ply = 0;
    const v16 *bias = (const v16 *)nnue_params.input_biases;
    for (int i = 0; i < NNUE_VECTORS; i++) nnue.stack[0].vec[0][i] = nnue.stack[0].vec[1][i] = bias[i];
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++)
            for (u64 bb = pos->pieces[c][pt]; bb; ) {
                const v16 *w, *b;
                nnue_feat(c, pt, pop_lsb(&bb), &w, &b);
                for (int i = 0; i < NNUE_VECTORS; i++) {
                    nnue.stack[0].vec[0][i] += w[i];
                    nnue.stack[0].vec[1][i] += b[i];
                }
            }
}

int nnue_evaluate(const Position *pos) {
    int stm = pos->side;
    const Accumulator *acc = &nnue.stack[acc_ply];
    const i16 *acc0 = acc->accumulation[stm], *acc1 = acc->accumulation[stm ^ 1];
    const i16 *w0 = nnue_params.output_weights[0], *w1 = nnue_params.output_weights[1];
    i32 score = 0;
    for (int i = 0; i < NNUE_HIDDEN_SIZE; i++)
        score += nnue_screlu(acc0[i]) * (i32)w0[i] + nnue_screlu(acc1[i]) * (i32)w1[i];
    return (int)(((score / NNUE_L0_SCALE) + nnue_params.output_bias) * NNUE_EVAL_SCALE / (NNUE_L0_SCALE * NNUE_L1_SCALE));
}

int nnue_init(const char *path) {
    size_t need = sizeof(nnue_params);
    if ((size_t)nnue_embedded_size >= need) {
        memcpy(&nnue_params, nnue_embedded, need);
        return (nnue.loaded = 1);
    }
    if (path) {
        FILE *f = fopen(path, "rb");
        if (f) {
            size_t got = fread(&nnue_params, 1, need, f);
            fclose(f);
            return (nnue.loaded = (got == need));
        }
    }
    return 0;
}

/* ================================================================
   S5  BOARD REPRESENTATION & MAKE/UNDO
   ================================================================

   The board state is kept in two forms at once: bitboards (for fast
   piece sets, occupancy, and attack generation) and a 64-byte mailbox
   array (so we know what piece occupies a square without checking
   twelve bitboards). `set_piece` and `clear_piece` maintain both.

   To test whether square S is attacked by side C, we look outward
   from S along each piece's movement pattern: if a knight jump from S
   lands on a C knight, or a diagonal ray from S hits a C bishop or
   queen, the square is attacked. A king is in check when its square
   is attacked by the opposing side.

   `make_move` updates the board and records irreversible state
   (hash, castling rights, en passant square, 50-move clock, and the
   captured piece) in an UndoInfo structure on the stack. The NNUE
   accumulator is updated incrementally for the new ply. `undo_move`
   restores the previous board and decrements the accumulator stack.
   Castling rights are updated by ANDing a mask for each touched square:
     castling &= castle_rights_mask[from] & castle_rights_mask[to];
*/

static inline int is_square_attacked(const Position *pos, int sq, int s) {
    return (pawn_attacks[s ^ 1][sq] & pos->pieces[s][PAWN]) || (knight_attacks[sq] & pos->pieces[s][KNIGHT]) ||
           (king_attacks[sq] & pos->pieces[s][KING]) || (get_bishop_attacks(sq, pos->occ[BOTH]) & (pos->pieces[s][BISHOP] | pos->pieces[s][QUEEN])) ||
           (get_rook_attacks(sq, pos->occ[BOTH]) & (pos->pieces[s][ROOK] | pos->pieces[s][QUEEN]));
}

static inline int in_check(const Position *pos, int side) { return is_square_attacked(pos, lsb(pos->pieces[side][KING]), side ^ 1); }

static inline u64 compute_attacks(const Position *pos, int side) {
    u64 p = pos->pieces[side][PAWN];
    u64 atks = (side == WHITE) ? (((p << 9) & C64(0xFEFEFEFEFEFEFEFE)) | ((p << 7) & C64(0x7F7F7F7F7F7F7F7F)))
                               : (((p >> 7) & C64(0xFEFEFEFEFEFEFEFE)) | ((p >> 9) & C64(0x7F7F7F7F7F7F7F7F)));
    for (u64 n = pos->pieces[side][KNIGHT]; n; ) atks |= knight_attacks[pop_lsb(&n)];
    atks |= king_attacks[lsb(pos->pieces[side][KING])];
    for (u64 bq = pos->pieces[side][BISHOP] | pos->pieces[side][QUEEN]; bq; ) atks |= get_bishop_attacks(pop_lsb(&bq), pos->occ[BOTH]);
    for (u64 rq = pos->pieces[side][ROOK] | pos->pieces[side][QUEEN]; rq; ) atks |= get_rook_attacks(pop_lsb(&rq), pos->occ[BOTH]);
    return atks;
}

static inline void set_piece(Position *pos, int side, int pt, int sq) {
    pos->pieces[side][pt] |= bit(sq); pos->occ[side] |= bit(sq); pos->occ[BOTH] |= bit(sq); pos->mailbox[sq] = (uint8_t)pt;
}
static inline void clear_piece(Position *pos, int side, int pt, int sq) {
    pos->pieces[side][pt] &= ~bit(sq); pos->occ[side] &= ~bit(sq); pos->occ[BOTH] &= ~bit(sq); pos->mailbox[sq] = NO_PIECE;
}

#define CASTLE_WK_OCC (bit(F1) | bit(G1))
#define CASTLE_WQ_OCC (bit(B1) | bit(C1) | bit(D1))
#define CASTLE_BK_OCC (bit(F8) | bit(G8))
#define CASTLE_BQ_OCC (bit(B8) | bit(C8) | bit(D8))

const uint8_t castle_rights_mask[64] = { 13,15,15,15,12,15,15,14, 15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15,  7,15,15,15, 3,15,15,11 };

void make_move(Position *pos, Move m, UndoInfo *undo) {
    int from = move_from(m), to = move_to(m), flag = move_flag(m);
    int us = pos->side, them = us ^ 1, pt = pos->mailbox[from], cap = pos->mailbox[to];
    undo->hash = pos->hash; undo->castling = pos->castling; undo->ep = pos->ep; undo->halfmove = pos->halfmove; undo->captured = cap;

    pos->hash ^= zobrist_side;
    if (pos->ep != SQ_NONE) pos->hash ^= zobrist_ep[pos->ep];
    pos->ep = SQ_NONE;
    pos->halfmove = (pt == PAWN || cap != NO_PIECE) ? 0 : pos->halfmove + 1;

    Accumulator *src = &nnue.stack[acc_ply], *dst = &nnue.stack[acc_ply + 1];

    if (flag == FLAG_CASTLE) {
        clear_piece(pos, us, KING, from); set_piece(pos, us, KING, to);
        pos->hash ^= zobrist_piece[us][KING][from] ^ zobrist_piece[us][KING][to];
        int rf = (to == G1) ? H1 : (to == C1) ? A1 : (to == G8) ? H8 : A8;
        int rt = (to == G1) ? F1 : (to == C1) ? D1 : (to == G8) ? F8 : D8;
        clear_piece(pos, us, ROOK, rf); set_piece(pos, us, ROOK, rt);
        pos->hash ^= zobrist_piece[us][ROOK][rf] ^ zobrist_piece[us][ROOK][rt];
        if (nnue.loaded) nnue_add2sub2(dst, src, us, KING, to, us, ROOK, rt, us, KING, from, us, ROOK, rf);
    } else if (flag == FLAG_EP) {
        clear_piece(pos, us, PAWN, from); set_piece(pos, us, PAWN, to);
        int ep_cap_sq = (us == WHITE) ? to - 8 : to + 8;
        clear_piece(pos, them, PAWN, ep_cap_sq);
        pos->hash ^= zobrist_piece[us][PAWN][from] ^ zobrist_piece[us][PAWN][to] ^ zobrist_piece[them][PAWN][ep_cap_sq];
        if (nnue.loaded) nnue_add1sub2(dst, src, us, PAWN, to, us, PAWN, from, them, PAWN, ep_cap_sq);
    } else {
        if (cap != NO_PIECE) { clear_piece(pos, them, cap, to); pos->hash ^= zobrist_piece[them][cap][to]; }
        clear_piece(pos, us, pt, from); pos->hash ^= zobrist_piece[us][pt][from];
        int final_pt = (flag >= FLAG_PROMO_N) ? promo_piece(flag) : pt;
        set_piece(pos, us, final_pt, to); pos->hash ^= zobrist_piece[us][final_pt][to];
        if (flag == FLAG_DOUBLE_PUSH) { pos->ep = (us == WHITE) ? from + 8 : from - 8; pos->hash ^= zobrist_ep[pos->ep]; }
        if (nnue.loaded) {
            if (cap != NO_PIECE) nnue_add1sub2(dst, src, us, final_pt, to, us, pt, from, them, cap, to);
            else nnue_add1sub1(dst, src, us, final_pt, to, us, pt, from);
        }
    }

    acc_ply++;
    pos->hash ^= zobrist_castle[pos->castling];
    pos->castling &= castle_rights_mask[from] & castle_rights_mask[to];
    pos->hash ^= zobrist_castle[pos->castling];
    pos->side ^= 1;
}

void undo_move(Position *pos, Move m, const UndoInfo *undo) {
    acc_ply--; pos->side ^= 1;
    int from = move_from(m), to = move_to(m), flag = move_flag(m);
    int us = pos->side, them = us ^ 1;
    pos->hash = undo->hash; pos->castling = undo->castling; pos->ep = undo->ep; pos->halfmove = undo->halfmove;

    if (flag == FLAG_CASTLE) {
        clear_piece(pos, us, KING, to); set_piece(pos, us, KING, from);
        int rf = (to == G1) ? H1 : (to == C1) ? A1 : (to == G8) ? H8 : A8;
        int rt = (to == G1) ? F1 : (to == C1) ? D1 : (to == G8) ? F8 : D8;
        clear_piece(pos, us, ROOK, rt); set_piece(pos, us, ROOK, rf);
    } else if (flag == FLAG_EP) {
        clear_piece(pos, us, PAWN, to); set_piece(pos, us, PAWN, from);
        set_piece(pos, them, PAWN, (us == WHITE) ? to - 8 : to + 8);
    } else {
        int final_pt = (flag >= FLAG_PROMO_N) ? promo_piece(flag) : pos->mailbox[to];
        clear_piece(pos, us, final_pt, to);
        set_piece(pos, us, (flag >= FLAG_PROMO_N) ? PAWN : final_pt, from);
        if (undo->captured != NO_PIECE) set_piece(pos, them, undo->captured, to);
    }
}

/* ================================================================
   S6  MOVE GENERATION
   ================================================================

   A chess move is legal if the moving piece follows the rules and
   does not leave its own king in check. Testing for absolute pins
   and check evasions during generation is intricate and slow.
   Instead, `generate_moves` produces "pseudo-legal" moves: moves
   that obey the geometry of the piece regardless of pins. The
   search plays each candidate:
     make_move(pos, m, &undo);
     if (!in_check(pos, pos->side ^ 1)) { ... legal ... }
     undo_move(pos, m, &undo);
   and immediately discards any move that leaves the king exposed.
   Because most positions have few or no pinned pieces, this test
   after the fact is much faster than checking pins up front.

   The generator accepts a mode parameter to restrict output:
     0  All pseudo-legal moves, used in regular alpha-beta search.
     1  Captures and promotions only, used in quiescence search.
     2  Quiet moves only.

   The target mask (`opp_or_empty`) determines valid destination
   squares in a single bitwise operation:
     Mode 1 : pos->occ[them]   (enemy-occupied squares only)
     Mode 2 : ~pos->occ[BOTH]  (empty squares only)
     Mode 0 : ~pos->occ[us]    (any square not holding a friendly piece)

   Pawn moves are generated separately because pawns move and capture
   differently. We test single pushes into empty squares, double
   pushes from the starting rank (provided both intervening squares
   are empty), diagonal captures into enemy squares, en passant
   captures when an en passant square is active, and expand any move
   landing on the back rank into four promotions (Q, R, B, N).

   Knights, kings, and sliding pieces (bishops, rooks, queens)
   generate moves by intersecting their attack bitboards with
   `opp_or_empty`. We then extract destination squares one by one
   with `pop_lsb` and append the encoded moves to the move list.

   Castling verifies that the king and chosen rook have not moved,
   that the squares between them are completely empty, and that the
   king is not currently in check, does not cross an attacked square,
   and does not land in check.
*/

void generate_moves(const Position *pos, MoveList *list, int mode) {
    list->count = 0;
    int us = pos->side, them = us ^ 1, promo_rank = (us == WHITE) ? 7 : 0, start_rank = (us == WHITE) ? 1 : 6;
    u64 opp_or_empty = (mode == 1) ? pos->occ[them] : (mode == 2) ? ~pos->occ[BOTH] : ~pos->occ[us];

    for (u64 pawns = pos->pieces[us][PAWN]; pawns; ) {
        int sq = pop_lsb(&pawns), r = sq / 8, single_to = (us == WHITE) ? sq + 8 : sq - 8;
        if (single_to >= 0 && single_to < 64 && !(pos->occ[BOTH] & bit(single_to))) {
            if (single_to / 8 == promo_rank) push_promos(list, sq, single_to);
            else if (mode != 1) {
                push_move(list, encode_move(sq, single_to, FLAG_QUIET));
                int double_to = (us == WHITE) ? sq + 16 : sq - 16;
                if (r == start_rank && !(pos->occ[BOTH] & bit(double_to)))
                    push_move(list, encode_move(sq, double_to, FLAG_DOUBLE_PUSH));
            }
        }
        if (mode != 2) {
            for (u64 atks = pawn_attacks[us][sq] & pos->occ[them]; atks; ) {
                int to = pop_lsb(&atks);
                if (to / 8 == promo_rank) push_promos(list, sq, to);
                else push_move(list, encode_move(sq, to, FLAG_QUIET));
            }
            if (pos->ep != SQ_NONE && (pawn_attacks[us][sq] & bit(pos->ep)))
                push_move(list, encode_move(sq, pos->ep, FLAG_EP));
        }
    }

    for (u64 n = pos->pieces[us][KNIGHT]; n; ) {
        int sq = pop_lsb(&n);
        for (u64 a = knight_attacks[sq] & opp_or_empty; a; ) push_move(list, encode_move(sq, pop_lsb(&a), FLAG_QUIET));
    }
    for (u64 bq = pos->pieces[us][BISHOP] | pos->pieces[us][QUEEN]; bq; ) {
        int sq = pop_lsb(&bq);
        for (u64 a = get_bishop_attacks(sq, pos->occ[BOTH]) & opp_or_empty; a; ) push_move(list, encode_move(sq, pop_lsb(&a), FLAG_QUIET));
    }
    for (u64 rq = pos->pieces[us][ROOK] | pos->pieces[us][QUEEN]; rq; ) {
        int sq = pop_lsb(&rq);
        for (u64 a = get_rook_attacks(sq, pos->occ[BOTH]) & opp_or_empty; a; ) push_move(list, encode_move(sq, pop_lsb(&a), FLAG_QUIET));
    }
    int ksq = lsb(pos->pieces[us][KING]);
    for (u64 a = king_attacks[ksq] & opp_or_empty; a; ) push_move(list, encode_move(ksq, pop_lsb(&a), FLAG_QUIET));

    if (mode != 1) {
        int k = (us == WHITE) ? E1 : E8;
        int cr_k = (us == WHITE) ? CASTLE_WK : CASTLE_BK, cr_q = (us == WHITE) ? CASTLE_WQ : CASTLE_BQ;
        u64 occ_k = (us == WHITE) ? CASTLE_WK_OCC : CASTLE_BK_OCC, occ_q = (us == WHITE) ? CASTLE_WQ_OCC : CASTLE_BQ_OCC;
        if ((pos->castling & cr_k) && !(pos->occ[BOTH] & occ_k) && !is_square_attacked(pos, k, them) && !is_square_attacked(pos, k + 1, them) && !is_square_attacked(pos, k + 2, them))
            push_move(list, encode_move(k, k + 2, FLAG_CASTLE));
        if ((pos->castling & cr_q) && !(pos->occ[BOTH] & occ_q) && !is_square_attacked(pos, k, them) && !is_square_attacked(pos, k - 1, them) && !is_square_attacked(pos, k - 2, them))
            push_move(list, encode_move(k, k - 2, FLAG_CASTLE));
    }
}

/* ================================================================
   S7  SEARCH & HEURISTICS
   ================================================================

   The search uses the negamax formulation of alpha-beta minimax.
   Because chess is a zero-sum game, White's advantage is Black's
   disadvantage. Evaluating every position from the perspective of
   the side to move reduces the recursive step to a single symmetry:
     score = -search(-beta, -alpha);
   Alpha is the lower bound: the best score the mover can guarantee.
   Beta is the upper bound: the score the opponent can hold them to.
   If any move returns a score >= beta, the opponent would have avoided
   this branch earlier in the tree. We cut off search immediately
   (a "beta cutoff") and ignore the remaining moves.

   Alpha-beta is fastest when the strongest moves are searched first.
   Move ordering sorts candidates before searching:
     1. TT move      : The best move found at this position in an
                       earlier or shallower search iteration.
     2. Good captures: Captures ordered by MVV-LVA (Most Valuable
                       Victim, Least Valuable Attacker) and verified
                       by Static Exchange Evaluation (SEE >= 0).
     3. Killer moves : Quiet moves that caused a beta cutoff at this
                       same ply in a sibling branch.
     4. History      : Butterfly tables tracking quiet moves that
                       frequently cause cutoffs, indexed by [from][to]
                       and refined by threat context (whether from/to
                       squares are under attack).

   Quiescence search (`qsearch`) prevents the "horizon effect", where
   the engine might stop searching right before a queen is recaptured.
   It examines only captures and promotions until the board settles.
   If the current static evaluation ("stand-pat") already meets or
   exceeds beta, the side to move can simply refuse to capture further,
   allowing an immediate cutoff.

   Selective pruning and reductions keep the search tree manageable:
     - Reverse futility : If static eval beats beta by a wide margin
                          at low depth, return eval immediately.
     - Razoring         : If static eval is far below alpha near leaf
                          nodes, verify with a quick qsearch cutoff.
     - Null-move pruning: Pass our turn to the opponent; if our score
                          still beats beta at reduced depth, cutoff.
                          Disabled in king-and-pawn endgames to avoid
                          zugzwang errors.
     - Late-move red.   : Moves ordered late in the list are unlikely
                          to beat alpha. We search them at reduced depth
                          (LMR), re-searching at full depth only if they
                          surprise us by beating alpha. Moves that escape
                          an opponent threat receive less reduction.
     - Futility & LMP   : Skip late quiet moves entirely at low depths
                          when winning is unlikely.
     - Aspiration window: Search with a narrow [score-delta, score+delta]
                          window around the previous depth's score,
                          widening only on a fail-high or fail-low.
*/

enum { INF = 32000, MATE = 30000, MAX_PLY = 64 };
static inline int is_tactical(const Position *pos, Move m) { return (pos->mailbox[move_to(m)] != NO_PIECE) || (move_flag(m) >= FLAG_EP); }

u64 pos_history[1024];
int pos_history_count = 0, time_over_flag = 0;
clock_t search_start_time;
int64_t search_hard_limit_ms = 0, search_soft_limit_ms = 0;
u64 nodes_searched = 0;
Move best_root_move = 0;

static inline void check_time(void) {
    if (search_hard_limit_ms > 0 && (nodes_searched & 2047) == 0) {
        if ((int64_t)((clock() - search_start_time) * 1000 / CLOCKS_PER_SEC) >= search_hard_limit_ms)
            time_over_flag = 1;
    }
}

static inline int is_material_draw(const Position *pos) {
    if (pos->pieces[WHITE][PAWN] | pos->pieces[BLACK][PAWN] |
        pos->pieces[WHITE][ROOK] | pos->pieces[BLACK][ROOK] |
        pos->pieces[WHITE][QUEEN] | pos->pieces[BLACK][QUEEN])
        return 0;

    int wn = popcount(pos->pieces[WHITE][KNIGHT]), bn = popcount(pos->pieces[BLACK][KNIGHT]);
    int wb = popcount(pos->pieces[WHITE][BISHOP]), bb = popcount(pos->pieces[BLACK][BISHOP]);
    int wm = wn + wb, bm = bn + bb;

    if (wm == 0 && bm <= 1) return 1;
    if (bm == 0 && wm <= 1) return 1;
    if (wm == 1 && bm == 1) return 1;
    if (wm == 2 && wn == 2 && bm == 0) return 1;
    if (bm == 2 && bn == 2 && wm == 0) return 1;
    return 0;
}

static inline int is_draw(const Position *pos, int ply) {
    if (pos->halfmove >= 100 || is_material_draw(pos)) return 1;
    if (ply > 0) {
        int limit = (pos->halfmove < pos_history_count - 1) ? pos->halfmove : pos_history_count - 1;
        for (int i = 2; i <= limit; i += 2) {
            if (i > ply && i < 4) continue;
            if (pos_history[pos_history_count - 1 - i] == pos->hash) return 1;
        }
    }
    return 0;
}

enum {
    SCORE_ROOT_PV = 300000, SCORE_TT_MOVE = 200000, SCORE_PROMO_BASE = 60000, SCORE_CAPTURE_BASE = 40000,
    SCORE_KILLER_1 = 25000, SCORE_KILLER_2 = 20000, MAX_HISTORY = 16384, MAX_BONUS = 2000,
    SEE_PRUNING_MAX_DEPTH = 3, SEE_PRUNING_NOISY_MARGIN = -120, SEE_PRUNING_QUIET_MARGIN = -60,
    RFP_MAX_DEPTH = 6, RFP_MARGIN = 75, NMP_MIN_DEPTH = 3, NMP_EVAL_MARGIN = 30,
    FP_MAX_DEPTH = 3, FP_MARGIN = 90, LMP_MAX_DEPTH = 5, HP_MAX_DEPTH = 4, HP_MARGIN = 2048,
    RAZORING_MAX_DEPTH = 3, RAZORING_MARGIN = 150
};

Move killers[2][MAX_PLY];
int history[2][64][64][2][2], lmr_table[64][64];
Move root_best_move = 0;

static void init_lmr(void) {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++) {
            double r = log((double)d) * log((double)m) / 2.1350 + 0.2319;
            lmr_table[d][m] = r < 0.0 ? 0 : (int)r;
        }
    lmr_table[0][0] = lmr_table[0][1] = lmr_table[1][0] = lmr_table[1][1] = 0;
}

typedef struct { int count; Move moves[MAX_PLY]; } PVLine;
static inline void pvline_clear(PVLine *pv) { pv->count = 0; }
static inline void pvline_update(PVLine *pv, Move m, const PVLine *child) {
    pv->moves[0] = m;
    int len = child ? child->count : 0;
    if (len > MAX_PLY - 1) len = MAX_PLY - 1;
    if (len > 0) memcpy(&pv->moves[1], child->moves, len * sizeof(Move));
    pv->count = 1 + len;
}

static inline void print_score(int s) {
    if (s > MATE - 512) printf("score mate %d ", (MATE - s + 1) / 2);
    else if (s < -MATE + 512) printf("score mate %d ", -(MATE + s + 1) / 2);
    else printf("score cp %d ", s);
}

static inline int history_bonus(int d) { int b = d * d; return b > MAX_BONUS ? MAX_BONUS : b; }
static inline void update_history(int side, int from, int to, int src_th, int dst_th, int bonus) {
    int cl = bonus > MAX_BONUS ? MAX_BONUS : (bonus < -MAX_BONUS ? -MAX_BONUS : bonus);
    int cur = history[side][from][to][src_th][dst_th];
    history[side][from][to][src_th][dst_th] = cur + cl - (cur * (cl < 0 ? -cl : cl)) / MAX_HISTORY;
}

static const int mvv_lva[7][7] = {
    { 15, 14, 13, 12, 11, 10, 0 }, { 25, 24, 23, 22, 21, 20, 0 },
    { 35, 34, 33, 32, 31, 30, 0 }, { 45, 44, 43, 42, 41, 40, 0 },
    { 55, 54, 53, 52, 51, 50, 0 }, { 0 }, { 0 }
};

static inline int score_move(const Position *pos, Move m, int ply, u64 opp_threats) {
    if (ply == 0 && m == root_best_move) return SCORE_ROOT_PV;
    int from = move_from(m), to = move_to(m), flag = move_flag(m);
    int cap = (flag == FLAG_EP) ? PAWN : pos->mailbox[to], pt = pos->mailbox[from];
    if (flag >= FLAG_PROMO_N) return SCORE_PROMO_BASE + (flag - FLAG_PROMO_N + 1) * 100;
    if (cap != NO_PIECE) return SCORE_CAPTURE_BASE + mvv_lva[cap][pt];
    if (ply < MAX_PLY) {
        if (m == killers[0][ply]) return SCORE_KILLER_1;
        if (m == killers[1][ply]) return SCORE_KILLER_2;
    }
    return history[pos->side][from][to][(opp_threats & bit(from)) != 0][(opp_threats & bit(to)) != 0];
}

static const int see_values[7] = { 100, 300, 300, 500, 900, 0, 0 };

static inline u64 attacks_to_square(const Position *pos, int sq, u64 occ) {
    u64 wp = pos->pieces[WHITE][PAWN], bp = pos->pieces[BLACK][PAWN];
    u64 n = pos->pieces[WHITE][KNIGHT] | pos->pieces[BLACK][KNIGHT];
    u64 k = pos->pieces[WHITE][KING] | pos->pieces[BLACK][KING];
    u64 bq = pos->pieces[WHITE][BISHOP] | pos->pieces[BLACK][BISHOP] | pos->pieces[WHITE][QUEEN] | pos->pieces[BLACK][QUEEN];
    u64 rq = pos->pieces[WHITE][ROOK] | pos->pieces[BLACK][ROOK] | pos->pieces[WHITE][QUEEN] | pos->pieces[BLACK][QUEEN];
    return (pawn_attacks[BLACK][sq] & wp) | (pawn_attacks[WHITE][sq] & bp) |
           (knight_attacks[sq] & n) | (king_attacks[sq] & k) |
           (get_bishop_attacks(sq, occ) & bq) | (get_rook_attacks(sq, occ) & rq);
}

int see_ge(const Position *pos, Move m, int threshold) {
    int flag = move_flag(m);
    if (flag == FLAG_CASTLE) return threshold <= 0;

    int from_sq = move_from(m), to_sq = move_to(m), moving_piece = pos->mailbox[from_sq];
    int captured = (flag == FLAG_EP) ? PAWN : pos->mailbox[to_sq];
    int promo = (flag >= FLAG_PROMO_N) ? promo_piece(flag) : NO_PIECE;

    int score = ((captured != NO_PIECE) ? see_values[captured] : 0) - threshold;

    if (promo != NO_PIECE) {
        score += see_values[promo] - see_values[PAWN];
        if (score < 0) return 0;
        score -= see_values[promo];
        if (score >= 0) return 1;
    } else {
        if (score < 0) return 0;
        score -= see_values[moving_piece];
        if (score >= 0) return 1;
    }

    u64 occ = pos->occ[BOTH] ^ bit(from_sq) ^ bit(to_sq);
    if (flag == FLAG_EP) occ ^= bit((pos->side == WHITE) ? to_sq - 8 : to_sq + 8);

    u64 diag = pos->pieces[WHITE][BISHOP] | pos->pieces[BLACK][BISHOP] | pos->pieces[WHITE][QUEEN] | pos->pieces[BLACK][QUEEN];
    u64 straight = pos->pieces[WHITE][ROOK] | pos->pieces[BLACK][ROOK] | pos->pieces[WHITE][QUEEN] | pos->pieces[BLACK][QUEEN];

    u64 attackers = attacks_to_square(pos, to_sq, occ) & occ;
    int stm = pos->side ^ 1;

    while (1) {
        u64 our_attackers = attackers & pos->occ[stm];
        if (!our_attackers) break;

        int next_piece = NO_PIECE;
        u64 least_attacker_bb = 0;

        for (int pt = PAWN; pt <= KING; pt++) {
            u64 bb = our_attackers & pos->pieces[stm][pt];
            if (bb) { least_attacker_bb = bit(lsb(bb)); next_piece = pt; break; }
        }

        if (next_piece == KING && (attackers & pos->occ[stm ^ 1])) break;

        occ ^= least_attacker_bb;
        if (next_piece == PAWN || next_piece == BISHOP || next_piece == QUEEN)
            attackers |= get_bishop_attacks(to_sq, occ) & diag;
        if (next_piece == ROOK || next_piece == QUEEN)
            attackers |= get_rook_attacks(to_sq, occ) & straight;
        attackers &= occ;

        score = -score - 1 - see_values[next_piece];
        stm ^= 1;
        if (score >= 0) break;
    }

    return stm != pos->side;
}

int qsearch(Position *pos, int alpha, int beta, int ply) {
    nodes_searched++; check_time();
    if (time_over_flag) return 0;
    if (ply > 0 && is_draw(pos, ply)) return 0;
    if (ply >= MAX_PLY - 1) return in_check(pos, pos->side) ? 0 : nnue_evaluate(pos);

    int is_pv = (beta - alpha > 1), tt_score = 0;
    Move tt_move = 0;
    if (tt_probe(pos->hash, 0, alpha, beta, ply, &tt_move, &tt_score, NULL) && !is_pv && ply > 0)
        return tt_score;

    int check = in_check(pos, pos->side), best_score = -INF, old_alpha = alpha;

    if (!check) {
        int stand_pat = nnue_evaluate(pos);
        if (stand_pat >= beta) return stand_pat;
        if (stand_pat > alpha) alpha = stand_pat;
        best_score = stand_pat;
    }

    MoveList list;
    generate_moves(pos, &list, check ? 0 : 1);

    int scores[256];
    for (int i = 0; i < list.count; i++)
        scores[i] = (list.moves[i] == tt_move) ? SCORE_TT_MOVE : score_move(pos, list.moves[i], ply, 0);

    UndoInfo undo;
    int legal = 0;
    Move best_move = 0;

    for (int i = 0; i < list.count; i++) {
        int best_idx = i;
        for (int j = i + 1; j < list.count; j++)
            if (scores[j] > scores[best_idx]) best_idx = j;
        if (best_idx != i) {
            int ts = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = ts;
            Move tm = list.moves[i]; list.moves[i] = list.moves[best_idx]; list.moves[best_idx] = tm;
        }

        Move m = list.moves[i];
        if (!check && !see_ge(pos, m, 0)) continue;

        make_move(pos, m, &undo);
        if (!in_check(pos, pos->side ^ 1)) {
            legal++;
            pos_history[pos_history_count++] = pos->hash;
            int score = -qsearch(pos, -beta, -alpha, ply + 1);
            pos_history_count--;
            undo_move(pos, m, &undo);

            if (time_over_flag) return 0;
            if (score > best_score) { best_score = score; best_move = m; }
            if (score >= beta) { tt_store(pos->hash, 0, score, TT_BETA, m, ply); return score; }
            if (score > alpha) alpha = score;
            continue;
        }
        undo_move(pos, m, &undo);
    }

    if (check && legal == 0) return -(MATE - ply);
    if (!time_over_flag) tt_store(pos->hash, 0, best_score, (best_score <= old_alpha) ? TT_ALPHA : TT_EXACT, best_move, ply);
    return best_score;
}

int search(Position *pos, int depth, int alpha, int beta, int ply, PVLine *pv, int was_null) {
    if (pv) pvline_clear(pv);
    check_time();
    if (time_over_flag) return 0;
    if (ply >= MAX_PLY - 1) return nnue_evaluate(pos);
    if (ply > 0 && is_draw(pos, ply)) return 0;

    /* Mate distance pruning */
    if (ply > 0) {
        int r_alpha = (alpha > -MATE + ply) ? alpha : -MATE + ply;
        int r_beta  = (beta < MATE - ply - 1) ? beta : MATE - ply - 1;
        if (r_alpha >= r_beta) return r_alpha;
        alpha = r_alpha; beta = r_beta;
    }

    int in_chk = in_check(pos, pos->side);
    if (in_chk && ply > 0) depth++;

    int is_pv = (beta - alpha > 1), tt_score = 0, tt_bound = TT_NONE;
    Move tt_move = 0;
    if (tt_probe(pos->hash, depth, alpha, beta, ply, &tt_move, &tt_score, &tt_bound) && !is_pv && ply > 0)
        return tt_score;

    /* Internal Iterative Reductions (IIR) */
    if (depth >= 4 && !tt_move && !in_chk) depth--;
    if (depth <= 0) return qsearch(pos, alpha, beta, ply);
    nodes_searched++;

    int static_eval = in_chk ? -INF : nnue_evaluate(pos);

    /* TT Evaluation Bound Refinement */
    if (tt_bound != TT_NONE) {
        if ((tt_bound == TT_BETA || tt_bound == TT_EXACT) && tt_score > static_eval) static_eval = tt_score;
        else if ((tt_bound == TT_ALPHA || tt_bound == TT_EXACT) && tt_score < static_eval) static_eval = tt_score;
    }

    /* Reverse Futility Pruning (RFP) */
    if (!is_pv && ply > 0 && !in_chk && depth <= RFP_MAX_DEPTH && beta < MATE - MAX_PLY) {
        int margin = RFP_MARGIN * depth;
        if (static_eval - margin >= beta) return static_eval - margin;
    }

    /* Razoring */
    if (!is_pv && ply > 0 && !in_chk && depth <= RAZORING_MAX_DEPTH && alpha > -MATE + MAX_PLY) {
        if (static_eval + RAZORING_MARGIN * depth <= alpha) {
            int razor_score = qsearch(pos, alpha, beta, ply);
            if (time_over_flag) return 0;
            if (razor_score <= alpha) return razor_score;
        }
    }

    /* Null Move Pruning (NMP) */
    if (!is_pv && ply > 0 && !was_null && !in_chk && depth >= NMP_MIN_DEPTH && beta < MATE - MAX_PLY) {
        u64 non_pawns = pos->pieces[pos->side][KNIGHT] | pos->pieces[pos->side][BISHOP] | pos->pieces[pos->side][ROOK] | pos->pieces[pos->side][QUEEN];
        if (non_pawns && static_eval >= beta + NMP_EVAL_MARGIN) {
            int R = 3 + depth / 4 + (static_eval - beta) / 200;
            if (R > 6) R = 6;

            int ep_prev = pos->ep;
            pos->hash ^= zobrist_side;
            if (pos->ep != SQ_NONE) pos->hash ^= zobrist_ep[pos->ep];
            pos->ep = SQ_NONE;
            pos->side ^= 1;
            pos_history[pos_history_count++] = pos->hash;

            int null_score = -search(pos, depth - 1 - R, -beta, -beta + 1, ply + 1, NULL, 1);

            pos_history_count--;
            pos->side ^= 1;
            pos->ep = ep_prev;
            if (pos->ep != SQ_NONE) pos->hash ^= zobrist_ep[pos->ep];
            pos->hash ^= zobrist_side;

            if (time_over_flag) return 0;
            if (null_score >= beta) return null_score >= (MATE - MAX_PLY) ? beta : null_score;
        }
    }

    MoveList list;
    generate_moves(pos, &list, 0);

    u64 opp_threats = compute_attacks(pos, pos->side ^ 1);

    int scores[256];
    for (int i = 0; i < list.count; i++)
        scores[i] = (list.moves[i] == tt_move) ? SCORE_TT_MOVE : score_move(pos, list.moves[i], ply, opp_threats);

    Move quiets[256], best_move = 0;
    int num_quiets = 0, best_score = -INF, bound = TT_ALPHA, legal = 0;
    UndoInfo undo;

    for (int i = 0; i < list.count; i++) {
        int best_idx = i;
        for (int j = i + 1; j < list.count; j++)
            if (scores[j] > scores[best_idx]) best_idx = j;
        if (best_idx != i) {
            int ts = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = ts;
            Move tm = list.moves[i]; list.moves[i] = list.moves[best_idx]; list.moves[best_idx] = tm;
        }

        Move m = list.moves[i];
        int from = move_from(m), to = move_to(m);
        int src_th = (opp_threats & bit(from)) != 0, dst_th = (opp_threats & bit(to)) != 0;
        int is_noisy = is_tactical(pos, m);
        int is_killer = (ply < MAX_PLY && (m == killers[0][ply] || m == killers[1][ply]));
        int hist = history[pos->side][from][to][src_th][dst_th];

        /* History Pruning */
        if (!is_pv && !in_chk && depth <= HP_MAX_DEPTH && best_score > -INF && !is_noisy && !is_killer && hist < -HP_MARGIN * (depth - 1))
            continue;

        /* Late Move Pruning */
        if (!is_pv && !in_chk && depth <= LMP_MAX_DEPTH && best_score > -INF && !is_noisy && legal >= 3 + depth * depth)
            continue;

        /* Futility Pruning */
        if (!is_pv && !in_chk && depth <= FP_MAX_DEPTH && best_score > -INF && !is_noisy && static_eval + FP_MARGIN * depth <= alpha)
            continue;

        /* SEE Pruning */
        if (!in_chk && depth <= SEE_PRUNING_MAX_DEPTH && best_score > -INF) {
            int margin = is_noisy ? (SEE_PRUNING_NOISY_MARGIN * depth) : (SEE_PRUNING_QUIET_MARGIN * depth);
            if (!see_ge(pos, m, margin)) continue;
        }

        make_move(pos, m, &undo);
        if (!in_check(pos, pos->side ^ 1)) {
            legal++;
            if (!is_noisy && num_quiets < 256) quiets[num_quiets++] = m;
            pos_history[pos_history_count++] = pos->hash;

            int gives_check = in_check(pos, pos->side);
            PVLine child_pv = { 0 };
            int score = 0;

            /* Late Move Reductions (LMR) */
            if (depth > 1 && legal > 1 && !(is_pv && is_noisy)) {
                int d_idx = depth < 64 ? depth : 63, m_idx = legal < 64 ? legal : 63;
                int R = lmr_table[d_idx][m_idx] - history[pos->side ^ 1][from][to][src_th][dst_th] / 2048;
                if (!is_pv) R += 2;
                if (is_killer) R -= 2;
                if (gives_check) R--;
                if (!is_noisy && src_th && !dst_th) R--; /* Escape Reduction */

                if (R < 1) R = 1;
                if (R > depth - 1) R = depth - 1;

                score = -search(pos, depth - 1 - R, -alpha - 1, -alpha, ply + 1, &child_pv, 0);
                if (score > alpha && R > 1)
                    score = -search(pos, depth - 1, -alpha - 1, -alpha, ply + 1, &child_pv, 0);
            } else if (!is_pv || legal > 1) {
                score = -search(pos, depth - 1, -alpha - 1, -alpha, ply + 1, &child_pv, 0);
            }

            if (is_pv && (legal == 1 || (score > alpha && score < beta)))
                score = -search(pos, depth - 1, -beta, -alpha, ply + 1, &child_pv, 0);

            pos_history_count--;
            undo_move(pos, m, &undo);

            if (time_over_flag) return 0;
            if (score > best_score) best_score = score;
            if (score >= beta) {
                if (!is_tactical(pos, m)) {
                    if (ply < MAX_PLY && m != killers[0][ply]) {
                        killers[1][ply] = killers[0][ply];
                        killers[0][ply] = m;
                    }
                    int bonus = history_bonus(depth);
                    update_history(pos->side, from, to, src_th, dst_th, bonus);
                    for (int q = 0; q < num_quiets - 1; q++) {
                        Move qm = quiets[q];
                        int qf = move_from(qm), qt = move_to(qm);
                        update_history(pos->side, qf, qt, (opp_threats & bit(qf)) != 0, (opp_threats & bit(qt)) != 0, -bonus);
                    }
                }
                tt_store(pos->hash, depth, score, TT_BETA, m, ply);
                return score;
            }
            if (score > alpha) {
                alpha = score; bound = TT_EXACT; best_move = m;
                if (pv) pvline_update(pv, m, &child_pv);
            }
            continue;
        }
        undo_move(pos, m, &undo);
    }

    if (!legal) return in_chk ? -(MATE - ply) : 0;
    tt_store(pos->hash, depth, best_score, bound, best_move, ply);
    return best_score;
}

static void print_move(Move m) {
    if (!m) return;
    int from = move_from(m), to = move_to(m), flag = move_flag(m);
    printf("%c%c%c%c", (from % 8) + 'a', (from / 8) + '1', (to % 8) + 'a', (to / 8) + '1');
    if (flag >= FLAG_PROMO_N) printf("%c", "nbrq"[flag - FLAG_PROMO_N]);
}

int search_root(Position *pos, int max_depth, int64_t hard_limit_ms, int64_t soft_limit_ms) {
    tt_age++; search_start_time = clock();
    search_hard_limit_ms = hard_limit_ms; search_soft_limit_ms = soft_limit_ms;
    time_over_flag = nodes_searched = root_best_move = 0;
    memset(killers, 0, sizeof(killers));
    nnue_refresh(pos);

    PVLine root_pv = { 0 };

    /* Initialize fallback with first strictly legal move */
    MoveList root_list;
    generate_moves(pos, &root_list, 0);
    UndoInfo root_undo;
    for (int i = 0; i < root_list.count; i++) {
        make_move(pos, root_list.moves[i], &root_undo);
        int legal = !in_check(pos, pos->side ^ 1);
        undo_move(pos, root_list.moves[i], &root_undo);
        if (legal) { root_pv.moves[0] = root_list.moves[i]; root_pv.count = 1; break; }
    }

    int score = 0, prev_score = 0;

    for (int depth = 1; depth <= max_depth; depth++) {
        PVLine iter_pv = { 0 };

        if (depth < 5) {
            score = search(pos, depth, -INF, INF, 0, &iter_pv, 0);
        } else {
            int delta = 12, search_depth = depth;
            int alpha = (prev_score - delta > -INF) ? (prev_score - delta) : -INF;
            int beta  = (prev_score + delta <  INF) ? (prev_score + delta) :  INF;

            while (1) {
                if (alpha < -2000) alpha = -INF;
                if (beta > 2000) beta = INF;

                score = search(pos, search_depth, alpha, beta, 0, &iter_pv, 0);
                if (time_over_flag) break;

                if (score <= alpha) {
                    beta = (alpha + beta) / 2;
                    alpha = (alpha - delta > -INF) ? (alpha - delta) : -INF;
                    search_depth = depth;
                } else if (score >= beta) {
                    beta = (beta + delta < INF) ? (beta + delta) : INF;
                    if (search_depth > 1) search_depth--;
                } else break;
                delta += delta / 2;
            }
        }

        if (time_over_flag && root_pv.count > 0) break;
        prev_score = score;

        if (iter_pv.count > 0) {
            root_pv = iter_pv;
            root_best_move = root_pv.moves[0];
        }

        clock_t elapsed = clock() - search_start_time;
        int64_t ms = (int64_t)(elapsed * 1000 / CLOCKS_PER_SEC);
        u64 nps = ms > 0 ? (nodes_searched * 1000 / ms) : 0;

        printf("info depth %d ", depth);
        print_score(score);
        printf("nodes %llu time %lld nps %llu pv ", (unsigned long long)nodes_searched, (long long)ms, (unsigned long long)nps);
        for (int i = 0; i < root_pv.count; i++) { print_move(root_pv.moves[i]); printf(" "); }
        printf("\n");
        fflush(stdout);

        if (score > MATE - 512 || score < -MATE + 512) break;

        if (search_soft_limit_ms > 0 && depth >= 4) {
            int64_t current_soft = search_soft_limit_ms;
            if (depth >= 6) {
                int complexity = abs(score - nnue_evaluate(pos));
                if (complexity > 200) complexity = 200;
                current_soft = (int64_t)(current_soft * (0.8 + 0.4 * ((double)complexity / 200.0)));
            }
            if (ms >= current_soft) break;
        }
    }

    printf("bestmove ");
    if (root_pv.count > 0) print_move(root_pv.moves[0]);
    printf("\n");
    fflush(stdout);
    return score;
}

/* ================================================================
   S8  UCI & BENCHMARKS
   ================================================================

   The Universal Chess Interface (UCI) is an open text protocol
   spoken between chess engines and graphical user interfaces
   (such as Arena, Cutechess, or Lichess bots) over standard
   input and standard output.

   The engine processes commands line by line:
     uci        : Identify engine name and author, and report options.
     isready    : Synchronization ping; engine replies "readyok".
     ucinewgame : Prepare for a new game; clear hash and history tables.
     position   : Set up the board from FEN or "startpos", then replay
                  any subsequent move history.
     go         : Begin searching under the specified clock or depth
                  limits, then print "bestmove".
     quit       : Terminate the program.

   Time management calculates two boundaries from the clock parameters:
     - Soft limit : The target search time, scaled by position
                    volatility (eval vs search divergence). Iterative
                    deepening stops cleanly after completing any depth
                    iteration past this duration.
     - Hard limit : The emergency ceiling. The search polls the clock
                    every 2048 nodes and terminates immediately if this
                    boundary is crossed mid-depth.

   `perft` (Performance Test) counts all leaf paths to a specified depth
   by recursive move generation without evaluation or pruning. Because
   the exact count of positions at each depth is a mathematical
   invariant, comparing results against reference values provides a
   definitive test of move generation, check detection, and make/undo.
*/

void parse_fen(Position *pos, const char *fen) {
    memset(pos, 0, sizeof(Position));
    for (int i = 0; i < 64; i++) pos->mailbox[i] = NO_PIECE;
    int r = 7, f = 0;
    while (*fen && *fen != ' ') {
        if (*fen == '/') { r--; f = 0; }
        else if (isdigit(*fen)) f += *fen - '0';
        else {
            int s = isupper(*fen) ? WHITE : BLACK, c = tolower(*fen);
            set_piece(pos, s, (c=='p')?PAWN:(c=='n')?KNIGHT:(c=='b')?BISHOP:(c=='r')?ROOK:(c=='q')?QUEEN:KING, r * 8 + f++);
        }
        fen++;
    }
    if (*fen) fen++;
    pos->side = (*fen == 'b') ? BLACK : WHITE;
    if (*fen) fen++;
    if (*fen) fen++;
    pos->castling = 0;
    while (*fen && *fen != ' ') {
        if (*fen == 'K') pos->castling |= CASTLE_WK;
        if (*fen == 'Q') pos->castling |= CASTLE_WQ;
        if (*fen == 'k') pos->castling |= CASTLE_BK;
        if (*fen == 'q') pos->castling |= CASTLE_BQ;
        fen++;
    }
    pos->ep = SQ_NONE;
    if (*fen) fen++;
    if (*fen != '-' && *fen >= 'a' && *fen <= 'h') pos->ep = (fen[1] - '1') * 8 + (fen[0] - 'a');
    pos->hash = generate_hash(pos);
    if (nnue.loaded) nnue_refresh(pos);
}

static Move parse_move(const Position *pos, const char *s) {
    if (!s || strlen(s) < 4) return 0;
    int fr = (s[0] - 'a') + (s[1] - '1') * 8, to = (s[2] - 'a') + (s[3] - '1') * 8;
    int pfl = (strlen(s) >= 5) ? ((tolower(s[4]) == 'n') ? FLAG_PROMO_N : (tolower(s[4]) == 'b') ? FLAG_PROMO_B : (tolower(s[4]) == 'r') ? FLAG_PROMO_R : FLAG_PROMO_Q) : 0;
    MoveList list; generate_moves(pos, &list, 0);
    for (int i = 0; i < list.count; i++) {
        Move m = list.moves[i];
        if (move_from(m) == fr && move_to(m) == to && (pfl ? (move_flag(m) == pfl) : (move_flag(m) < FLAG_PROMO_N))) return m;
    }
    return 0;
}

static void parse_position(Position *pos, char *line) {
    line += 8; while (*line == ' ') line++;
    if (!strncmp(line, "startpos", 8)) { parse_fen(pos, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); line += 8; }
    else if (!strncmp(line, "fen", 3)) { line += 3; while (*line == ' ') line++; parse_fen(pos, line); }
    char *mstr = strstr(line, "moves");
    pos_history[0] = pos->hash; pos_history_count = 1;
    if (mstr) for (char *tok = strtok(mstr + 5, " \t\r\n"); tok; tok = strtok(NULL, " \t\r\n")) {
        Move m = parse_move(pos, tok);
        if (m) { UndoInfo u; make_move(pos, m, &u); pos_history[pos_history_count++] = pos->hash; }
    }
}

static void parse_go(Position *pos, char *line) {
    int64_t wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0, hard_limit = 0, soft_limit = 0;
    int movestogo = 0, depth = 64; char *p;
    if ((p = strstr(line, "wtime"))) wtime = atoll(p + 6);
    if ((p = strstr(line, "btime"))) btime = atoll(p + 6);
    if ((p = strstr(line, "winc")))  winc  = atoll(p + 5);
    if ((p = strstr(line, "binc")))  binc  = atoll(p + 5);
    if ((p = strstr(line, "movestogo"))) movestogo = atoi(p + 10);
    if ((p = strstr(line, "movetime")))  movetime = atoll(p + 9);
    if ((p = strstr(line, "depth")))     depth = atoi(p + 6);
    if (movetime > 0) hard_limit = soft_limit = movetime;
    else if ((pos->side == WHITE && wtime > 0) || (pos->side == BLACK && btime > 0)) {
        int64_t my_time = (pos->side == WHITE) ? wtime : btime, my_inc = (pos->side == WHITE) ? winc : binc;
        int64_t base = (movestogo > 0 ? my_time / (movestogo + 1) : my_time / 20) + my_inc * 3 / 4;
        hard_limit = base * 25 / 10;
        if (hard_limit > my_time - 30) hard_limit = my_time - 30;
        if (hard_limit < 10) hard_limit = 10;
        soft_limit = base * 6 / 10;
        if (soft_limit > hard_limit) soft_limit = hard_limit;
        if (soft_limit < 5) soft_limit = 5;
    }
    search_root(pos, depth, hard_limit, soft_limit);
}

u64 perft(Position *pos, int depth) {
    if (depth <= 0) return 1;
    MoveList list; generate_moves(pos, &list, 0);
    u64 nodes = 0; UndoInfo undo;
    for (int i = 0; i < list.count; i++) {
        make_move(pos, list.moves[i], &undo);
        if (!in_check(pos, pos->side ^ 1)) nodes += perft(pos, depth - 1);
        undo_move(pos, list.moves[i], &undo);
    }
    return nodes;
}

void run_bench(void) {
    static const struct { const char *f; int d; u64 n; } p[6] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594}
    };
    const char *s[6] = { p[0].f, "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5", "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8", "r1b1k2r/ppppqppp/2n5/8/1bPP4/2N5/PP2BPPP/R1BQK2R w KQkq - 0 9", p[2].f, p[1].f };
    Position pos; u64 tot = 0; clock_t st = clock();
    printf("Running Perft Test Suite...\n");
    for (int i = 0; i < 6; i++) {
        parse_fen(&pos, p[i].f); u64 n = perft(&pos, p[i].d); tot += n;
        printf("Pos %d (depth %d): %llu nodes %s\n", i + 1, p[i].d, (unsigned long long)n, n == p[i].n ? "[PASS]" : "[FAIL]");
    }
    double el = (double)(clock() - st) / CLOCKS_PER_SEC;
    printf("Total Nodes: %llu, Time: %.3fs, NPS: %.2f MNPS\n\nSearch Benchmark (Pure Negamax Depth 4):\n", (unsigned long long)tot, el, tot / el / 1e6);
    for (int i = 0; i < 6; i++) {
        parse_fen(&pos, s[i]); tt_clear();
        printf("Position %d:\n", i + 1); search_root(&pos, 4, 0, 0); printf("\n");
    }
}

void uci_loop(int argc, char **argv) {
    Position pos; parse_fen(&pos, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if (argc > 1 && !strcmp(argv[1], "bench")) { run_bench(); return; }
    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        if (!strncmp(line, "uci", 3) && (line[3] == '\n' || line[3] == '\r' || line[3] == ' ' || line[3] == '\0')) {
            printf("id name Chal " CHAL_VERSION "\nid author Naman Thanki\noption name Hash type spin default 16 min 1 max 1024\nuciok\n");
            fflush(stdout);
        } else if (!strncmp(line, "isready", 7)) { printf("readyok\n"); fflush(stdout); }
        else if (!strncmp(line, "ucinewgame", 10)) { tt_clear(); memset(history, 0, sizeof(history)); pos_history_count = 0; parse_fen(&pos, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }
        else if (!strncmp(line, "position", 8)) parse_position(&pos, line);
        else if (!strncmp(line, "go", 2)) parse_go(&pos, line);
        else if (!strncmp(line, "setoption name Hash value", 25)) { int mb = atoi(line + 25); if (mb >= 1 && mb <= 1024) tt_allocate(mb); }
        else if (!strncmp(line, "eval", 4)) { printf("NNUE Eval: %+d cp\n", nnue_evaluate(&pos)); fflush(stdout); }
        else if (!strncmp(line, "perft", 5)) {
            int d = atoi(line + 6); if (d < 1) d = 5;
            clock_t start = clock(); u64 nodes = perft(&pos, d);
            double el = (double)(clock() - start) / CLOCKS_PER_SEC;
            printf("Perft(%d) = %llu nodes in %.3fs (%.2f MNPS)\n", d, (unsigned long long)nodes, el, (double)nodes / el / 1e6);
            fflush(stdout);
        } else if (!strncmp(line, "quit", 4)) break;
    }
}

int main(int argc, char **argv) {
    setvbuf(stdin, NULL, _IONBF, 0); setvbuf(stdout, NULL, _IONBF, 0);
    init_attacks(); init_zobrist(); init_lmr();
    nnue_init("src/net.nnue");
    tt_allocate(16); memset(history, 0, sizeof(history));
    uci_loop(argc, argv);
    return 0;
}
