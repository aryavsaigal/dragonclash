use std::{i32};

use crate::board::{Bitboard, Board, Colour, Move, Pieces, State};
use std::time::Instant;

const TABLE_SIZE: usize = 1_usize << 22;

pub struct Engine {
    depth: u32,
    transposition_table: Box<[Option<TTEntry>]>,
}

impl Engine {
    pub fn new(depth: u32) -> Engine {
        Engine {
            depth,
            transposition_table: vec![None; TABLE_SIZE].into_boxed_slice()
        }
    }

    pub fn search(&mut self, board: &mut Board) -> Move {
        let start = Instant::now();

        let moves = board.get_legal_moves(board.turn);
        let mut best_move = None;
        let mut best_score = i32::MIN;

        let mut nodes_checked = 0;

        for m in moves {
            board.make_move(m, false).unwrap();
            let move_evaluation = -self.negamax(board, self.depth - 1, i32::MIN, i32::MAX, &mut nodes_checked);
            board.unmake_move().unwrap();
            if move_evaluation > best_score {
                best_score = move_evaluation;
                best_move = Some(m);
            }
        }
        let duration = start.elapsed();
        println!("nodes checkec: {nodes_checked}");
        println!("time taken: {:?}", duration);
        println!("nps: {:?}", (nodes_checked as f64 / duration.as_secs_f64()).round() as u64);

        best_move.unwrap()
    }

    #[inline(always)]
    fn get_index(hash: u64) -> usize {
        (hash & (TABLE_SIZE - 1) as u64).try_into().unwrap()
    }

    fn negamax(&mut self, board: &mut Board, depth: u32, mut alpha: i32, beta: i32, counter: &mut u64) -> i32 {
        let original_alpha = alpha;
        let tt_index = Engine::get_index(board.hash);

        if let Some(entry) = self.transposition_table[tt_index] {
            if entry.hash == board.hash && entry.depth as u32 >= depth {
                match entry.flag {
                    Flag::Exact => return entry.value,
                    Flag::Lower if entry.value >= beta => return entry.value,
                    Flag::Upper if entry.value <= alpha => return entry.value,
                    _ => {}
                };
            }
        }

        let moves = board.get_legal_moves(board.turn);

        if depth == 0 || moves.len() == 0 {
            let mut position_score = Engine::material_score(&board.bitboards, board.turn);
            if moves.len() == 0 {
                position_score += match board.get_game_state(false) {
                    State::Checkmate(c) => if c == board.turn { -1000 } else { 1000 },
                    _ => 0
                };
            };

            *counter += 1;
            return position_score;
        };

        let mut best_score = i32::MIN; 

        for m in moves {
            board.make_move(m, false).unwrap();
            best_score = std::cmp::max(best_score, -self.negamax(board,depth - 1, -beta, -alpha, counter));
            board.unmake_move().unwrap();
            alpha = std::cmp::max(alpha, best_score);

            if alpha >= beta {
                break;
            }
        }

        let flag = if best_score <= original_alpha {
            Flag::Upper
        } else if best_score >= beta {
            Flag::Lower
        } else {
            Flag::Exact
        };

        self.transposition_table[tt_index] = Some(TTEntry::new(board.hash, best_score, depth as u8, flag, None));

        best_score
    }

    #[inline(always)]
    fn material_score(bitboards: &[[Bitboard; 6];2], c: Colour) -> i32 {
        let mut score: i32 = 0;
        for (colour, bits) in bitboards.iter().enumerate() {
            for (piece, board) in bits.iter().enumerate() {
                score += (board.count_ones() * Engine::score(Pieces::from_num(piece).unwrap())) as i32 * if c as usize == colour { 1 } else { -1 };
            }
        }
        score
    }

    #[inline(always)]
    fn score(piece: Pieces) -> u32 {
        match piece {
            Pieces::King => 200,
            Pieces::Queen => 9,
            Pieces::Rook => 5,
            Pieces::Bishop => 3,
            Pieces::Knight => 3,
            Pieces::Pawn => 1,
        }
    }
}

#[derive(Clone)]
pub struct ZobristHashing {
    pub pieces: [[[u64; 64]; 6]; 2],
    pub black_to_move: u64,
    pub castling_rights: [u64; 4], // K Q k q
    pub en_passant: [u64; 8]
}

impl ZobristHashing {
    pub fn new(random: &mut SplitMix64) -> ZobristHashing {
        let mut pieces = [[[0u64; 64]; 6]; 2];
        for c in 0..=1 {
            for piece in 0..=5 {
                for sq in 0..=63 {
                    pieces[c][piece][sq] = random.next_u64();
                }
            }
        }

        let mut castling_rights = [0;4];
        let mut en_passant = [0;8];

        castling_rights.fill_with(|| random.next_u64());
        en_passant.fill_with(|| random.next_u64());

        let black_to_move = random.next_u64();

        ZobristHashing { pieces, black_to_move, castling_rights, en_passant }
    }
}

pub struct SplitMix64 {
    state: u64
}

impl SplitMix64 {
    pub fn new(state: u64) -> SplitMix64 {
        SplitMix64 { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;

        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[derive(Clone, Copy)]
pub enum Flag {
    Exact,
    Lower,
    Upper
}

#[derive(Clone, Copy)]
pub struct TTEntry {
    pub hash: u64,
    pub value: i32,
    pub depth: u8,
    pub flag: Flag,
    pub best_move: Option<Move>
}

impl TTEntry {
    pub fn new(hash: u64, value: i32, depth: u8, flag: Flag, best_move: Option<Move>) -> TTEntry {
        TTEntry {
            hash,
            value,
            depth,
            flag,
            best_move,
        }
    }
}

