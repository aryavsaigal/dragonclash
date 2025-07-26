use std::i32;

use crate::board::{Bitboard, Board, Colour, Move, Pieces, State};
use std::time::Instant;

pub struct Engine {
    depth: u32
}

impl Engine {
    pub fn new(depth: u32) -> Engine {
        Engine {
            depth
        }
    }

    pub fn search(&self, board: &mut Board) -> Move {
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

        best_move.unwrap()
    }

    fn negamax(&self, board: &mut Board, depth: u32, mut alpha: i32, beta: i32, counter: &mut u64) -> i32 {
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

        best_score
    }

    fn material_score(bitboards: &[[Bitboard; 6];2], c: Colour) -> i32 {
        let mut score: i32 = 0;
        for (colour, bits) in bitboards.iter().enumerate() {
            for (piece, board) in bits.iter().enumerate() {
                score += (board.count_ones() * Engine::score(Pieces::from_num(piece).unwrap())) as i32 * if c as usize == colour { 1 } else { -1 };
            }
        }
        score
    }

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