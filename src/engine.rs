use std::{i32, time::Duration};

use crate::{
    board::{Bitboard, Board, Colour, Move, Pieces, State},
    piece_square::*,
};
use std::time::Instant;

pub const TABLE_SIZE: usize = 1_usize << 23;
const MIN: i32 = -500000;
const MAX: i32 = 500000;
const ASPIRATION_WINDOW: i32 = 25;
const MAX_DEPTH: usize = 64;
const MAX_HISTORY: u32 = 325;
const KILLER_MOVES: u32 = 200;
const FUTILITY_MARGIN: [i32; 3] = [0, 200, 300]; 
const R: u32 = 2;

pub struct Engine {
    depth: u32,
    endgame_depth: u32,
    transposition_table: Box<[TTEntry]>,
    history_table: [[[u32;64];64];2],
    killer_moves: [[Option<Move>; 2];MAX_DEPTH],
}

impl Engine {
    pub fn new(depth: u32, endgame_depth: u32) -> Engine {
        Engine {
            depth,
            endgame_depth,
            transposition_table: vec![TTEntry::default(); TABLE_SIZE].into_boxed_slice(),
            history_table: [[[0;64];64];2],
            killer_moves: [[None; 2];MAX_DEPTH],
        }
    }

    pub fn set_depth(&mut self, depth: u32) {
        self.depth = depth;
        self.endgame_depth = depth;
    }

    // pub fn render_pv(&self) -> String {
    //     let mut out = String::new();
    //     for m in &self.pv_table[0] {
    //         out = format!("{}{}{} ", out, Board::bit_to_algebraic(1 << m.from), Board::bit_to_algebraic(1 << m.to));
    //     }
    //     out
    // }

    pub fn search(&mut self, board: &mut Board, deadline: Option<Instant>, debug: bool) -> Move {
        let start = Instant::now();

        let mut moves = self.generate_moves(board, board.turn, None, false);
        let mut best_move = None;
        let mut best_score = MIN;
        let mut nodes_checked = 0;
        let mut nm_counter = 0;
        let mut quiescence_counter = 0;
        let mut redo = 0;
        let mut max_depth = 0;

        for depth in 1..=(if Engine::endgame(board.bitboards) {self.endgame_depth} else {self.depth}) {
            let mut window = ASPIRATION_WINDOW;
            let mut alpha = if depth > 1 { best_score - window } else { MIN };
            let mut beta = if depth > 1 { best_score + window } else { MAX };

            if let Some(prev) = best_move {
                if let Some(pos) = moves.iter().position(|&m| m == prev) {
                    let swap = moves.remove(pos);
                    moves.insert(0, swap);
                }
            }

            max_depth = depth;
            
            'm: loop {
                let mut a = alpha;
                let b = beta;
                best_score = MIN;

                for m in &moves {
                    board.make_move(*m, false).unwrap();
                    let move_evaluation = -self.negamax(board, depth - 1, -b, -a, &mut nodes_checked, &mut nm_counter, &mut quiescence_counter, deadline, 2);
                    board.unmake_move().unwrap();
                    if move_evaluation > best_score {
                        best_score = move_evaluation;
                        best_move = Some(*m);

                    }
                    a = std::cmp::max(a, best_score);
                    if let Some(dl) = deadline {
                        if Instant::now() >= dl {
                            break 'm;
                        }
                    }
                }

                if best_score <= alpha {
                    alpha -= window;
                    window *= 2;
                    redo += 1;
                    continue;
                }
                else if best_score >= beta {
                    beta += window;
                    window *= 2;
                    redo += 1;
                    continue;
                }
                break;
            }
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    break;
                }
            }
            self.decay_history();
            if debug {
                // println!("Depth {} complete in {:?}", depth, start.elapsed());
            }
        }
        // self.decay_history();
        if debug {
            let duration = start.elapsed();
            // println!("best score: {}", best_score);
            // println!("nodes checked: {nodes_checked}");
            // println!("time taken: {:?}", duration);
            // println!(
            //     "nps: {:?}",
            //     (nodes_checked as f64 / duration.as_secs_f64()).round() as u64
            // );
            // println!("aspiration rechecked: {}", redo);
            // println!("nmp: {}", nm_counter);
            // println!("quiescence: {}", quiescence_counter);
            let mut s = "cp";
            let mut e = Engine::evaluate(board, false, true, 0) * if board.turn == Colour::White { 1 } else { -1 };
            if best_score.abs() > 399000 {
                s = "mate";
                e = best_score.signum() * (400000-best_score.abs());
                println!("debug: {}", best_score);
            } 
            if !best_move.is_none() { println!("info depth {} score {} nodes {} nps {} time {} pv {}", max_depth, format!("{} {}", s, e), nodes_checked, (nodes_checked as f64 / duration.as_secs_f64()).round() as u64, duration.as_millis(), format!("{}{}", Board::bit_to_algebraic(1 << best_move.unwrap().from), Board::bit_to_algebraic(1 << best_move.unwrap().to))) };
            // println!("{}", board.export_fen());
        }
        // println!("Max Depth: {}", max_depth);

        best_move.unwrap_or_else(|| {
            moves[0]
        })
    }

    #[inline(always)]
    pub fn get_index(hash: u64) -> usize {
        (hash & (TABLE_SIZE - 1) as u64).try_into().unwrap()
    }

    #[inline(always)]
    pub fn decay_history(&mut self) {
        for from in 0..63 {
            for to in 0..63 {
                self.history_table[Colour::White as usize][from][to] /= 2;
                self.history_table[Colour::Black as usize][from][to] /= 2;
            }
        }
    }

    fn quiescence_search(&self, board: &mut Board, mut alpha: i32, beta: i32, ply: i32, counter: &mut u64, depth: Option<u32>) -> i32 {
        *counter += 1;
        let mut best_value = Engine::evaluate(board, board.state != State::Continue, true, ply);

        if best_value >= beta {
            return beta;
        }

        let mut new_depth = None;

        if best_value > alpha {
            alpha = best_value;
        }

        if let Some(d) = depth {
            new_depth = Some(d-1);
            if d == 0 {
                return alpha;
            }
        }

        let captures = self.generate_moves(board, board.turn, None, true);
        for m in captures {
            board.make_move(m, false).unwrap();
            let score = -self.quiescence_search(board, -beta, -alpha, ply+1, counter, new_depth);
            board.unmake_move().unwrap();

            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }

            if score > best_value {
                best_value = score;
            }
        }
        alpha

    }

    #[inline(always)]
    fn generate_moves(&self, board: &mut Board, colour: Colour, depth: Option<usize>, captures_only: bool) -> Vec<Move> {
        let mut moves = board.get_legal_moves(colour, Some(&self.transposition_table), captures_only);
        moves.sort_unstable_by(|a, b| {
            let mut b_cmp = b.score + self.history_table[colour as usize][b.from][b.to];
            let mut a_cmp = a.score + self.history_table[colour as usize][a.from][a.to];

            if let Some(d) = depth {
                a_cmp += Engine::killer_bonus(self.killer_moves[d], a);
                b_cmp += Engine::killer_bonus(self.killer_moves[d], b);
            }

            b_cmp.cmp(&a_cmp)
        });
        moves
    }

    #[inline(always)]
    fn killer_bonus(killers: [Option<Move>; 2], mv: &Move) -> u32 {
        killers.iter().filter(|&k| k == &Some(*mv)).count() as u32 * KILLER_MOVES
    }

    #[inline(always)]
    pub fn endgame(bb: [[Bitboard; 6]; 2]) -> bool {
        let mut total_score = 0;
        for colour in 0..=1 {
            for piece in 0..=5 {
                total_score += PIECE_SCORES[piece] * bb[colour][piece].count_ones();
            }
        }
        total_score < 1300 || (bb[0][0] | bb[1][0]).count_ones() < 5
    }

    #[inline(always)]
    fn is_repetition(board: &mut Board,) -> bool {
        board.hash_history
            .iter()
            .filter(|&&h| h == board.hash)
            .count()
            >= 2
    }


    fn negamax(
        &mut self,
        board: &mut Board,
        depth: u32,
        mut alpha: i32,
        beta: i32,
        counter: &mut u64,
        nm_counter: &mut u64,
        quiescence_counter: &mut u64,
        deadline: Option<Instant>,
        ply: i32
    ) -> i32 {
        let original_alpha = alpha;
        let tt_index = Engine::get_index(board.hash);
        let mut replace_tt = true;
        let entry = &self.transposition_table[tt_index];

        if entry.valid {
            replace_tt = (entry.depth as u32) < depth || entry.hash != board.hash;
            if entry.hash == board.hash && entry.depth as u32 >= depth {
                match entry.flag {
                    Flag::Exact => return entry.value,
                    Flag::Lower if entry.value >= beta => return entry.value,
                    Flag::Upper if entry.value <= alpha => return entry.value,
                    _ => {}
                };
            }
        }

        if depth == 0 || board.state != State::Continue {
            *counter += 1;
            return self.quiescence_search(board, alpha, beta, ply, quiescence_counter, None);
        };

        let moves = self.generate_moves(board, board.turn, Some(depth as usize), false);

        if moves.len() == 0 {
            *counter += 1;
            return Engine::evaluate(board, moves.len() == 0, false, ply);
        };

        let mut best_score = MIN;
        let mut best_move = None;

        if depth >= R+1 && !board.is_check(board.turn) && !Engine::endgame(board.bitboards) {
            board.make_null_move();
            let score = -self.negamax(board, depth - R - 1, -beta, -beta + 1, counter, nm_counter, quiescence_counter, deadline, ply+1);
            board.unmake_null_move();

            if score >= beta {
                *nm_counter += 1;
                return beta;
            }
        }

        let static_eval = Engine::evaluate(board, board.state != State::Continue, true, ply);

        for i in 0..moves.len() {
            let m = moves[i];

            if depth <= 2 
                && !board.is_check(board.turn) 
                && m.capture.is_none()
                && m.promotion.is_none()
                && static_eval + FUTILITY_MARGIN[depth as usize] <= alpha
            {
                continue;
            }

            let mut reduction = 0;
            board.make_move(m, false).unwrap();

            if depth >= 3 && m.capture.is_none() && m.promotion.is_none() && m.score == 0 && !Engine::endgame(board.bitboards) && !board.is_check(board.turn) && !board.is_check(!board.turn) {
                reduction = 2;
            }

            let mut eval = -self.negamax(board, depth - 1 - reduction, -beta, -alpha, counter, nm_counter, quiescence_counter, deadline, ply+1);

            if reduction > 0 && eval > alpha {
                eval = -self.negamax(board, depth - 1, -beta, -alpha, counter, nm_counter, quiescence_counter, deadline, ply+1);
            }

            if eval > best_score {
                best_score = eval;
                best_move = Some(m);
            }

            board.unmake_move().unwrap();
            alpha = std::cmp::max(alpha, best_score);

            if alpha >= beta {
                if m.capture.is_none() {
                    let ht = &mut self.history_table[board.turn as usize][m.from][m.to];
                    *ht += depth * depth;
                    if *ht > MAX_HISTORY {
                        *ht /= 2;
                    }

                    if !self.killer_moves[depth as usize].contains(&Some(m)) {
                        self.killer_moves[depth as usize][1] = self.killer_moves[depth as usize][0];
                        self.killer_moves[depth as usize][0] = Some(m);
                    }
                }
                break;
            }
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    return if best_score == MIN {
                        alpha
                    } else {
                        best_score
                    }
                }
            }
        }

        best_score = match best_score {
            MIN => alpha,
            _ => best_score
        };

        if replace_tt {
            let flag = if best_score <= original_alpha {
                Flag::Upper
            } else if best_score >= beta {
                Flag::Lower
            } else {
                Flag::Exact
            };

            self.transposition_table[tt_index] = TTEntry::new(
                board.hash,
                best_score,
                depth as u8,
                flag,
                best_move,
            );
        }
        // if Engine::is_repetition(board) {
        //     // best_score -= if board.turn == Colour::White { 50 } else { -50 };
        //     best_score -= 50;
        // }
        best_score
    }

    #[inline(always)]
    pub fn evaluate(board: &mut Board, terminal_state: bool, validate: bool, ply: i32) -> i32 {
        let mut score = Engine::square_score(board.bitboards) * if board.turn == Colour::White {1} else {-1};
        if terminal_state {
            score = match board.get_game_state(validate) {
                State::Checkmate(c) => {
                    if c == board.turn {
                        -400000+ply
                    } else { 
                        400000-ply
                    }
                },
                State::Draw | State::FiftyMoveRule | State::InsufficientMaterial | State::Stalemate | State::ThreeFoldRepetition => 0,
                State::Continue => score,
            };
        };

        // if board.is_check(board.turn) {
        //     score -= 200;
        // }
        // else if board.is_check(!board.turn) {
        //     score += 200;
        // }

        score
    }

    #[inline(always)]
    pub fn mvv_lva_score(attacker: Pieces, victim: Pieces) -> u32 {
        PIECE_SCORES[victim as usize] * 10 - (attacker as u32)
    }

    #[inline(always)]
    fn square_score(bitboards: [[Bitboard; 6]; 2]) -> i32 {
        let mut score = 0;
        let mut white_bb = bitboards[Colour::White as usize];
        let mut black_bb = bitboards[Colour::Black as usize];
        let endgame = Engine::endgame(bitboards);

        for (i, bb) in white_bb.iter_mut().enumerate() {
            while let Some(sq) = Board::pop_lsb(bb) {
                if i == 5 && endgame {
                    score += KING_ENDGAME[Engine::mirror(sq)]
                }
                else {
                    score += TABLE[i][Engine::mirror(sq)];
                }
                score += PIECE_SCORES[i] as i32;
            }
        }

        for (i, bb) in black_bb.iter_mut().enumerate() {
            while let Some(sq) = Board::pop_lsb(bb) {
                if i == 5 && endgame {
                    score -= KING_ENDGAME[sq]
                }
                else {
                    score -= TABLE[i][sq];
                }
                score -= PIECE_SCORES[i] as i32;
            }
        }

        score
    }

    #[inline(always)]
    fn mirror(i: usize) -> usize {
        i ^ 56
    }

    #[inline(always)]
    fn score(piece: Pieces) -> u32 {
        PIECE_SCORES[piece as usize]
    }
}

#[derive(Clone)]
pub struct ZobristHashing {
    pub pieces: [[[u64; 64]; 6]; 2],
    pub black_to_move: u64,
    pub castling_rights: [u64; 4], // K Q k q
    pub en_passant: [u64; 8],
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

        let mut castling_rights = [0; 4];
        let mut en_passant = [0; 8];

        castling_rights.fill_with(|| random.next_u64());
        en_passant.fill_with(|| random.next_u64());

        let black_to_move = random.next_u64();

        ZobristHashing {
            pieces,
            black_to_move,
            castling_rights,
            en_passant,
        }
    }
}

pub struct SplitMix64 {
    state: u64,
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
    Upper,
}

#[derive(Clone)]
pub struct TTEntry {
    pub hash: u64,
    pub value: i32,
    pub depth: u8,
    pub flag: Flag,
    pub best_move: Option<Move>,
    pub valid: bool,
}

impl TTEntry {
    pub fn new(hash: u64, value: i32, depth: u8, flag: Flag, best_move: Option<Move>) -> TTEntry {
        TTEntry {
            hash,
            value,
            depth,
            flag,
            best_move,
            valid: true
        }
    }
}

impl Default for TTEntry {
    fn default() -> Self {
        Self { hash: Default::default(), value: Default::default(), depth: Default::default(), flag: Flag::Exact, best_move: Default::default(), valid: false }
    }
}
