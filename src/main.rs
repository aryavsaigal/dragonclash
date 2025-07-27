pub(crate) mod board;
pub(crate) mod engine;

use std::io::Write;
use board::Board;
use engine::Engine;
use std::time::Instant;

use crate::board::{Colour, Pieces, State};


const DEFAULT: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const SEED: u64 = 180280668239036416;
fn main() {
    let mut board = Board::new(SEED);
    let mut engine = Engine::new(7);

    board.load_fen(DEFAULT.to_string());
    let mut error_message = String::new();
    println!("{:?}", board.castling_rights);
    
    loop {
        // if board.full_moves > 7 {
        //     println!("{}",board.full_moves);
        //     break;
        // }
        println!("\x1B[2J\x1B[1;1H");
        board.display();
        println!("hash: {}", board.hash);
        // let m = engine.search(&mut board);
        // board.make_move(m, false).unwrap();

        // if board.turn == Colour::Black {
        //     let m = engine.search(&mut board);
        //     board.make_move(m, false).unwrap();
        // }
        // else {
            let mut m = String::new();
            print!("{error_message}");
            std::io::stdout().flush().unwrap();
            std::io::stdin().read_line(&mut m).unwrap();
    
            if m.trim() == "undo" {
                if let Err(e) = board.unmake_move() {
                    error_message = format!("Error Occured: {e}\n");
                };
                continue;
            }
    
            let mov = match board.move_parser(m.trim().to_string()) {
                Ok(m) => m,
                Err(e) => {
                    error_message = format!("Error Occured: {e}\n");
                    continue;
                }
            };
    
    
            if let Err(e) = board.make_move(mov, true) {
                error_message = format!("Error Occured: {e}\n");
            }
            else {
                error_message = String::new();
            }

        match board.get_game_state(true) {
            State::Checkmate(c) => {
                error_message = format!("{:?} is checkmated", c);
                break;
            },
            State::Stalemate => {
                error_message = format!("Game under stalemate by {:?}", !board.turn);
                break;
            },
            State::Draw => {
                error_message = format!("Game drawed");
                break;
            }
            _ => {}
        }

        if board.is_check(board::Colour::White) {
            error_message = "White in Check\n".to_string();
        }
        else if board.is_check(board::Colour::Black) {
            error_message = "Black in Check\n".to_string();
        }

        // board.verbose_perft(board.turn, 1);
 
        // board.make_move(board.get_pseudo_legal_moves(board::Colour::Black)[2]);
    }

    println!("\x1B[2J\x1B[1;1H");
    board.display();
    print!("{error_message}");
}
