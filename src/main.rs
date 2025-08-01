pub(crate) mod board;
pub(crate) mod engine;
pub(crate) mod piece_square;

use board::Board;
use engine::Engine;
use std::io::Write;
use std::time::Instant;

use crate::board::{Colour, Pieces, State};

const DEFAULT: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const SEED: u64 = 180280668239036416;
fn main() {
    let mut board = Board::new(SEED);
    let mut engine = Engine::new(9);

    board.load_fen(DEFAULT.to_string());
    let mut error_message = String::new();
    // board.verbose_perft(7);

    loop {
        println!("\x1B[2J\x1B[1;1H");
        board.display();
        // board.display_from_bitboards();
        println!("{}{}. {:?} to play", error_message, board.full_moves, board.turn);
        let m = engine.search(&mut board);
        board.make_move(m, false).unwrap();
        
    //     if board.turn == Colour::Black {
    //         let m = engine.search(&mut board);
    //         board.make_move(m, false).unwrap();
    //     } else {
    //         let mut m = String::new();
    //         print!("{error_message}");
    //         std::io::stdout().flush().unwrap();
    //         std::io::stdin().read_line(&mut m).unwrap();

    //         if m.trim() == "undo" {
    //             if let Err(e) = board.unmake_move() {
    //                 error_message = format!("Error Occured: {e}\n");
    //             };
    //             continue;
    //         }

    //         let mov = match board.move_parser(m.trim().to_string()) {
    //             Ok(m) => m,
    //             Err(e) => {
    //                 error_message = format!("Error Occured: {e}\n");
    //                 continue;
    //             }
    //         };

    //         if let Err(e) = board.make_move(mov, true) {
    //             error_message = format!("Error Occured: {e}\n");
    //         } else {
    //             error_message = String::new();
    //         }

        match board.get_game_state(true) {
            State::Checkmate(c) => {
                error_message = format!("{:?} is checkmated", c);
                break;
            }
            State::Stalemate => {
                error_message = format!("Game under stalemate by {:?}", !board.turn);
                break;
            }
            State::Draw => {
                error_message = format!("Game drawed");
                break;
            }
            State::FiftyMoveRule => {
                error_message = "Draw by fifty-move rule".to_string();
                break;
            }
            State::InsufficientMaterial => {
                error_message = "Draw by insufficient material".to_string();
                break;
            },
            State::ThreeFoldRepetition => {
                error_message = "Draw by threefold repetition".to_string();
                break;
            }
            _ => {}
        }

    // }
    if board.is_check(board::Colour::White) {
        error_message = "White in Check\n".to_string();
    } else if board.is_check(board::Colour::Black) {
        error_message = "Black in Check\n".to_string();
    }
    }
    println!("\x1B[2J\x1B[1;1H");
    board.display();
    print!("{error_message}");
}
