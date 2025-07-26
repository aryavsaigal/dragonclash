pub(crate) mod board;

use std::io::Write;
use board::Board;
use std::time::Instant;

use crate::board::{Colour, Pieces};


const DEFAULT: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
fn main() {
    let mut board = Board::new();
    board.init();
    board.load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1".to_string());
    let mut error_message = String::new();
    let start = Instant::now();
    board.verbose_perft(board.turn, 5);
    let duration = start.elapsed();
    println!("time taken: {:?}", duration);
    
    // loop {
    //     println!("\x1B[2J\x1B[1;1H");
    //     board.display();

    //     Board::display_bitboard(board.bitboards[1][Pieces::Pawn as usize]);

    //     let mut m = String::new();
    //     print!("{error_message}> ");
    //     std::io::stdout().flush().unwrap();
    //     std::io::stdin().read_line(&mut m).unwrap();

    //     if m.trim() == "undo" {
    //         if let Err(e) = board.unmake_move() {
    //             error_message = format!("Error Occured: {e}\n");
    //         };
    //         continue;
    //     }

    //     let mov = match board.move_parser(m.trim().to_string()) {
    //         Ok(m) => m,
    //         Err(e) => {
    //             error_message = format!("Error Occured: {e}\n");
    //             continue;
    //         }
    //     };


    //     if let Err(e) = board.make_move(mov, true) {
    //         error_message = format!("Error Occured: {e}\n");
    //     }
    //     else {
    //         error_message = String::new();
    //     }

    //     if board.is_check(board::Colour::White) {
    //         error_message = "White in Check\n".to_string();
    //     }
    //     else if board.is_check(board::Colour::Black) {
    //         error_message = "Black in Check\n".to_string();
    //     }

    //     // board.verbose_perft(board.turn, 1);
 
    //     // board.make_move(board.get_pseudo_legal_moves(board::Colour::Black)[2]);
    // }
}
