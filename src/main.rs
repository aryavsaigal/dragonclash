pub(crate) mod board;

use std::io::Write;
use board::Board;

const DEFAULT: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
fn main() {
    let mut board = Board::new();
    board.init();
    board.load_fen("rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 1".to_string());
    let mut error_message = String::new();
    board.verbose_perft(board::Colour::White, 3);
    
    // loop {
    //     println!("\x1B[2J\x1B[1;1H");
    //     board.display();
    //     let mut m = String::new();
    //     print!("{error_message}> ");
    //     std::io::stdout().flush().unwrap();
    //     std::io::stdin().read_line(&mut m).unwrap();

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

 
    //     // board.make_move(board.get_pseudo_legal_moves(board::Colour::Black)[2]);
    // }
}
