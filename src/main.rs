pub(crate) mod board;
pub(crate) mod engine;
pub(crate) mod piece_square;

use board::Board;
use engine::Engine;
use std::io::Write;
use std::io::{self, BufRead};
use std::time::{Duration, Instant};

use crate::board::{Colour, Pieces, State};

const DEFAULT: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const SEED: u64 = 180280668239036416;
fn main() {
    let mut init_command= String::new();
    io::stdout().flush().unwrap();
    std::io::stdin().read_line(&mut init_command).unwrap();

    if init_command.trim() == "uci" {
        println!("id name DragonClash");
        println!("id author Aryav");
        println!("uciok");
        let mut board = Board::new(SEED);
        let mut engine = Engine::new(15, 15);
        let stdin = io::stdin();
        let mut init = false;

        loop {
            let mut cmd = String::new();
            stdin.read_line(&mut cmd).unwrap();
            cmd = cmd.trim().to_string();

            if cmd == "isready" {
                println!("readyok");
            }
            else if cmd == "ucinewgame" {
                board = Board::new(SEED);
                engine = Engine::new(15, 15);
                init = false;
            }
            else if cmd.starts_with("position") {
                let parts: Vec<&str> = cmd.split_ascii_whitespace().skip(1).collect();
                let mut parse_moves = false;
                let mut i = 0;
                if !init {
                    init = true;
                    while i < parts.len() {
                        let part = parts[i];
                        if part == "fen" {
                            let fen = parts[i+1..i+7].join(" ");
                            board.load_fen(fen);
                            i += 7;
                        }
                        else if part == "startpos" {
                            board.load_fen(DEFAULT.to_string());
                            i += 1;
                        }
                        else if part == "moves" {
                            parse_moves = true;
                            i += 1;
                        }
                        else if parse_moves {
                            let m = board.move_parser(part.trim().to_string()).unwrap();
                            board.make_move(m, false).unwrap();
                            i += 1;
                        }
                        else {
                            println!("{}", part);
                            i += 1;
                        }
                    }
                } else {
                    let m = board.move_parser(parts.last().unwrap().trim().to_string()).unwrap();
                    board.make_move(m, false).unwrap(); 
                    break;
                }
            }
            else if cmd.starts_with("go") {
                let parts: Vec<&str> = cmd.split_ascii_whitespace().skip(1).collect();
                let mut wtime = 0;
                let mut btime = 0;
                let mut winc = 0;
                let mut binc = 0;
                let mut depth: Option<u32> = None;
                let mut movetime = None;
                let mut moves_to_go = None;
                let mut i = 0;
                while i < parts.len() {
                    let part = parts[i];

                    if part == "wtime" {
                        wtime = parts[i+1].parse().unwrap();
                        i += 2;
                    }
                    else if part == "winc" {
                        winc = parts[i+1].parse().unwrap();
                        i += 2;
                    }
                    else if part == "btime" {
                        btime = parts[i+1].parse().unwrap();
                        i += 2;
                    }
                    else if part == "binc" {
                        binc = parts[i+1].parse().unwrap();
                        i += 2;
                    }
                    else if part == "depth" {
                        depth = Some(parts[i+1].parse().unwrap());
                        i += 2;
                    }
                    else if part == "movestogo" {
                        moves_to_go = Some(parts[i+1].parse().unwrap());
                        i += 2;
                    }
                    else if part == "movetime" {
                        movetime = Some(parts[i+1].parse().unwrap());
                        i += 2;
                    }
                    else {
                        i += 1;
                    }
                }

                if let Some(depth) = depth {
                    engine.set_depth(depth);
                    let m = engine.search(&mut board, None, true);
                    let p = match m.promotion {
                        Some(p) => match p {
                            Pieces::Bishop => "b",
                            Pieces::Knight => "n",
                            Pieces::Rook => "r",
                            Pieces::Queen => "q",
                            _ => ""
                        },
                        None => "",
                    };
                    board.make_move(m, false).unwrap();
                    println!("bestmove {}{}{}", Board::bit_to_algebraic(1u64 << m.from), Board::bit_to_algebraic(1u64 << m.to), p);
                }
                else {
                    let time_left = if board.turn == Colour::White { wtime } else { btime };
                    let increment = if board.turn == Colour::White { winc } else { binc };

                    let move_time = movetime.unwrap_or_else(|| time_left / moves_to_go.unwrap_or(20) + increment / 2);
                    println!("Time per move: {:?}", Duration::from_millis(move_time as u64));
                    let m = engine.search(&mut board, Some(Instant::now() + Duration::from_millis(move_time as u64)), true);
                    let p = match m.promotion {
                        Some(p) => match p {
                            Pieces::Bishop => "b",
                            Pieces::Knight => "n",
                            Pieces::Rook => "r",
                            Pieces::Queen => "q",
                            _ => ""
                        },
                        None => "",
                    };
                    board.make_move(m, false).unwrap();
                    println!("bestmove {}{}{}", Board::bit_to_algebraic(1u64 << m.from), Board::bit_to_algebraic(1u64 << m.to), p);
                }
            }
            else if cmd == "quit" {
                break;
            }
        }

    }
    else {

        println!("Initialising board...");
        let mut board = Board::new(SEED);
        let mut engine = Engine::new(15, 15);
        let mut user_colour = Colour::White;
        let mut ai = false;
        board.load_fen(DEFAULT.to_string());
        loop {
            let mut side = String::new();
            print!("Choose side (w/b/ai) > ");
            io::stdout().flush().unwrap();
            std::io::stdin().read_line(&mut side).unwrap();
    
            user_colour = match side.trim().to_lowercase().as_str() {
                "w" => Colour::White,
                "b" => Colour::Black,
                "ai" => {
                    ai = true;
                    break;
                },
                _ => {
                    println!("Invalid input: '{}'", side.trim());
                    continue;
                }
            };
            break;
        }
    
        let mut error_message = String::new();
        // board.verbose_perft(7);
    
        loop {
            println!("\x1B[2J\x1B[1;1H");
            board.display();
            println!("Score: {} | Endgame: {}", Engine::evaluate(&mut board, false, false, 0 as i32) * if board.turn == Colour::White { 1 } else { -1 }, Engine::endgame(board.bitboards));
            // board.display_from_bitboards();
            println!("{}{}. {:?} to play", error_message, board.full_moves, board.turn);
    
            if ai {
                let m = engine.search(&mut board, Some(Instant::now() + Duration::from_secs(4)), true);
                board.make_move(m, false).unwrap();
            }
            else {
                if board.turn == !user_colour {
                    let m = engine.search(&mut board, Some(Instant::now() + Duration::from_secs(5)), true);
                    board.make_move(m, false).unwrap();
                } else {
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
                    } else {
                        error_message = String::new();
                    }
                }
            }
    
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
}
