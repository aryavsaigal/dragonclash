use std::ops::Not;

pub type Bitboard = u64;

#[derive(Clone)]
pub struct Board {
    pub bitboards: [[Bitboard; 6]; 2],
    turn: Colour,
    full_moves: usize,
    half_moves: usize,
    castling_rights: Castle,
    en_passant: Bitboard,
    pieces: [Option<(Colour, Pieces)>; 64],
    attack_tables: [[[Bitboard; 64]; 6]; 2],
}

fn file(n: usize) -> Bitboard {
    (0x0101010101010101 as Bitboard) << n
}

fn files<I: IntoIterator<Item = usize>>(c: I) -> Bitboard {
    let mut result = 0;
    for i in c {
        result |= file(i);
    }
    result
}


fn rank(n: usize) -> Bitboard {
    (0xFF as Bitboard) << (8 * n)
}

fn ranks<I: IntoIterator<Item = usize>>(c: I) -> Bitboard {
    let mut result = 0;
    for i in c {
        result |= rank(i);
    }
    result
}

impl Board {
    pub fn new() -> Board {
        Board {
            bitboards: [[0; 6]; 2],
            attack_tables: [[[0; 64]; 6]; 2],
            pieces: [None; 64],
            turn: Colour::White,
            full_moves: 1,
            half_moves: 0,
            castling_rights: Castle::new(),
            en_passant: 0,
        }
    }

    pub fn init(&mut self) {
        self.precompute_king_attacks();
        self.precompute_knight_attacks();
        self.precompute_pawn_attacks();
    }

    pub fn get_pseudo_legal_moves(&self, colour: Colour) -> Vec<Move> {
        let mut moves = Vec::new();
        for (i, mut b) in self.bitboards[colour as usize].iter().copied().enumerate() {
            if i == Pieces::King as usize {
                let from = Board::pop_lsb(&mut b).unwrap();
                let mut possible_moves = self.attack_tables[colour as usize][i][from]
                    | self.castling_rights.colour(colour);
                possible_moves &= !self.pieces(colour);

                while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                    moves.push(Move::new(
                        from,
                        to,
                        (colour, Pieces::King),
                        self.pieces[to],
                        None,
                        false,
                    ));
                }
            } else if i == Pieces::Knight as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let mut possible_moves = self.attack_tables[colour as usize][i][from];
                    possible_moves &= !self.pieces(colour);
                    while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                        moves.push(Move::new(
                            from,
                            to,
                            (colour, Pieces::Knight),
                            self.pieces[to],
                            None,
                            false,
                        ));
                    }
                }
            } else if i == Pieces::Pawn as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let mut possible_moves = self.attack_tables[colour as usize][i][from];

                    possible_moves &= self.pieces(!colour) | self.en_passant;
                    possible_moves |= self.pawn_moves((1 as Bitboard) << from, colour);
 
                    while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                        if to / 8 == 7 && colour == Colour::White
                            || to / 8 == 0 && colour == Colour::Black
                        {
                            for promotion in
                                [Pieces::Queen, Pieces::Rook, Pieces::Bishop, Pieces::Knight]
                            {
                                moves.push(Move::new(
                                    from,
                                    to,
                                    (colour, Pieces::Pawn),
                                    self.pieces[to],
                                    Some(promotion),
                                    false,
                                ));
                            }
                        } else {
                            moves.push(Move::new(
                                from,
                                to,
                                (colour, Pieces::Pawn),
                                self.pieces[to],
                                None,
                                false,
                            ));
                        }
                    }
                }
            }
            else if i == Pieces::Rook as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let mut possible_moves = self.brute_rook_moves(from, colour);
                    while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                        moves.push(Move::new(
                            from,
                            to,
                            (colour, Pieces::Rook),
                            self.pieces[to],
                            None,
                            false,
                        ));
                    }
                }
            }
            else if i == Pieces::Bishop as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let mut possible_moves = self.brute_bishop_moves(from, colour);
                    while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                        moves.push(Move::new(
                            from,
                            to,
                            (colour, Pieces::Bishop),
                            self.pieces[to],
                            None,
                            false,
                        ));
                    }
                }
            }
            else if i == Pieces::Queen as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let mut possible_moves = self.brute_rook_moves(from, colour) | self.brute_bishop_moves(from, colour);
                    while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                        moves.push(Move::new(
                            from,
                            to,
                            (colour, Pieces::Queen),
                            self.pieces[to],
                            None,
                            false,
                        ));
                    }
                }
            }
        }

        moves
    }

    fn precompute_king_attacks(&mut self) {
        for square in 0..=63 {
            let mut bit = 0 as Bitboard;
            Board::set_bit(&mut bit, square);
            self.attack_tables[Colour::White as usize][Pieces::King as usize][square] = (bit << 1
                & !file(0))
                | (bit >> 1 & !file(7))
                | (bit << 7 & !file(7))
                | bit << 8
                | (bit << 9 & !file(0))
                | (bit >> 9 & !file(7))
                | bit >> 8
                | (bit >> 7 & !file(0));
            self.attack_tables[Colour::Black as usize][Pieces::King as usize][square] =
                self.attack_tables[Colour::White as usize][Pieces::King as usize][square];
        }
    }

    fn precompute_knight_attacks(&mut self) {
        for square in 0..=63 {
            let mut bit = 0 as Bitboard;
            Board::set_bit(&mut bit, square);
            self.attack_tables[Colour::White as usize][Pieces::Knight as usize][square] = bit << 6
                & !(file(6) | file(7))
                | bit << 10 & !(file(0) | file(1))
                | bit >> 10 & !(file(6) | file(7))
                | bit >> 6 & !(file(0) | file(1))
                | bit << 15 & !file(7)
                | bit << 17 & !file(0)
                | bit >> 17 & !file(7)
                | bit >> 15 & !file(0);
            self.attack_tables[Colour::Black as usize][Pieces::Knight as usize][square] =
                self.attack_tables[Colour::White as usize][Pieces::Knight as usize][square];
        }
    }

    fn precompute_pawn_attacks(&mut self) {
        for square in 0..=63 {
            let mut bit = 0 as Bitboard;
            Board::set_bit(&mut bit, square);
            self.attack_tables[Colour::White as usize][Pieces::Pawn as usize][square] =
                bit << 7 & !(file(7)) | bit << 9 & !(file(0));
            self.attack_tables[Colour::Black as usize][Pieces::Pawn as usize][square] =
                bit >> 9 & !(file(7)) | bit >> 7 & !(file(0));
        }
    }

    fn pawn_moves(&self, bit: Bitboard, colour: Colour) -> Bitboard {
        let mut moves = 0 as Bitboard;
        moves |= match colour {
            Colour::White => (bit << 8 | bit << 16 & rank(3)) & !self.all_pieces(),
            Colour::Black => (bit >> 8 | bit >> 16 & rank(4)) & !self.all_pieces(),
        };

        moves
    }

    fn brute_rook_moves(&self, sq: usize, colour: Colour) -> Bitboard {
        let mut attacks = 0 as Bitboard;
        let rook = (1_u64 << sq) as Bitboard; 
        let all_pieces = self.all_pieces();

        for c in 1..=7 {
            let possible_move = rook << 8 * c;
            if possible_move == 0 { break; }

            attacks |= possible_move;

            if possible_move & all_pieces != 0 { break; }
        }

        for c in 1..=7 {
            let possible_move = rook >> 8 * c;
            if possible_move == 0 { break; }

            attacks |= possible_move;

            if possible_move & all_pieces != 0 { break; }
        }

        for c in 1..=7 {
            let possible_move = rook << c;
            if possible_move & rank(sq / 8) == 0 { break; }

            attacks |= possible_move;

            if possible_move & all_pieces != 0 { break; }
        }

        for c in 1..=7 {
            let possible_move = rook >> c;
            if possible_move & rank(sq / 8) == 0 { break; }

            attacks |= possible_move;

            if possible_move & all_pieces != 0 { break; }
        }
        

        attacks & !self.pieces(colour)
    }

    fn brute_bishop_moves(&self, sq: usize, colour: Colour) -> Bitboard {
        let mut attacks = 0 as Bitboard;
        let bishop = (1_u64 << sq) as Bitboard; 
        let all_pieces = self.all_pieces();

        for c in 1..=7 {
            let possible_move = bishop << 9 * c;
            if possible_move & files((sq%8)..=7) == 0 { break; }

            attacks |= possible_move;

            if possible_move & all_pieces != 0 { break; }
        }

        for c in 1..=7 {
            let possible_move = bishop >> 9 * c;
            if possible_move & files(0..=(sq%8)) == 0 { break; }

            attacks |= possible_move;

            if possible_move & all_pieces != 0 { break; }
        }

        for c in 1..=7 {
            let possible_move = bishop << 7 * c;
            if possible_move & files(0..=(sq%8)) == 0 { break; }

            attacks |= possible_move;

            if possible_move & all_pieces != 0 { break; }
        }

        for c in 1..=7 {
            let possible_move = bishop >> 7 * c ;
            if possible_move & files((sq%8)..=7) == 0 { break; }

            attacks |= possible_move;

            if possible_move & all_pieces != 0 { break; }
        }
        

        attacks & !self.pieces(colour)
    }

    pub fn is_check(&mut self, colour: Colour) -> bool {
        let king_sq = self.bitboards[colour as usize][Pieces::King as usize].trailing_zeros() as usize;
        (self.attack_tables[!colour as usize][Pieces::Knight as usize][king_sq] & self.bitboards[!colour as usize][Pieces::Knight as usize]) | (self.attack_tables[colour as usize][Pieces::Pawn as usize][king_sq] & self.bitboards[!colour as usize][Pieces::Pawn as usize]) | (self.brute_rook_moves(king_sq, colour) & self.bitboards[!colour as usize][Pieces::Rook as usize]) | (self.brute_bishop_moves(king_sq, colour) & self.bitboards[!colour as usize][Pieces::Bishop as usize]) | ((self.brute_rook_moves(king_sq, colour) | self.brute_bishop_moves(king_sq, colour)) & self.bitboards[!colour as usize][Pieces::Queen as usize]) != 0
    }

    pub fn get_legal_moves(&mut self, colour: Colour) -> Vec<Move> {
        let pseudo_legal_moves = self.get_pseudo_legal_moves(colour);

        let legal_moves: Vec<Move> = pseudo_legal_moves.iter().filter(|m| {
            let mut board_clone = self.clone();
            board_clone.make_move(**m, false).unwrap();
            !board_clone.is_check(colour)

        }).cloned().collect();

        legal_moves
    }

    pub fn verbose_perft(&mut self, colour: Colour, depth: usize) {
        let moves = self.get_legal_moves(colour);
        let mut counter = 0;

        for m in moves {
            let mut board_clone = self.clone();
            let mut node_counter = 0;

            board_clone.make_move(m, false).unwrap();
            board_clone.perft(!colour, depth-1, &mut node_counter);

            println!("{}{}: {}", Board::bit_to_algebraic(1 << m.from), Board::bit_to_algebraic(1 << m.to), node_counter);
            counter += node_counter;
        }
        println!("Total: {}", counter);
    }

    pub fn perft(&mut self, colour: Colour, depth: usize, counter: &mut u64) {
        let moves = self.get_legal_moves(colour);

        if depth == 0 {
            *counter+=1;
            return;
        }

        for m in moves {
            let mut board_clone = self.clone();
            board_clone.make_move(m, false).unwrap();
            board_clone.perft(!colour, depth-1, counter);
        }
    }

    fn generate_blocker_permutation(mask: Bitboard) -> Vec<Bitboard> {
        let bits: Vec<usize> = (0..=63).filter(|&i| Board::get_bit(mask, i) != 0).collect();
        let count = 1 << bits.len();
        let mut occupancies = Vec::with_capacity(count);

        for index in 0..count {
            let mut occupancy = 0u64;
            for (j, &bit) in bits.iter().enumerate() {
                if (index >> j) & 1 != 0 {
                    occupancy |= 1 << bit;
                }
            }
            occupancies.push(occupancy);
        }

        occupancies
    }

    fn rook_mask(sq: usize) -> Bitboard {
        let r = sq / 8;
        let f = sq % 8;

        file(f) | rank(r)
    }

    fn get_bit_position(rank: usize, file: usize) -> usize {
        rank * 8 + file
    }

    fn all_pieces(&self) -> Bitboard {
        self.white_pieces() | self.black_pieces()
    }

    fn empty(&self) -> Bitboard {
        !self.all_pieces()
    }

    fn pieces(&self, colour: Colour) -> Bitboard {
        match colour {
            Colour::White => self.white_pieces(),
            Colour::Black => self.black_pieces(),
        }
    }

    fn white_pieces(&self) -> Bitboard {
        self.bitboards[Colour::White as usize]
            .iter()
            .copied()
            .reduce(|a, b| a | b)
            .unwrap()
    }

    fn black_pieces(&self) -> Bitboard {
        self.bitboards[Colour::Black as usize]
            .iter()
            .copied()
            .reduce(|a, b| a | b)
            .unwrap()
    }

    fn set_bit(bitboard: &mut Bitboard, i: usize) {
        *bitboard |= 1 << i;
    }

    fn get_bit(bitboard: Bitboard, i: usize) -> Bitboard {
        bitboard & 1 << i
    }

    fn clear_bit(bitboard: &mut Bitboard, i: usize) {
        *bitboard &= !(1 << i);
    }

    fn pop_lsb(bitboard: &mut Bitboard) -> Option<usize> {
        let i = bitboard.trailing_zeros() as usize;
        if *bitboard > 0 {
            *bitboard &= *bitboard - 1
        };
        if i == 64 { None } else { Some(i) }
    }

    pub fn algebraic_to_bit(m: String) -> Bitboard {
        let mut bit = 0;

        if m == "-" {
            return 0;
        }

        if m.len() != 2 {
            panic!("Invalid length");
        }

        let file = (m.chars().nth(0).unwrap() as u8 - b'a') as usize;
        let rank = m.chars().nth(1).unwrap().to_digit(10).unwrap() as usize;

        Board::set_bit(&mut bit, Board::get_bit_position(rank, file));

        bit
    }

    pub fn bit_to_algebraic(bit: Bitboard) -> String {
        let sq = bit.trailing_zeros();
        let rank = ((sq % 8) as u8 + b'a') as char;
        let file = sq / 8 + 1;
        format!("{}{}", rank, file)

    }

    pub fn load_fen(&mut self, fen: String) {
        let mut rank = 7;
        let mut file = 0;

        for (i, field) in fen.split_ascii_whitespace().enumerate() {
            match i {
                0 => {
                    for c in field.chars() {
                        if c == '/' {
                            rank -= 1;
                            file = 0;
                        } else if c.is_digit(10) {
                            file += c.to_digit(10).unwrap() as usize;
                        } else if c.is_ascii_alphabetic() {
                            let (colour, piece) = map_char(c);
                            Board::set_bit(
                                &mut self.bitboards[colour as usize][piece as usize],
                                Board::get_bit_position(rank, file),
                            );
                            self.pieces[Board::get_bit_position(rank, file)] =
                                Some((colour, piece));
                            file += 1;
                        }
                    }
                }
                1 => {
                    self.turn = match field {
                        "w" => Colour::White,
                        "b" => Colour::Black,
                        _ => panic!("Invalid turn field: {}", field),
                    };
                }
                2 => {
                    self.castling_rights.white_king = if field.contains("K") {
                        Castle::new().white_king
                    } else {
                        0
                    };
                    self.castling_rights.white_queen = if field.contains("Q") {
                        Castle::new().white_queen
                    } else {
                        0
                    };
                    self.castling_rights.black_king = if field.contains("k") {
                        Castle::new().black_king
                    } else {
                        0
                    };
                    self.castling_rights.black_queen = if field.contains("q") {
                        Castle::new().black_queen
                    } else {
                        0
                    };
                }
                3 => {
                    self.en_passant = Board::algebraic_to_bit(field.to_string());
                }
                4 => {
                    self.half_moves = field.parse::<usize>().unwrap();
                }
                5 => {
                    self.full_moves = field.parse::<usize>().unwrap();
                }
                _ => continue,
            }
        }
    }

    pub fn display(&mut self) {
        for rank in (0..=7).rev() {
            print!("\x1b[38;5;15m\x1b[48;5;236m{} \x1b[0m", 8 - rank);
            for file in 0..=7 {
                let symbol = match self.pieces[Board::get_bit_position(rank, file)] {
                    None => " ".to_string(),
                    Some((colour, piece)) => colour.symbol().to_string() + piece.symbol(),
                };

                print!(
                    "{}{} \x1b[0m",
                    if (file % 2 == 0) ^ (rank % 2 == 0) {
                        "\x1b[48;5;250m"
                    } else {
                        "\x1b[48;5;240m"
                    },
                    symbol
                );
            }
            println!();
        }
        print!("\x1b[38;5;15m\x1b[48;5;236m ");
        for c in 'a'..='h' {
            print!("\x1b[38;5;15m\x1b[48;5;236m {}\x1b[0m", c);
        }
        println!();
    }

    pub fn display_bitboard(bitboard: Bitboard) {
        for rank in (0..=7).rev() {
            print!("\x1b[38;5;15m\x1b[48;5;236m{} \x1b[0m", 8 - rank);
            for file in 0..=7 {
                let symbol = if Board::get_bit(bitboard, rank * 8 + file) != 0 {
                    "1"
                } else {
                    "0"
                };
                print!(
                    "{}{} \x1b[0m",
                    if (file % 2 == 0) ^ (rank % 2 == 0) {
                        "\x1b[48;5;250m"
                    } else {
                        "\x1b[48;5;240m"
                    },
                    symbol
                );
            }
            println!();
        }
        print!("\x1b[38;5;15m\x1b[48;5;236m ");
        for c in 'a'..='h' {
            print!("\x1b[38;5;15m\x1b[48;5;236m {}\x1b[0m", c);
        }
        println!();
    }

    pub fn move_parser(&mut self, m: String) -> Result<Move, String> {
        if m.len() < 4 || m.len() > 5 {
            return Err("Invalid length".to_string());
        }

        let (i, f) = m.split_at(2);

        let from = (i.chars().nth(0).unwrap() as u8 - b'a') as usize
            + (i.chars().nth(1).unwrap().to_digit(10).unwrap() as usize - 1) * 8;
        let to = (f.chars().nth(0).unwrap() as u8 - b'a') as usize
            + (f.chars().nth(1).unwrap().to_digit(10).unwrap() as usize - 1) * 8;

        if from > 63 || to > 63 {
            return Err("Invalid move".to_string());
        }
        let piece = self.pieces[from].ok_or("Invalid move, no initial piece".to_string())?;
        let capture = self.pieces[to];
        let promotion = if f.len() == 3 {
            Some(match f.chars().nth(2).unwrap() {
                'q' => Pieces::Queen,
                'r' => Pieces::Rook,
                'b' => Pieces::Bishop,
                'n' => Pieces::Knight,
                _ => return Err("Invalid promotion".to_string()),
            })
        } else {
            None
        };
        // if piece.1 == Pieces::Rook { Board::display_bitboard(self.brute_rook_moves(from, piece.0)); }
        Ok(Move::new(from, to, piece, capture, promotion, false))
    }

    pub fn make_move(&mut self, m: Move, validate: bool) -> Result<(), String> {
        if m.piece.0 != self.turn {
            return Err(format!("Invalid turn, it is {:?}'s move", self.turn));
        }

        if validate && !self.get_legal_moves(m.piece.0).contains(&m) {
            return Err("Invalid move".to_string());
        }

        self.pieces[m.from] = None;
        self.pieces[m.to] = Some(m.piece);

        self.half_moves += 1;

        Board::clear_bit(
            &mut self.bitboards[m.piece.0 as usize][m.piece.1 as usize],
            m.from,
        );
        Board::set_bit(
            &mut self.bitboards[m.piece.0 as usize][m.piece.1 as usize],
            m.to,
        );

        if let Some(promotion) = m.promotion {
            self.pieces[m.to] = Some((m.piece.0, promotion));
            Board::clear_bit(
                &mut self.bitboards[m.piece.0 as usize][m.piece.1 as usize],
                m.to,
            );
            Board::set_bit(
                &mut self.bitboards[m.piece.0 as usize][promotion as usize],
                m.to,
            );
        }

        if m.piece.1 == Pieces::King {
            if (m.from as isize - m.to as isize).abs() == 2 {
                match m.piece.0 {
                    Colour::White => {
                        if m.to == self.castling_rights.white_king.trailing_zeros() as usize {
                            self.pieces[7] = None;
                            self.pieces[5] = Some((Colour::White, Pieces::Rook));
                            Board::clear_bit(
                                &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                                7,
                            );
                            Board::set_bit(
                                &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                                5,
                            );
                        } else if m.to == self.castling_rights.white_queen.trailing_zeros() as usize
                        {
                            self.pieces[0] = None;
                            self.pieces[2] = Some((Colour::White, Pieces::Rook));
                            Board::clear_bit(
                                &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                                0,
                            );
                            Board::set_bit(
                                &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                                2,
                            );
                        }
                    }
                    Colour::Black => {
                        if m.to == self.castling_rights.black_king.trailing_zeros() as usize {
                            self.pieces[63] = None;
                            self.pieces[61] = Some((Colour::Black, Pieces::Rook));
                            Board::clear_bit(
                                &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                                63,
                            );
                            Board::set_bit(
                                &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                                61,
                            );
                        } else if m.to == self.castling_rights.black_queen.trailing_zeros() as usize
                        {
                            self.pieces[56] = None;
                            self.pieces[58] = Some((Colour::Black, Pieces::Rook));
                            Board::clear_bit(
                                &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                                56,
                            );
                            Board::set_bit(
                                &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                                58,
                            );
                        }
                    }
                }
            }
            self.castling_rights.void(m.piece.0);
        }

        if m.from == 0 || m.to == 0 {
            self.castling_rights.white_queen = 0;
        } else if m.from == 7 || m.to == 7 {
            self.castling_rights.white_king = 0;
        } else if m.from == 56 || m.to == 56 {
            self.castling_rights.black_queen = 0;
        } else if m.from == 63 || m.to == 63 {
            self.castling_rights.black_king = 0;
        }

        if let Some(piece) = m.capture {
            Board::clear_bit(
                &mut self.bitboards[piece.0 as usize][piece.1 as usize],
                m.to,
            );

            self.half_moves = 0;
        } else {
            if m.piece.1 == Pieces::Pawn {
                self.half_moves = 0;
                if (1 as Bitboard) << m.to == self.en_passant {
                    match m.piece.0 {
                        Colour::White => {
                            self.pieces[m.to - 8] = None;
                            Board::clear_bit(
                                &mut self.bitboards[Colour::Black as usize][Pieces::Pawn as usize],
                                m.to - 8,
                            );
                        }
                        Colour::Black => {
                            self.pieces[m.to + 8] = None;
                            Board::clear_bit(
                                &mut self.bitboards[Colour::White as usize][Pieces::Pawn as usize],
                                m.to + 8,
                            );
                        }
                    }
                }
            }

        }

        if (m.from as isize - m.to as isize).abs() == 16 && m.piece.1 == Pieces::Pawn {
            self.en_passant = (1 as Bitboard) << (m.from + m.to) / 2;
        } else {
            self.en_passant = 0;
        }

        if self.turn == Colour::Black {
            self.full_moves += 1;
        }

        self.turn = !self.turn;

        Ok(())
    }
}

fn map_char(c: char) -> (Colour, Pieces) {
    match c {
        'P' => (Colour::White, Pieces::Pawn),
        'N' => (Colour::White, Pieces::Knight),
        'B' => (Colour::White, Pieces::Bishop),
        'R' => (Colour::White, Pieces::Rook),
        'Q' => (Colour::White, Pieces::Queen),
        'K' => (Colour::White, Pieces::King),
        'p' => (Colour::Black, Pieces::Pawn),
        'n' => (Colour::Black, Pieces::Knight),
        'b' => (Colour::Black, Pieces::Bishop),
        'r' => (Colour::Black, Pieces::Rook),
        'q' => (Colour::Black, Pieces::Queen),
        'k' => (Colour::Black, Pieces::King),
        _ => panic!("Invalid FEN piece character: {}", c),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Move {
    from: usize,
    to: usize,
    piece: (Colour, Pieces),
    capture: Option<(Colour, Pieces)>,
    promotion: Option<Pieces>,
    en_passant: bool,
}

impl Move {
    pub fn new(
        from: usize,
        to: usize,
        piece: (Colour, Pieces),
        capture: Option<(Colour, Pieces)>,
        promotion: Option<Pieces>,
        en_passant: bool,
    ) -> Move {
        Move {
            from,
            to,
            promotion,
            en_passant,
            capture,
            piece,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Magic {
    magic_number: Bitboard,
    shift: usize,
    attacks: Vec<Bitboard>
}

impl Magic {
    fn new() -> Magic {
        Magic { magic_number: 0, shift: 0, attacks: Vec::new() }
    }

    fn get_index(&self, pieces: Bitboard) -> usize {
        (pieces.wrapping_mul(self.magic_number) >> self.shift) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pieces {
    Pawn = 0,
    Bishop,
    Knight,
    Rook,
    Queen,
    King,
}

impl Pieces {
    fn from_num(num: usize) -> Option<Pieces> {
        match num {
            0 => Some(Pieces::Pawn),
            1 => Some(Pieces::Bishop),
            2 => Some(Pieces::Knight),
            3 => Some(Pieces::Rook),
            4 => Some(Pieces::Queen),
            5 => Some(Pieces::King),
            _ => None,
        }
    }

    fn symbol(&self) -> &str {
        match self {
            Pieces::Rook => "♜",
            Pieces::Bishop => "♝",
            Pieces::Knight => "♞",
            Pieces::Queen => "♛",
            Pieces::King => "♚",
            Pieces::Pawn => "♟",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Colour {
    White = 0,
    Black = 1,
}

impl Colour {
    fn from_num(num: usize) -> Option<Colour> {
        match num {
            0 => Some(Colour::White),
            1 => Some(Colour::Black),
            _ => None,
        }
    }

    fn symbol(&self) -> &str {
        match self {
            Colour::White => "",
            Colour::Black => "\x1b[38;5;16m",
        }
    }
}

impl Not for Colour {
    type Output = Colour;

    fn not(self) -> Self::Output {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

#[derive(Clone)]
struct Castle {
    white_king: Bitboard,
    white_queen: Bitboard,
    black_king: Bitboard,
    black_queen: Bitboard,
}

impl Castle {
    fn new() -> Castle {
        Castle {
            white_king: (1 as Bitboard) << 6,
            white_queen: (1 as Bitboard) << 1,
            black_king: (1 as Bitboard) << 62,
            black_queen: (1 as Bitboard) << 57,
        }
    }

    fn colour(&self, colour: Colour) -> Bitboard {
        match colour {
            Colour::White => self.white_king | self.white_queen,
            Colour::Black => self.black_king | self.black_queen,
        }
    }

    fn void(&mut self, colour: Colour) {
        match colour {
            Colour::White => {
                self.white_king = 0;
                self.white_queen = 0;
            }
            Colour::Black => {
                self.black_king = 0;
                self.black_queen = 0;
            }
        };
    }
}
