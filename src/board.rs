use std::ops::Not;

use crate::engine::{Engine, SplitMix64, TTEntry, ZobristHashing};

const MAX_MOVES: usize = 218;

pub type Bitboard = u64;

#[derive(Clone)]
pub struct Board {
    pub bitboards: [[Bitboard; 6]; 2],
    pub turn: Colour,
    pub full_moves: usize,
    pub half_moves: usize,
    pub hash: u64,
    pub zobrist_hash: ZobristHashing,
    pub castling_rights: Castle,
    pub state: State,
    en_passant: Bitboard,
    pieces: [Option<(Colour, Pieces)>; 64],
    attack_tables: [[[Bitboard; 64]; 6]; 2],
    move_history: Vec<(Move, Castle, Bitboard, usize)>,
    rook_magic_table: Vec<Magic>,
    bishop_magic_table: Vec<Magic>,
}

#[inline(always)]
fn file(n: usize) -> Bitboard {
    (0x0101010101010101 as Bitboard) << n
}

#[inline(always)]
fn files<I: IntoIterator<Item = usize>>(c: I) -> Bitboard {
    let mut result = 0;
    for i in c {
        result |= file(i);
    }
    result
}

#[inline(always)]
fn rank(n: usize) -> Bitboard {
    (0xFF as Bitboard) << (8 * n)
}

#[inline(always)]
fn ranks<I: IntoIterator<Item = usize>>(c: I) -> Bitboard {
    let mut result = 0;
    for i in c {
        result |= rank(i);
    }
    result
}

impl Board {
    pub fn new(seed: u64) -> Board {
        let mut b = Board {
            bitboards: [[0; 6]; 2],
            attack_tables: [[[0; 64]; 6]; 2],
            pieces: [None; 64],
            turn: Colour::White,
            full_moves: 1,
            half_moves: 0,
            state: State::Continue,
            castling_rights: Castle::new(),
            en_passant: 0,
            move_history: Vec::with_capacity(5989),
            rook_magic_table: Vec::with_capacity(64),
            bishop_magic_table: Vec::with_capacity(64),
            hash: 0,
            zobrist_hash: ZobristHashing::new(&mut SplitMix64::new(seed))
        };

        b.init();

        b
    }

    fn init(&mut self) {
        self.precompute_king_attacks();
        self.precompute_knight_attacks();
        self.precompute_pawn_attacks();
        self.generate_magic_table_rook();
        self.generate_magic_table_bishop();
    }

    pub fn get_pseudo_legal_moves(&mut self, colour: Colour) -> Vec<Move> {
        let mut moves = Vec::with_capacity(MAX_MOVES);
        let bitboards_copy: Vec<(usize, u64)> = self.bitboards[colour as usize]
            .iter()
            .copied()
            .enumerate()
            .collect();

        let my_pieces = self.pieces(colour);
        let opp_pieces = self.pieces(!colour);
        let all_pieces = my_pieces | opp_pieces;

        for (i, mut b) in bitboards_copy {
            if i == Pieces::King as usize {
                let from = Board::pop_lsb(&mut b).unwrap();
                let is_check = self.is_check(colour);

                let mut possible_moves = self.attack_tables[colour as usize][i][from];
                if !is_check && self.castling_rights.has_castle(colour) {
                    let king_castle_conflicts = match colour {
                        Colour::White => [self.bitboard_under_attack(WHITE_KING_CHECK, colour), self.bitboard_under_attack(WHITE_QUEEN_CHECK, colour)],
                        Colour::Black => [self.bitboard_under_attack(BLACK_KING_CHECK, colour), self.bitboard_under_attack(BLACK_QUEEN_CHECK, colour)]
                    };
                    possible_moves |= self.castling_rights.colour(
                        colour,
                        all_pieces,
                        king_castle_conflicts,
                    )
                };
                possible_moves &= !my_pieces;
                while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                    moves.push(Move::new(
                        from,
                        to,
                        (colour, Pieces::King),
                        self.pieces[to],
                        None,
                    ));
                }
            } else if i == Pieces::Knight as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let mut possible_moves = self.attack_tables[colour as usize][i][from];
                    possible_moves &= !my_pieces;
                    while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                        moves.push(Move::new(
                            from,
                            to,
                            (colour, Pieces::Knight),
                            self.pieces[to],
                            None,
                        ));
                    }
                }
            } else if i == Pieces::Pawn as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let mut possible_moves = self.attack_tables[colour as usize][i][from];

                    possible_moves &= opp_pieces | self.en_passant;
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
                                ));
                            }
                        } else {
                            moves.push(Move::new(
                                from,
                                to,
                                (colour, Pieces::Pawn),
                                self.pieces[to],
                                None,
                            ));
                        }
                    }
                }
            } else if i == Pieces::Rook as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let index = self.rook_magic_table[from].get_index(all_pieces & self.rook_magic_table[from].mask);
                    let mut possible_moves = self.rook_magic_table[from].attacks[index].unwrap() & !my_pieces;
                    while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                        moves.push(Move::new(
                            from,
                            to,
                            (colour, Pieces::Rook),
                            self.pieces[to],
                            None,
                        ));
                    }
                }
            } else if i == Pieces::Bishop as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let index = self.bishop_magic_table[from].get_index(all_pieces & self.bishop_magic_table[from].mask);
                    let mut possible_moves = self.bishop_magic_table[from].attacks[index].unwrap() & !my_pieces;
                    while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                        moves.push(Move::new(
                            from,
                            to,
                            (colour, Pieces::Bishop),
                            self.pieces[to],
                            None,
                        ));
                    }
                }
            } else if i == Pieces::Queen as usize {
                while let Some(from) = Board::pop_lsb(&mut b) {
                    let index_r = self.rook_magic_table[from].get_index(all_pieces & self.rook_magic_table[from].mask);
                    let index_b = self.bishop_magic_table[from].get_index(all_pieces & self.bishop_magic_table[from].mask);
                    let mut possible_moves = (self.rook_magic_table[from].attacks[index_r].unwrap() | self.bishop_magic_table[from].attacks[index_b].unwrap()) & !my_pieces;
                    while let Some(to) = Board::pop_lsb(&mut possible_moves) {
                        moves.push(Move::new(
                            from,
                            to,
                            (colour, Pieces::Queen),
                            self.pieces[to],
                            None,
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
            Colour::White => (bit << 8) & !self.all_pieces(),
            Colour::Black => (bit >> 8) & !self.all_pieces(),
        };

        moves |= match colour {
            Colour::White => (moves << 8 & rank(3)) & !self.all_pieces(),
            Colour::Black => (moves >> 8 & rank(4)) & !self.all_pieces(),
        };

        moves
    }

    fn magic_rook_moves(sq: usize, occ: Bitboard) -> Bitboard {
        let mut attacks = 0 as Bitboard;
        let rook = (1_u64 << sq) as Bitboard;

        for c in 1..=7 {
            let possible_move = rook << 8 * c;
            if possible_move == 0 {
                break;
            }

            attacks |= possible_move;

            if possible_move & occ != 0 {
                break;
            }
        }

        for c in 1..=7 {
            let possible_move = rook >> 8 * c;
            if possible_move == 0 {
                break;
            }

            attacks |= possible_move;

            if possible_move & occ != 0 {
                break;
            }
        }

        for c in 1..=7 {
            let possible_move = rook << c;
            if possible_move & rank(sq / 8) == 0 {
                break;
            }

            attacks |= possible_move;

            if possible_move & occ != 0 {
                break;
            }
        }

        for c in 1..=7 {
            let possible_move = rook >> c;
            if possible_move & rank(sq / 8) == 0 {
                break;
            }

            attacks |= possible_move;

            if possible_move & occ != 0 {
                break;
            }
        }

        attacks
    }

    fn magic_bishop_moves(sq: usize, occ: Bitboard) -> Bitboard {
        let mut attacks = 0 as Bitboard;
        let bishop = (1_u64 << sq) as Bitboard;

        for c in 1..=7 {
            let possible_move = bishop << 9 * c;
            if possible_move & files((sq % 8)..=7) == 0 {
                break;
            }

            attacks |= possible_move;

            if possible_move & occ != 0 {
                break;
            }
        }

        for c in 1..=7 {
            let possible_move = bishop >> 9 * c;
            if possible_move & files(0..=(sq % 8)) == 0 {
                break;
            }

            attacks |= possible_move;

            if possible_move & occ != 0 {
                break;
            }
        }

        for c in 1..=7 {
            let possible_move = bishop << 7 * c;
            if possible_move & files(0..=(sq % 8)) == 0 {
                break;
            }

            attacks |= possible_move;

            if possible_move & occ != 0 {
                break;
            }
        }

        for c in 1..=7 {
            let possible_move = bishop >> 7 * c;
            if possible_move & files((sq % 8)..=7) == 0 {
                break;
            }

            attacks |= possible_move;

            if possible_move & occ != 0 {
                break;
            }
        }

        attacks
    }

    pub fn bitboard_under_attack(&mut self, mut bit: Bitboard, colour: Colour) -> bool {
        while let Some(sq) = Board::pop_lsb(&mut bit) {
            let rook_index = self.rook_magic_table[sq].get_index(self.all_pieces() & self.rook_magic_table[sq].mask);
            let rook_moves = self.rook_magic_table[sq].attacks[rook_index].unwrap() & self.pieces(!colour);

            let bishop_index = self.bishop_magic_table[sq].get_index(self.all_pieces() & self.bishop_magic_table[sq].mask);
            let bishop_moves = self.bishop_magic_table[sq].attacks[bishop_index].unwrap() & self.pieces(!colour);

            if (self.attack_tables[!colour as usize][Pieces::Knight as usize][sq]
                & self.bitboards[!colour as usize][Pieces::Knight as usize])
                | (self.attack_tables[colour as usize][Pieces::Pawn as usize][sq]
                    & self.bitboards[!colour as usize][Pieces::Pawn as usize])
                | (rook_moves
                    & self.bitboards[!colour as usize][Pieces::Rook as usize])
                | (bishop_moves
                    & self.bitboards[!colour as usize][Pieces::Bishop as usize])
                | ((rook_moves | bishop_moves)
                    & self.bitboards[!colour as usize][Pieces::Queen as usize])
                != 0
            {
                return true;
            }
        }
        false
    }

    #[inline(always)]
    pub fn is_check(&mut self, colour: Colour) -> bool {
        let king_sq = self.bitboards[colour as usize][Pieces::King as usize].trailing_zeros() as usize;

        let rook_index = self.rook_magic_table[king_sq].get_index(self.all_pieces() & self.rook_magic_table[king_sq].mask);
        let rook_moves = self.rook_magic_table[king_sq].attacks[rook_index].unwrap() & self.pieces(!colour);

        let bishop_index = self.bishop_magic_table[king_sq].get_index(self.all_pieces() & self.bishop_magic_table[king_sq].mask);
        let bishop_moves = self.bishop_magic_table[king_sq].attacks[bishop_index].unwrap() & self.pieces(!colour);

        (self.attack_tables[!colour as usize][Pieces::Knight as usize][king_sq]
            & self.bitboards[!colour as usize][Pieces::Knight as usize])
            | (self.attack_tables[colour as usize][Pieces::Pawn as usize][king_sq]
                & self.bitboards[!colour as usize][Pieces::Pawn as usize])
            | (rook_moves
                & self.bitboards[!colour as usize][Pieces::Rook as usize])
            | (bishop_moves
                & self.bitboards[!colour as usize][Pieces::Bishop as usize])
            | ((rook_moves | bishop_moves)
                & self.bitboards[!colour as usize][Pieces::Queen as usize])
            != 0
    }

    pub fn get_legal_moves(&mut self, colour: Colour, ttable: Option<&Box<[Option<TTEntry>]>>) -> Vec<Move> {
        let mut pseudo_legal_moves = self.get_pseudo_legal_moves(colour);
        let mut legal_moves = Vec::with_capacity(MAX_MOVES);

        for m in &mut pseudo_legal_moves {
            self.make_move(*m, false).unwrap();
            if self.bitboards[!colour as usize][Pieces::King as usize] != 0 && !self.is_check(colour) {
                if let Some(capture) = m.capture {
                    m.score = Engine::mvv_lva_score(m.piece.1, capture.1);
                    if let Some(transposition_table) = ttable {
                        let index = Engine::get_index(self.hash);
                        if let Some(entry) = transposition_table[index] {
                            if entry.hash == self.hash {
                                m.score += 500000;
                            }
                        }
                    }
                }
                legal_moves.push(*m);
            }
            self.unmake_move().unwrap();
        }

        legal_moves
    }

    pub fn verbose_perft(&mut self, depth: usize) {
        let start = std::time::Instant::now();
        let colour = self.turn;
        let moves = self.get_legal_moves(colour, None);
        let mut counter = 0;

        for m in moves {
            let mut node_counter = 0;

            self.make_move(m, false).unwrap();
            self.perft(!colour, depth - 1, &mut node_counter);
            self.unmake_move().unwrap();

            println!(
                "{}{}: {}",
                Board::bit_to_algebraic(1 << m.from),
                Board::bit_to_algebraic(1 << m.to),
                node_counter
            );
            counter += node_counter;
        }
        let duration = start.elapsed();
        println!("Total: {}", counter);
        println!("time taken: {:?}", duration);
        println!("nps: {:?}", counter / (duration.as_secs_f64()).round() as u64);
    }

    // pub fn move_history_to_algebraic(&self) -> String {
    //     let mut h = String::new();
    //     for (i, history) in self.move_history.iter().enumerate() {
    //         h = format!("{}{}. {}{}\n", h, i+1, Board::bit_to_algebraic(1u64 << history.0.from), Board::bit_to_algebraic(1u64 << history.0.to));
    //     }
    //     h
    // }

    pub fn perft(&mut self, colour: Colour, depth: usize, counter: &mut u64) {
        if depth == 0 {
            *counter += 1;
            return;
        }
        
        let moves = self.get_legal_moves(colour, None);

        for m in moves {
            self.make_move(m, false).unwrap();
            self.perft(!colour, depth - 1, counter);
            self.unmake_move().unwrap();
        }
    }

    fn generate_magic_table_rook(&mut self) {
        for sq in 0..=63 {
            let mask = Board::rook_mask(sq);
            let occupancies = Board::generate_blocker_permutation(mask);
            let mut magic = Magic::new();
            magic.shift = (64 - mask.count_ones() ) as usize;
            magic.mask = mask;

            loop {
                magic.magic_number = rand_u64();
                magic.attacks = [None; 4096];
                let mut fail = false;

                for &occ in &occupancies {
                    let index = magic.get_index(occ);
                    let attack = Board::magic_rook_moves(sq, occ);

                    if let Some(existing) = magic.attacks[index] {
                        if existing != attack {
                            fail = true;
                            break;
                        }
                    } else {
                        magic.attacks[index] = Some(attack);
                    }
                }

                if !fail {
                    break;
                }
            }

            self.rook_magic_table.push(magic);
        }
    }

    fn generate_magic_table_bishop(&mut self) {
        for sq in 0..=63 {
            let mask = Board::bishop_mask(sq);
            let occupancies = Board::generate_blocker_permutation(mask);
            let mut magic = Magic::new();
            magic.shift = (64 - mask.count_ones() ) as usize;
            magic.mask = mask;

            loop {
                magic.magic_number = rand_u64();
                magic.attacks = [None; 4096];
                let mut fail = false;

                for &occ in &occupancies {
                    let index = magic.get_index(occ);
                    let attack = Board::magic_bishop_moves(sq, occ);

                    if let Some(existing) = magic.attacks[index] {
                        if existing != attack {
                            fail = true;
                            break;
                        }
                    } else {
                        magic.attacks[index] = Some(attack);
                    }
                }

                if !fail {
                    break;
                }
            }

            self.bishop_magic_table.push(magic);
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

    pub fn rook_mask(sq: usize) -> Bitboard {
        let rank = sq / 8;
        let file = sq % 8;
        let mut mask = 0u64;

        // Horizontal moves (exclude edges)
        for f in (file + 1)..7 {
            mask |= 1u64 << (rank * 8 + f);
        }
        for f in (1..file).rev() {
            mask |= 1u64 << (rank * 8 + f);
        }

        // Vertical moves (exclude edges)
        for r in (rank + 1)..7 {
            mask |= 1u64 << (r * 8 + file);
        }
        for r in (1..rank).rev() {
            mask |= 1u64 << (r * 8 + file);
        }

        mask
    }

    pub fn bishop_mask(sq: usize) -> u64 {
        let rank = sq / 8;
        let file = sq % 8;
        let mut mask = 0u64;
        let directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

        for (dr, df) in directions {
            let mut r = rank as i32 + dr;
            let mut f = file as i32 + df;

            // exclude edges → r>0 && r<7 && f>0 && f<7
            while r > 0 && r < 7 && f > 0 && f < 7 {
                mask |= 1u64 << (r * 8 + f);
                r += dr;
                f += df;
            }
        }
        mask
    }

    #[inline(always)]
    fn get_bit_position(rank: usize, file: usize) -> usize {
        rank * 8 + file
    }

    #[inline(always)]
    pub fn all_pieces(&self) -> Bitboard {
        self.white_pieces() | self.black_pieces()
    }

    #[inline(always)]
    pub fn empty(&self) -> Bitboard {
        !self.all_pieces()
    }

    #[inline(always)]
    fn pieces(&self, colour: Colour) -> Bitboard {
        match colour {
            Colour::White => self.white_pieces(),
            Colour::Black => self.black_pieces(),
        }
    }

    #[inline(always)]
    pub fn white_pieces(&self) -> Bitboard {
        self.bitboards[Colour::White as usize]
            .iter()
            .copied()
            .reduce(|a, b| a | b)
            .unwrap()
    }

    #[inline(always)]
    pub fn black_pieces(&self) -> Bitboard {
        self.bitboards[Colour::Black as usize]
            .iter()
            .copied()
            .reduce(|a, b| a | b)
            .unwrap()
    }

    #[inline(always)]
    fn set_bit(bitboard: &mut Bitboard, i: usize) {
        *bitboard |= 1 << i;
    }

    #[inline(always)]
    fn get_bit(bitboard: Bitboard, i: usize) -> Bitboard {
        bitboard & 1 << i
    }

    #[inline(always)]
    fn clear_bit(bitboard: &mut Bitboard, i: usize) {
        *bitboard &= !(1 << i);
    }

    #[inline(always)]
    pub fn pop_lsb(bitboard: &mut Bitboard) -> Option<usize> {
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
                            self.hash ^= self.zobrist_hash.pieces[colour as usize][piece as usize][Board::get_bit_position(rank, file)];
                            file += 1;
                        }
                    }
                }
                1 => {
                    self.turn = match field {
                        "w" => Colour::White,
                        "b" => {
                            self.hash ^= self.zobrist_hash.black_to_move;
                            Colour::Black
                        },
                        _ => panic!("Invalid turn field: {}", field),
                    };
                }
                2 => {
                    self.castling_rights.white_king = if field.contains("K") {
                        self.hash ^= self.zobrist_hash.castling_rights[0];
                        Castle::new().white_king
                    } else {
                        0
                    };
                    self.castling_rights.white_queen = if field.contains("Q") {
                        self.hash ^= self.zobrist_hash.castling_rights[1];
                        Castle::new().white_queen
                    } else {
                        0
                    };
                    self.castling_rights.black_king = if field.contains("k") {
                        self.hash ^= self.zobrist_hash.castling_rights[2];
                        Castle::new().black_king
                    } else {
                        0
                    };
                    self.castling_rights.black_queen = if field.contains("q") {
                        self.hash ^= self.zobrist_hash.castling_rights[3];
                        Castle::new().black_queen
                    } else {
                        0
                    };
                }
                3 => {
                    self.en_passant = Board::algebraic_to_bit(field.to_string());
                    self.hash ^= self.zobrist_hash.en_passant[self.en_passant.trailing_zeros() as usize % 8];
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
            print!("\x1b[38;5;15m\x1b[48;5;236m{} \x1b[0m", rank + 1);
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

    pub fn display_from_bitboards(&mut self) {
        for rank in (0..=7).rev() {
            print!("\x1b[38;5;15m\x1b[48;5;236m{} \x1b[0m", rank + 1);
            for file in 0..=7 {
                let mut symbol= " ".to_string();
                for (colour, i) in self.bitboards.iter().enumerate() {
                    for (piece, j) in i.iter().enumerate() {
                        if Board::get_bit(*j, Board::get_bit_position(rank, file)) == 1 {
                            symbol = Colour::from_num(colour).unwrap().symbol().to_string() + Pieces::from_num(piece).unwrap().symbol();
                            break;
                        }
                    }
                }

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
            print!("\x1b[38;5;15m\x1b[48;5;236m{} \x1b[0m", rank+1);
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

    pub fn find_bitboard(&self, bit_position: usize, colour: Colour) -> Option<Pieces> {
        for (piece, bit) in self.bitboards[colour as usize].iter().enumerate() {
            if Board::get_bit(*bit, bit_position) != 0 {
                return Some(Pieces::from_num(piece).unwrap())
            }
        }

        None
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

        Ok(Move::new(from, to, piece, capture, promotion))
    }

    pub fn make_move(&mut self, mut m: Move, validate: bool) -> Result<(), String> {
        if m.piece.0 != self.turn {
            return Err(format!("Invalid turn, it is {:?}'s move", self.turn));
        }

        if validate {
            if !self.get_legal_moves(m.piece.0, None).contains(&m) { return Err("Invalid move".to_string()) };
        }

        self.pieces[m.from] = None;
        self.pieces[m.to] = Some(m.piece);

        self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][m.piece.1 as usize][m.from];
        self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][m.piece.1 as usize][m.to];

        let old_half_move = self.half_moves;
        self.half_moves += 1;

        let pre_castle_state = self.castling_rights.clone();
        let old_en_passant = self.en_passant;

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

            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][m.piece.1 as usize][m.to];
            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][promotion as usize][m.to];

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

                            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][Pieces::Rook as usize][7];
                            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][Pieces::Rook as usize][5];

                            m.castle = true;
                        } else if m.to == self.castling_rights.white_queen.trailing_zeros() as usize
                        {
                            self.pieces[0] = None;
                            self.pieces[3] = Some((Colour::White, Pieces::Rook));
                            Board::clear_bit(
                                &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                                0,
                            );
                            Board::set_bit(
                                &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                                3,
                            );

                            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][Pieces::Rook as usize][0];
                            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][Pieces::Rook as usize][3];

                            m.castle = true;
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

                            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][Pieces::Rook as usize][63];
                            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][Pieces::Rook as usize][61];

                            m.castle = true;
                        } else if m.to == self.castling_rights.black_queen.trailing_zeros() as usize
                        {
                            self.pieces[56] = None;
                            self.pieces[59] = Some((Colour::Black, Pieces::Rook));
                            Board::clear_bit(
                                &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                                56,
                            );
                            Board::set_bit(
                                &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                                59,
                            );

                            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][Pieces::Rook as usize][56];
                            self.hash ^= self.zobrist_hash.pieces[m.piece.0 as usize][Pieces::Rook as usize][59];

                            m.castle = true;
                        }
                    }
                }
            }
            self.castling_rights.void(m.piece.0, &mut self.hash, &self.zobrist_hash);
        }

        if m.from == 0 || m.to == 0 {
            if self.castling_rights.white_queen != 0 { self.hash ^= self.zobrist_hash.castling_rights[1] };
            self.castling_rights.white_queen = 0;
        } else if m.from == 7 || m.to == 7 {
        if self.castling_rights.white_king != 0 { self.hash ^= self.zobrist_hash.castling_rights[0] };
            self.castling_rights.white_king = 0;
        } else if m.from == 56 || m.to == 56 {
            if self.castling_rights.black_queen != 0 { self.hash ^= self.zobrist_hash.castling_rights[3] };
            self.castling_rights.black_queen = 0;
        } else if m.from == 63 || m.to == 63 {
        if self.castling_rights.black_king != 0 { self.hash ^= self.zobrist_hash.castling_rights[2] };
            self.castling_rights.black_king = 0;
        }

        if let Some(piece) = m.capture {
            Board::clear_bit(
                &mut self.bitboards[piece.0 as usize][piece.1 as usize],
                m.to,
            );

            self.hash ^= self.zobrist_hash.pieces[piece.0 as usize][piece.1 as usize][m.to];

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
                            self.hash ^= self.zobrist_hash.pieces[Colour::Black as usize][Pieces::Pawn as usize][m.to - 8];
                        }
                        Colour::Black => {
                            self.pieces[m.to + 8] = None;
                            Board::clear_bit(
                                &mut self.bitboards[Colour::White as usize][Pieces::Pawn as usize],
                                m.to + 8,
                            );

                            self.hash ^= self.zobrist_hash.pieces[Colour::White as usize][Pieces::Pawn as usize][m.to + 8];
                        }
                    }
                    m.en_passant = true;
                }
            }
        }

        if self.en_passant != 0 {
            self.hash ^= self.zobrist_hash.en_passant[self.en_passant.trailing_zeros() as usize % 8];
        }
    
        self.move_history
            .push((m, pre_castle_state, old_en_passant, old_half_move));

        if (m.from as isize - m.to as isize).abs() == 16 && m.piece.1 == Pieces::Pawn {
            self.en_passant = (1 as Bitboard) << (m.from + m.to) / 2;
            self.hash ^= self.zobrist_hash.en_passant[self.en_passant.trailing_zeros() as usize % 8];
        } else {
            self.en_passant = 0;
        }

        if self.half_moves >= 100 {
            self.state = State::Draw;
        }

        if self.turn == Colour::Black {

            self.full_moves += 1;
        }

        self.turn = !self.turn;
        self.hash ^= self.zobrist_hash.black_to_move;

        Ok(())
    }

    pub fn unmake_move(&mut self) -> Result<(), String> {
        let (last_move, old_castle, old_en_passant, half_move) = self
            .move_history
            .pop()
            .ok_or_else(|| "No move history".to_string())?;

        if self.castling_rights.white_king != old_castle.white_king {
            self.hash ^= self.zobrist_hash.castling_rights[0];
        }
        if self.castling_rights.white_queen != old_castle.white_queen {
            self.hash ^= self.zobrist_hash.castling_rights[1];
        }
        if self.castling_rights.black_king != old_castle.black_king {
            self.hash ^= self.zobrist_hash.castling_rights[2];
        }
        if self.castling_rights.black_queen != old_castle.black_queen {
            self.hash ^= self.zobrist_hash.castling_rights[3];
        }

        self.castling_rights = old_castle;
        self.half_moves = half_move;

        if self.en_passant != 0 {
            self.hash ^= self.zobrist_hash.en_passant[self.en_passant.trailing_zeros() as usize % 8];
        }

        if old_en_passant != 0 {
            self.hash ^= self.zobrist_hash.en_passant[old_en_passant.trailing_zeros() as usize % 8];
        }

        self.en_passant = old_en_passant;

        self.pieces[last_move.to] = None;
        self.pieces[last_move.from] = Some(last_move.piece);

        Board::clear_bit(
            &mut self.bitboards[last_move.piece.0 as usize][last_move.piece.1 as usize],
            last_move.to,
        );
        Board::set_bit(
            &mut self.bitboards[last_move.piece.0 as usize][last_move.piece.1 as usize],
            last_move.from,
        );

        self.hash ^= self.zobrist_hash.pieces[last_move.piece.0 as usize][last_move.piece.1 as usize][last_move.to];
        self.hash ^= self.zobrist_hash.pieces[last_move.piece.0 as usize][last_move.piece.1 as usize][last_move.from];
        

        if let Some(piece) = last_move.capture {
            self.pieces[last_move.to] = Some(piece);
            Board::set_bit(
                &mut self.bitboards[piece.0 as usize][piece.1 as usize],
                last_move.to,
            );

            self.hash ^= self.zobrist_hash.pieces[piece.0 as usize][piece.1 as usize][last_move.to];
        }

        if let Some(promotion) = last_move.promotion {
            Board::clear_bit(
                &mut self.bitboards[last_move.piece.0 as usize][promotion as usize],
                last_move.to,
            );

            self.hash ^= self.zobrist_hash.pieces[last_move.piece.0 as usize][promotion as usize][last_move.to];
        }

        if last_move.en_passant {
            match last_move.piece.0 {
                Colour::White => {
                    let sq = last_move.to - 8;
                    self.pieces[sq] = Some((Colour::Black, Pieces::Pawn));
                    Board::set_bit(
                        &mut self.bitboards[Colour::Black as usize][Pieces::Pawn as usize],
                        sq,
                    );
                    self.hash ^= self.zobrist_hash.pieces[Colour::Black as usize][Pieces::Pawn as usize][sq];
                }
                Colour::Black => {
                    let sq = last_move.to + 8;
                    self.pieces[sq] = Some((Colour::White, Pieces::Pawn));
                    Board::set_bit(
                        &mut self.bitboards[Colour::White as usize][Pieces::Pawn as usize],
                        sq,
                    );
                    self.hash ^= self.zobrist_hash.pieces[Colour::White as usize][Pieces::Pawn as usize][sq];
                }
            }
        }

        if last_move.castle {
            if last_move.to == 2 {
                self.pieces[3] = None;
                self.pieces[0] = Some((Colour::White, Pieces::Rook));

                Board::clear_bit(
                    &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                    3,
                );
                Board::set_bit(
                    &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                    0,
                );

                self.hash ^= self.zobrist_hash.pieces[Colour::White as usize][Pieces::Rook as usize][3];
                self.hash ^= self.zobrist_hash.pieces[Colour::White as usize][Pieces::Rook as usize][0];
            } else if last_move.to == 6 {
                self.pieces[5] = None;
                self.pieces[7] = Some((Colour::White, Pieces::Rook));

                Board::clear_bit(
                    &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                    5,
                );
                Board::set_bit(
                    &mut self.bitboards[Colour::White as usize][Pieces::Rook as usize],
                    7,
                );

                self.hash ^= self.zobrist_hash.pieces[Colour::White as usize][Pieces::Rook as usize][5];
                self.hash ^= self.zobrist_hash.pieces[Colour::White as usize][Pieces::Rook as usize][7];
            } else if last_move.to == 58 {
                self.pieces[59] = None;
                self.pieces[56] = Some((Colour::Black, Pieces::Rook));

                Board::clear_bit(
                    &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                    59,
                );
                Board::set_bit(
                    &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                    56,
                );

                self.hash ^= self.zobrist_hash.pieces[Colour::Black as usize][Pieces::Rook as usize][59];
                self.hash ^= self.zobrist_hash.pieces[Colour::Black as usize][Pieces::Rook as usize][56];
            } else if last_move.to == 62 {
                self.pieces[61] = None;
                self.pieces[63] = Some((Colour::Black, Pieces::Rook));

                Board::clear_bit(
                    &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                    61,
                );
                Board::set_bit(
                    &mut self.bitboards[Colour::Black as usize][Pieces::Rook as usize],
                    63,
                );

                self.hash ^= self.zobrist_hash.pieces[Colour::Black as usize][Pieces::Rook as usize][61];
                self.hash ^= self.zobrist_hash.pieces[Colour::Black as usize][Pieces::Rook as usize][63];
            }
        }

        // if last_move.piece.1 == Pieces::Pawn {
        //     self.half_moves = 0;
        // }

        if !self.turn == Colour::Black {
            if self.full_moves > 0 { self.full_moves -= 1 };
        }

        self.state = State::Continue;

        self.turn = !self.turn;
        self.hash ^= self.zobrist_hash.black_to_move;

        Ok(())
    }

    pub fn get_game_state(&mut self, validate_no_moves: bool) -> State {
        if validate_no_moves {
            if self.get_legal_moves(self.turn, None).len() != 0 { return State::Continue }
        }
        if self.is_check(Colour::White) {
            return State::Checkmate(Colour::White);
        }
        else if self.is_check(Colour::Black) {
            return State::Checkmate(Colour::Black);
        }
        else {
            return State::Stalemate;
        }
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

fn rand_u64() -> u64 {
    use rand::Rng;

    let mut rng = rand::rng();
    rng.random::<u64>() & rng.random::<u64>() & rng.random::<u64>()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Move {
    pub from: usize,
    pub to: usize,
    pub piece: (Colour, Pieces),
    pub capture: Option<(Colour, Pieces)>,
    pub promotion: Option<Pieces>,
    pub en_passant: bool,
    pub castle: bool,
    pub score: u32
}

impl Move {
    pub fn new(
        from: usize,
        to: usize,
        piece: (Colour, Pieces),
        capture: Option<(Colour, Pieces)>,
        promotion: Option<Pieces>,
    ) -> Move {
        Move {
            from,
            to,
            promotion,
            capture,
            piece,
            en_passant: false,
            castle: false,
            score: 0
        }
    }
}

#[derive(Debug, Clone)]
pub struct Magic {
    magic_number: Bitboard,
    shift: usize,
    attacks: [Option<Bitboard>; 4096],
    mask: Bitboard
}

impl Magic {
    fn new() -> Magic {
        Magic {
            magic_number: 0,
            shift: 0,
            attacks: [None; 4096],
            mask: 0,
        }
    }

    fn get_index(&self, occ: Bitboard) -> usize {
        (occ.wrapping_mul(self.magic_number) >> self.shift) as usize
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
    pub fn from_num(num: usize) -> Option<Pieces> {
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

    pub fn symbol(&self) -> &str {
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

#[derive(Clone, Debug)]
pub struct Castle {
    pub white_king: Bitboard,
    pub white_queen: Bitboard,
    pub black_king: Bitboard,
    pub black_queen: Bitboard,
}

const WHITE_KING_CHECK: Bitboard = (3 as Bitboard) << 5;
const BLACK_KING_CHECK: Bitboard = ((3 as Bitboard) << 5) << 56;
const WHITE_QUEEN_CHECK: Bitboard = (3 as Bitboard) << 2;
const BLACK_QUEEN_CHECK: Bitboard = ((3 as Bitboard) << 2) << 56;
const WHITE_MASK_KING: Bitboard = (3 as Bitboard) << 5;
const WHITE_MASK_QUEEN: Bitboard = (7 as Bitboard) << 1;
const BLACK_MASK_KING: Bitboard = ((3 as Bitboard) << 5) << 56;
const BLACK_MASK_QUEEN: Bitboard = ((7 as Bitboard) << 1) << 56;

impl Castle {
    fn new() -> Castle {
        Castle {
            white_king: (1 as Bitboard) << 6,
            white_queen: (1 as Bitboard) << 2,
            black_king: (1 as Bitboard) << 62,
            black_queen: (1 as Bitboard) << 58,
        }
    }

    fn colour(&self, colour: Colour, all_pieces: Bitboard, conflicts: [bool; 2]) -> Bitboard {
        match colour {
            Colour::White => {
                self.white_king
                    & (if all_pieces & WHITE_MASK_KING == 0 && !conflicts[0] {
                        u64::MAX
                    } else {
                        0
                    })
                    | self.white_queen
                        & (if all_pieces & WHITE_MASK_QUEEN == 0 && !conflicts[1] {
                            u64::MAX
                        } else {
                            0
                        })
            }
            Colour::Black => {
                self.black_king
                    & (if all_pieces & BLACK_MASK_KING == 0 && !conflicts[0] {
                        u64::MAX
                    } else {
                        0
                    })
                    | self.black_queen
                        & (if all_pieces & BLACK_MASK_QUEEN == 0 && !conflicts[1] {
                            u64::MAX
                        } else {
                            0
                        })
            }
        }
    }

    fn has_castle(&self, colour: Colour) -> bool {
        (match colour {
            Colour::White => self.white_king | self.white_queen,
            Colour::Black => self.black_king | self.black_queen,
        }) != 0
    }

    fn void(&mut self, colour: Colour, hash: &mut u64, zobrist: &ZobristHashing) {
        match colour {
            Colour::White => {
                if self.white_king != 0 { *hash ^= zobrist.castling_rights[0] };
                if self.white_queen != 0 { *hash ^= zobrist.castling_rights[1] };
                self.white_king = 0;
                self.white_queen = 0;
            }
            Colour::Black => {
                if self.black_king != 0 { *hash ^= zobrist.castling_rights[2] };
                if self.black_queen != 0 { *hash ^= zobrist.castling_rights[3] };
                self.black_king = 0;
                self.black_queen = 0;
            }
        };
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum State {
    Checkmate(Colour),
    Stalemate,
    Draw,
    Continue
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    #[test]
    fn perft_test_default() {
        let mut board = Board::new(0);
        board.init();
        board.load_fen(DEFAULT.to_string());

        let mut nodes_evaluated = 0;
        board.perft(board.turn, 5, &mut nodes_evaluated);
        assert_eq!(nodes_evaluated, 4865609);
    }

    #[test]
    fn perft_test_kiwipete() {
        let mut board = Board::new(0);
        board.init();
        board.load_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1".to_string(),
        );

        let mut nodes_evaluated = 0;
        board.perft(board.turn, 4, &mut nodes_evaluated);
        assert_eq!(nodes_evaluated, 4085603);
    }
}
