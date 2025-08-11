# dragonclash

> an attempt at making a chess engine that can successfully defeat a chimpanzee playing random moves (have not found a chimpanzee yet)

## features

it has (drumrolls please)

- bitboards!!!!
- magical bitboards!!!! (without the al in magical)
- negamax with alpha–beta pruning
- iterative deepening search framework
- aspiration windows
- quiescence search to avoid horizon effect
- null-move pruning
- late move reductions (LMR)
- futility pruning
- killer move heuristic (two killers per depth)
- history heuristic table for move ordering
- MVV–LVA move ordering
- repetition / stalemate / draw detection (including 50-move and insufficient material)
- mate distance scoring
- configurable search depth & optional time control
- material balance evaluation
- piece–square tables (midgame & endgame)
- king safety tables (midgame & endgame)
- pawn structure influence
- endgame detection logic
- mirrored square lookups for speed
- zobrist hashing (pieces, side to move, castling, en passant)
- efficient bitboard masks (files, ranks, diagonals, pawn types)
- full per-piece attack generation
- move legality checking
- transposition table with bound types and best-move storage
- SplitMix64 PRNG for hashing
- compact square & piece encoding
- fancy board representation (that doesn’t work in 99% of terminals)
- much more

## upcoming features

- everything else on the chessprogramming wiki
- a complete rewrite in assembly for optimisation
- opening book support (Polyglot or custom)
- endgame tablebases (Syzygy or similar)
- pondering / infinite search mode
- improved time management (per-move allocation)
- NNUE evaluation
- tuned eval parameters from self-play/datasets
- parallel search (multi-threading)
- check extensions
- advanced pawn structure eval (backward & doubled pawns)
- advanced king safety (pawn shields, open files)
- move ordering tuning for deep search
- earlier threefold detection in search
- play against a chimpanzee that is playing random moves

