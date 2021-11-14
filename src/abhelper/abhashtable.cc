/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "abhelper\abhashtable.h"

#include <intrin.h>
#include "chess\bitboard.h"


namespace lczero {

// Based on this link:
// https://web.archive.org/web/20071031100051/http://www.brucemo.com/compchess/programming/hashing.htm

 HashTableResponse HashTable::get(uint64_t key, int depth, int alpha,
                                 int beta) const {
  
  HashTableResponse response;

  int hashIndex = key % entryCount_;
  HashTableEntry entry = hashTable_[hashIndex];

  // confirm the keys match and we don't have a collision.
  response.bestMove = entry.move;
  if (entry.key == key) {
    if (entry.depth >= depth) {
      switch (entry.entryType) {
        case HashTableEntry::EntryType::EXACT:
          response.IsKnownValue = true;
          response.value = entry.eval;
          return response;
          break;
        case HashTableEntry::EntryType::UPPER_BOUND:
          if (entry.eval <= alpha) {
            response.IsKnownValue = true;
            response.value = alpha;
            return response;
          }
          break;
        default:
          // HashTableEntry::EntryType::LOWER_BOUND
          if (entry.eval >= beta) {
            response.IsKnownValue = true;
            response.value = beta;
            return response;
          }
          break;
      }
    }
  }
  return response;
};
 
 uint64_t HashTable::getKey(Position position) {
   return keyGenerator.getKey(position);
  };

  void HashTable::put(HashTableEntry entry) {
    hashTable_[entry.key % entryCount_] = entry;
  }
 
  void HashTable::put(uint64_t key, int depth, Move move,
                      int eval, HashTableEntry::EntryType entryType, int age) {
    HashTableEntry entry;
    entry.key = key;
    entry.depth = depth;
    entry.move = move;
    entry.entryType = entryType;
    entry.eval = eval;
    entry.age = age;

    put(entry);
  }

  void HashTable::put(uint64_t key, int depth, int eval,
                      HashTableEntry::EntryType entryType, int age) {
    HashTableEntry entry;
    entry.key = key;
    entry.depth = depth;
    //entry.move = move;
    entry.entryType = entryType;
    entry.eval = eval;
    entry.age = age;

    put(entry);
  }

  void ZobristKeys::initKeys() {
    int key_index = 0;
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 64; k++) {
          piece_position_key[i][j][k] = keys[key_index++];
        };
      };
    };
    black_to_move_key = keys[key_index++];

    for (int i = 0; i < 8; i++) {
      enpassant_file_key[i] = keys[key_index++];
    }

    for (int i = 0; i < 16; i++) {
      castling_key[i] = keys[key_index++];
    }
  };

  uint64_t ZobristKeys::getKey(Position position) {
    uint64_t key = 0;
    
    if (position.IsBlackToMove()) {    
      key = key ^ black_to_move_key;
    }
    
    ChessBoard board = position.GetWhiteBoard();
    int castling_index = board.castlings().as_int();
    key = key ^ castling_key[castling_index];

    BitBoard en_passant_board = board.en_passant();
    if (!en_passant_board.empty()) {
      unsigned long index;
      unsigned char position_index =
          _BitScanReverse64(&index, en_passant_board.as_int());
      if (index < 8) {
        key = key ^ enpassant_file_key[index];
      } else {
        key = key ^ enpassant_file_key[index - 56];
      }
    }
    
    key = key ^
          piece_position_key[AbEnum::AbPieceType::KING][AbEnum::AbColor::WHITE]
                                  [board.ourKing().as_int()];
    key = key ^
          piece_position_key[AbEnum::AbPieceType::KING][AbEnum::AbColor::BLACK]
                                  [board.theirKing().as_int()];
      
    setPiecesKey(board.queens(), board, AbEnum::AbPieceType::QUEEN, key);
    setPiecesKey(board.rooks(), board, AbEnum::AbPieceType::ROOK, key);
    setPiecesKey(board.bishops(), board, AbEnum::AbPieceType::BISHOP, key);
    setPiecesKey(board.knights(), board, AbEnum::AbPieceType::KNIGHT, key);
    setPiecesKey(board.pawns(), board, AbEnum::AbPieceType::PAWN, key);

    return key;
  };

  void ZobristKeys::setPiecesKey(const BitBoard& piece_board,
                               const ChessBoard& board,
                                 const AbEnum::AbPieceType& piece_type,
                               uint64_t& key) {
    if (!piece_board.empty()) {
      BitBoard white_pieces = board.ours();
      BitBoard black_pieces = board.theirs();

      setPiecesKeyByColour(piece_board, white_pieces, piece_type,
                           AbEnum::AbColor::WHITE, key);
      setPiecesKeyByColour(piece_board, black_pieces, piece_type,
                           AbEnum::AbColor::BLACK, key);
    }
  };

  void ZobristKeys::setPiecesKeyByColour(BitBoard piece_board,
                                         BitBoard color_board,
                                         AbEnum::AbPieceType piece_type,
                                         AbEnum::AbColor color,
                            uint64_t& key) {
    BitBoard one_color_position = piece_board & color_board;
    if (!one_color_position.empty()) {
      int piece_index = 0;
      int piece_total = one_color_position.count_few();
      uint64_t pos = one_color_position.as_int();
      while (piece_index < piece_total) {
        unsigned long index;
        unsigned char position_index = _BitScanForward64(&index, pos);
        key = key ^ piece_position_key[piece_type][color][index];
        pos &= pos - 1;
        piece_index++;
      }
    }
  };
}  // namespace lczero