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
#pragma once

#include "abhelper/absearch.h"

#include <intrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>
#include <map>

#include "abhelper/abhashtable.h"

#include "chess/bitboard.h"
#include "neural/cache.h"
#include "neural/encoder.h"
#include "utils/fastmath.h"
#include "utils/random.h"

namespace lczero {

static const int min_eval = std::numeric_limits<int>::min();
static const int max_eval = std::numeric_limits<int>::max();
static const int MAX_PLY = 50;

static const int mvvLva[6][6] = {
    {15, 14, 13, 12, 11, 10},  // victim Pawn
    {25, 24, 23, 22, 21, 20},  // victim Knight
    {35, 34, 33, 32, 31, 30},  // victim Bishop
    {45, 44, 43, 42, 41, 40},  // victim Rook
    {55, 54, 53, 52, 51, 50},  // victim Queen    
};
static const int bestMoveMvvLva = 100;

int AlphaBetaSearch1::SearchInit(Position position, int ply) {
    SearchData search_data;  //= SearchData(position);
    search_data.positionList.push_back(position);
    search_data.hash_key_list.push_back(hash.getKey(position));

    PrincipleVariation pv;
    int value = 0;
    for (int depth = 1; depth <= ply; depth++) {
      value = AlphaBeta(search_data, depth, min_eval, max_eval, 0 /* ply */, pv);

      //    if (TimedOut()) break;
    }
    return value;
  };

  int AlphaBetaSearch1::AlphaBeta(SearchData search_data, int depth, int alpha,
                                  int beta, int ply, PrincipleVariation& pv) {
    HashTableEntry::EntryType hashf = HashTableEntry::UPPER_BOUND;

    // hash first follows brucemo, xiphos has it the other way around
    uint64_t key = search_data.hash_key_list.back();
    HashTableResponse response = hash.get(key, depth, alpha, beta);
    if (response.IsKnownValue) {
      return response.value;
    }

    if (depth <= 0) {
        // do quiescence search--blunder & xiphos & tscp
      int eval = Evaluate(search_data.positionList.back());
      hash.put(key, depth, eval, HashTableEntry::EXACT, 0);

      return eval;
    }

    //--blunder& xiphos
    if (ply >= MAX_PLY) { 
      return Evaluate(search_data.positionList.back());
    }

    bool isRoot = ply == 0;
    if (!isRoot && IsDraw(search_data)) {
      return 0;
    }

    Position currentPosition = search_data.getCurrentPosition();
    std::multimap<int, Move, std::greater<int>> moveList =
        GetOrderedMoves(currentPosition, key);
        search_data.getCurrentPosition().GetBoard().GenerateLegalMoves();

    Move bestMove;
    int bestEval = alpha;

    for (std::pair<int, Move> elem : moveList) {
      Move move = elem.second;
      std::string move_data = move.as_string();
      makeMove(search_data, move);
      int eval =
          -AlphaBeta(search_data, depth - 1, -beta, -alpha, ply + 1, pv);
      std::cout << move_data << "(" << eval << ")" << std::endl;
      unmakeMove(search_data);
      if (eval > bestEval) {
        bestEval = eval;
        bestMove = move;
      }
      if (eval >= beta) {
        hashf = HashTableEntry::LOWER_BOUND;
     //     hash.put(key, depth, move, eval, , 0);
        // store killer --blunder
        break;
      }
      if (eval > alpha) {
        hashf = HashTableEntry::EXACT;
        bestMove = move;
        alpha = eval;
      }
    }
    hash.put(key, depth, bestMove, bestEval, hashf, 0);

    if (moveList.empty()) {
      if (currentPosition.GetBoard().IsUnderCheck()) {
        return min_eval + ply;
      } else {
        return 0;
      };
    };
    return bestEval;
  }

  std::multimap<int, Move, std::greater<int>> AlphaBetaSearch1::GetOrderedMoves(
      Position currentPosition, uint64_t key) {
    MoveList moveList = currentPosition.GetBoard().GenerateLegalMoves();
    std::multimap<int, Move, std::greater<int>> mmapOfPos;
    if (moveList.empty()) {
      return mmapOfPos;
    };

    Move bestMove;
    HashTableResponse response = hash.get(key, 0, min_eval, max_eval);
    if (response.IsKnownValue && response.bestMove.as_packed_int() != 0) {
      bestMove = response.bestMove;
      mmapOfPos.insert(std::pair<int, Move>(bestMoveMvvLva, bestMove));
    }
    for (Move move : moveList) {
      if (move == bestMove) {
        continue;
      }
      mmapOfPos.insert(
          std::pair<int, Move>(getMoveOrderKey(currentPosition, move), move));
    } 
   
    return mmapOfPos;
  }
  const int AlphaBetaSearch1::getMoveOrderKey(Position position, Move move) {
    int key = 0; 
    std::cout << move.as_string() << std::endl;
    
    if (isCapture(position, move) != 0) {
      std::pair<AbEnum::AbPieceType, AbEnum::AbPieceType> moveCapturePieces =
          getMoveCapturePieces(position, move);
      int answer = mvvLva[4][1];
      int answer2 = mvvLva[1][4];
      AbEnum::AbPieceType capture = moveCapturePieces.first;
      AbEnum::AbPieceType captured = moveCapturePieces.second;
      uint8_t ct = capture;
      uint8_t ctd = captured;

      key = mvvLva[captured][capture];
      int key1 = mvvLva[ct][ctd];
    } else {
      //
    }
    return key;
  }

  bool AlphaBetaSearch1::isCapture(Position position, Move move) {
    ChessBoard board = position.GetBoard();

    uint64_t moveTo = move.to().as_board();
    
    BitBoard bb = board.theirs();
    uint64_t bbInt = bb.as_int();
    uint64_t key = moveTo & bbInt;
    
    if (key != 0) {
      std::cout << "Capture!" << std::endl;
    }
    return key != 0;
  }

  std::pair<AbEnum::AbPieceType, AbEnum::AbPieceType> 
      AlphaBetaSearch1::getMoveCapturePieces(
      Position position, Move move) {
    ChessBoard board = position.GetBoard();
    BitBoard bb = board.theirs();
    AbEnum::AbPieceType capturingPiece =
        getPieceAtSquare(move.from().as_board(), board);
    AbEnum::AbPieceType capturedPiece =
        getPieceAtSquare(move.to().as_board(), board);

    return std::pair<AbEnum::AbPieceType, AbEnum::AbPieceType>(capturingPiece,
                                                               capturedPiece);
  }

  AbEnum::AbPieceType AlphaBetaSearch1::getPieceAtSquare(
      uint64_t key, ChessBoard board) {
    AbEnum::AbPieceType pieceType = AbEnum::QUEEN;
    BitBoard isPawn = board.pawns() & key;
    if (isPawn != 0) {
      pieceType = AbEnum::PAWN;
    } else {
      BitBoard isKnight = board.knights() & key;
      if (isKnight != 0) {
        pieceType = AbEnum::KNIGHT;
      } else {
        BitBoard isBishop = board.bishops() & key;
        if (isBishop != 0) {
          pieceType = AbEnum::BISHOP;
        } else {
          BitBoard isRook = board.rooks() & key;
          if (isRook != 0) {
            pieceType = AbEnum::ROOK;
          }
        }
      }
    }
    return pieceType;
  }
  void AlphaBetaSearch1::makeMove(SearchData& search_data, Move move) {
    //prints as white
    std::cout << "makeMove:" + move.as_string() << std::endl;

    Position currentPosition = search_data.getCurrentPosition();
    
    Position newPosition = Position(currentPosition, move);

    search_data.positionList.push_back(newPosition);
    // capture has happened if one of their pieces has disappeared.
    uint64_t newKey = hash.updateKey(search_data.hash_key_list.back(), currentPosition, newPosition);
    search_data.hash_key_list.push_back(newKey);
  };

  int AlphaBetaSearch1::Evaluate(Position position) { 
      return nnue.eval(position); };

  bool AlphaBetaSearch1::IsDraw(SearchData search_data) {
    //(search.Pos.Rule50 >= 100 || search.isDrawByRepition() ||
    //search.Pos.EndgameIsDrawn())
    return false;
  };

  }  // namespace lczero