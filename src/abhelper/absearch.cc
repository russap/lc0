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

static const int min_eval = -100000;
static const int max_eval = 100000;
static const int MAX_PLY = 50;
static const int NULL_MOVE_REDUCTION = 2;

static const int mvvLva[6][6] = {
    {15, 14, 13, 12, 11, 10},  // victim Pawn
    {25, 24, 23, 22, 21, 20},  // victim Knight
    {35, 34, 33, 32, 31, 30},  // victim Bishop
    {45, 44, 43, 42, 41, 40},  // victim Rook
    {55, 54, 53, 52, 51, 50},  // victim Queen    
};

static const int bestMoveMvvLva = 100;
static const int killerMoveMvvLva = 5;

std::string PrincipleVariation::printMoves() {
  std::string output = "";
  if (moveList.size() > 0) {
    for (int i = 0; i < moveList.size() - 1; i++) {
      output += moveList[i].as_string() + ",";
    }
    output += moveList[moveList.size() - 1].as_string();
  }
  return output;
}
int AlphaBetaSearch1::SearchInit(Position position, int ply) {
    SearchData search_data;  //= SearchData(position);
    search_data.positionList.push_back(position);
    search_data.hash_key_list.push_back(hash.getKey(position));

    PrincipleVariation pv;
    int value = 0;
    for (int depth = 1; depth <= ply; depth++) {
      std::cout << " AB search start: depth=" << depth << std::endl;
      value = AlphaBeta(search_data, depth, min_eval, max_eval, 0 /* ply */, pv);
      std::cout << " AB search end: depth=" << depth << ",nodes=" << search_data.nodes <<
          ",value = " << value
                << ",pv=" << pv.printMoves();
      
      std::cout << std::endl;
    }
    return value;
  };

  int AlphaBetaSearch1::AlphaBeta(SearchData& search_data, int depth, int alpha,
                                  int beta, int ply, PrincipleVariation& pv) {
   // std::cout << " AB search ply: " << ply << (depth == 0 ? ", (leaf-node)": "") << std::endl;
    HashTableEntry::EntryType hashf = HashTableEntry::UPPER_BOUND;
 

    uint64_t key = search_data.hash_key_list.back();
    HashTableResponse response = hash.get(key, depth, alpha, beta);
    if (response.IsKnownValue) {
      search_data.nodes++;
      return response.value;
    }

    Position currentPosition = search_data.getCurrentPosition();
    if (depth <= 0) {      
      int eval = QuiesceSearch(currentPosition, alpha, beta, ply);
      hash.put(key, depth, eval, HashTableEntry::EXACT, 0);
      return eval;
    }

    search_data.nodes++;
    
    if (ply >= MAX_PLY) { 
      return Evaluate(search_data.positionList.back());
    }

    bool isRoot = ply == 0;
    if (!isRoot && IsDraw(search_data)) {
      return 0;
    }
    PrincipleVariation childPVLine;

    bool bIsNullMoveCheckNeeded = search_data.isNullCheckNeeded && (depth >= (NULL_MOVE_REDUCTION + 1)) &&
                             !currentPosition.GetBoard().IsUnderCheck();
    if (bIsNullMoveCheckNeeded) {
      currentPosition.flipSideToMove(); //nullMove
      search_data.isNullCheckNeeded = false; //no null move check inside the null move check
      int eval = -AlphaBeta(search_data, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 100, ply + 1,
                        childPVLine);
      currentPosition.flipSideToMove();  // undo nullMove
      search_data.isNullCheckNeeded = true;
      childPVLine.moveList.empty();
      if (eval > beta) {
        return beta;
      }
    }
 
    std::multimap<int, Move, std::greater<int>> moveList =
        GetOrderedMoves(
            search_data.killers[ply], 
            currentPosition, key);

    Move bestMove;
    int bestEval = alpha;
    int moveNumber = 0;
    bool isPvFound = false;
    for (std::pair<int, Move> elem : moveList) {
      moveNumber++;
      Move move = elem.second;
      std::string move_data = move.as_string();
      makeMove(search_data, move);
      int eval = 0;
      if (isPvFound) { //PV search
        eval = -AlphaBeta(search_data, depth - 1, -alpha - 100, -alpha, ply + 1,
                          childPVLine);
        if (eval > alpha && eval < beta) {
          eval = -AlphaBeta(search_data, depth - 1, -beta, -alpha, ply + 1,
                            childPVLine);
        }
      } else {
        eval = -AlphaBeta(search_data, depth - 1, -beta, -alpha, ply + 1,
                              childPVLine);
      }
      /* std::cout << std::string(2 * ply, ' ') << (ply + 1) << ":"
                << moveNumber
                << " "
                << move_data
                << "(" << eval << ")"
                << std::endl;*/
      unmakeMove(search_data);
      if (eval > bestEval) {
        bestEval = eval;
        bestMove = move;

        pv.moveList.clear();
        pv = childPVLine;
        pv.moveList.push_front(move);
      }
      if (eval >= beta) {
        hashf = HashTableEntry::LOWER_BOUND;
        bool isCapture = elem.first > 0;
        if (!isCapture) {
          search_data.killers[ply].insert(move);
        }
        break;
      }
      if (eval > alpha) {
        hashf = HashTableEntry::EXACT;
        isPvFound = true;
        alpha = eval;
      }
      childPVLine.moveList.clear();
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

    int AlphaBetaSearch1::BasicAlphaBeta(SearchData& search_data, int depth, int alpha,
                                  int beta, int ply, PrincipleVariation& pv) {
    std::cout << " AB search ply: " << ply
              << (depth == 0 ? ", (leaf-node)" : "") << std::endl;
    HashTableEntry::EntryType hashf = HashTableEntry::UPPER_BOUND;

    uint64_t key = search_data.hash_key_list.back();
    HashTableResponse response = hash.get(key, depth, alpha, beta);
    if (response.IsKnownValue) {
      search_data.nodes++;
      return response.value;
    }

    Position currentPosition = search_data.getCurrentPosition();
    if (depth <= 0) {
      int eval = QuiesceSearch(currentPosition, alpha, beta, ply);
      hash.put(key, depth, eval, HashTableEntry::EXACT, 0);
      return eval;
    }

    search_data.nodes++;

    if (ply >= MAX_PLY) {
      return Evaluate(search_data.positionList.back());
    }

    bool isRoot = ply == 0;
    if (!isRoot && IsDraw(search_data)) {
      return 0;
    }
    PrincipleVariation childPVLine;

    std::multimap<int, Move, std::greater<int>> moveList =
        GetOrderedMoves(search_data.killers[ply], currentPosition, key);

    Move bestMove;
    int bestEval = alpha;
    int moveNumber = 0;
    for (std::pair<int, Move> elem : moveList) {
      moveNumber++;
      Move move = elem.second;
      std::string move_data = move.as_string();
      makeMove(search_data, move);
      int eval = -AlphaBeta(search_data, depth - 1, -beta, -alpha, ply + 1,
                            childPVLine);
      std::cout << std::string(2 * ply, ' ') << (ply + 1) << ":" << moveNumber
                << " " << move_data << "(" << eval << ")" << std::endl;
      unmakeMove(search_data);
      if (eval > bestEval) {
        bestEval = eval;
        bestMove = move;

        pv.moveList.clear();
        pv = childPVLine;
        pv.moveList.push_front(move);
      }
      if (eval >= beta) {
        hashf = HashTableEntry::LOWER_BOUND;
        bool isCapture = elem.first > 0;
        if (!isCapture) {
          search_data.killers[ply].insert(move);
        }
        break;
      }
      if (eval > alpha) {
        hashf = HashTableEntry::EXACT;

        alpha = eval;
      }
      childPVLine.moveList.clear();
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

  int AlphaBetaSearch1::QuiesceSearch(Position currentPosition, int alpha, int beta, int ply) {
    
    int eval = Evaluate(currentPosition);
    if (eval >= beta) {
      return eval;
    }
    if (eval > alpha) {
      alpha = eval;
    }

    std::set<Move> killerMoves;
    std::multimap<int, Move, std::greater<int>> moveList =
        GetOrderedMoves(
            killerMoves, 
            currentPosition, 0);

   for (std::pair<int, Move> elem : moveList) {
      // skip non captures
      if (elem.first <= 0) {
       continue;
      }
      Move move = elem.second;
      std::string move_data = move.as_string();
      Position newPosition = Position(currentPosition, move);
      eval = -QuiesceSearch(newPosition, -beta, -alpha, ply);

      if (eval >= beta) {
        return beta;
      }

      if (eval > alpha) {
        alpha = eval;
      }
    }

    return alpha;
  }

  std::multimap<int, Move, std::greater<int>> AlphaBetaSearch1::GetOrderedMoves(
      std::set<Move> killerMoves, 
      Position currentPosition, uint64_t key) {
    MoveList moveList = currentPosition.GetBoard().GenerateLegalMoves();
    std::multimap<int, Move, std::greater<int>> mmapOfPos;
    if (moveList.empty()) {
      return mmapOfPos;
    };

    Move bestMove;
    if (key > 0) {
      HashTableResponse response = hash.get(key, 0, min_eval, max_eval);
      if (response.IsKnownValue && response.bestMove.as_packed_int() != 0) {
        bestMove = response.bestMove;
        mmapOfPos.insert(std::pair<int, Move>(bestMoveMvvLva, bestMove));
      }
    }

    ChessBoard board = currentPosition.GetBoard();
    BitBoard bb = board.theirs();
    uint64_t bbInt = bb.as_int();

    for (Move move : moveList) {
      if (move == bestMove) {
        continue;
      }
      bool iskillerMove = killerMoves.find(move) != killerMoves.end();
      if (iskillerMove) {
        mmapOfPos.insert(std::pair<int, Move>(killerMoveMvvLva, move));
      } else {
        mmapOfPos.insert(
            std::pair<int, Move>(getMoveOrderKey(board, bbInt, move), move));
      }
    } 
   
    return mmapOfPos;
  }
  const int AlphaBetaSearch1::getMoveOrderKey(ChessBoard board, uint64_t bbInt,
                                              Move move) {
    int key = 0; 
    //std::cout << move.as_string() << std::endl;
    
    if (isCapture(bbInt, move) != 0) {
      std::pair<AbEnum::AbPieceType, AbEnum::AbPieceType> moveCapturePieces =
          getMoveCapturePieces(board, move);
      key = mvvLva[moveCapturePieces.second][moveCapturePieces.first];
    } else {
      //
    }
    return key;
  }

  bool AlphaBetaSearch1::isCapture(uint64_t bbInt, Move move) {

    uint64_t moveTo = move.to().as_board();
    uint64_t key = moveTo & bbInt;
    
    return key != 0;
  }

  std::pair<AbEnum::AbPieceType, AbEnum::AbPieceType> 
      AlphaBetaSearch1::getMoveCapturePieces(ChessBoard board, Move move) {
         
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
  //  std::cout << "makeMove:" + move.as_string() << std::endl;

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