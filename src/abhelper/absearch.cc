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

#include "abhelper/abhashtable.h"

#include "chess/bitboard.h"
#include "neural/cache.h"
#include "neural/encoder.h"
#include "utils/fastmath.h"
#include "utils/random.h"

namespace lczero {

static const int min_eval = std::numeric_limits<int>::min();
static const int max_eval = std::numeric_limits<int>::max();

struct SearchData {
  //SearchData(Position position) : position(position) {};
  Position getCurrentPosition() { return positionList.back(); }

  int thread_id;
  int hash_key_count;
  uint64_t hash_key{};

  int side{};
  std::vector<Position> positionList;
  HashTable hash_table{};

  uint64_t nodes{};
  uint64_t tablebase_hit_counter{};
  uint64_t key{};

  Move killers[100 /* MaxPly*/][2 /* MaxKillers*/];
  uint32_t history[2][64][64];

  uint8_t SpecifiedDepth{};
};

class AlphaBetaSearch1 {
 public:
  int SearchInit(Position position) {
    SearchData search_data;  //= SearchData(position);
    search_data.positionList.push_back(position);

    for (int depth = 1;; depth++) {
      int value = AlphaBeta(search_data, depth, min_eval, max_eval, 5);

      //    if (TimedOut()) break;
    }
  };

  int AlphaBeta(SearchData search_data, int depth, int alpha, int beta, int ply
                /*pv_node*/) {
    HashTableEntry::EntryType hashf = HashTableEntry::UPPER_BOUND;

    int64_t key = 0;
    HashTableResponse response = hash.get(key, depth, alpha, beta);
    if (response.IsKnownValue) {
      return response.value;
    }

    if (depth <= 0) {
      int eval = Evaluate(search_data.positionList.back());
      hash.put(key, depth, eval, HashTableEntry::EXACT, 0);

      return eval;
    }

    MoveList moveList =
        search_data.getCurrentPosition().GetBoard().GenerateLegalMoves();

    Move bestMove;

    for (Move move : moveList) {
      makeMove(search_data, move);
      int eval = -AlphaBeta(search_data, depth - 1, -beta, -alpha, ply + 1);
      unmakeMove(search_data);
      if (eval >= beta) {
        hash.put(key, depth, move, eval, HashTableEntry::LOWER_BOUND, 0);
        return beta;
      }
      if (eval > alpha) {
        hashf = HashTableEntry::EXACT;
        bestMove = move;
        alpha = eval;
      }
    }
    hash.put(key, depth, bestMove, alpha, HashTableEntry::UPPER_BOUND, 0);

    return alpha;
  }

  int Evaluate(Position position) { return 0; }

 protected:
  void makeMove(SearchData& search_data, Move move) {
    Position position_new = Position(search_data.getCurrentPosition(), move);
    search_data.positionList.push_back(position_new);
  }

  void unmakeMove(SearchData& search_data) {
    search_data.positionList.pop_back();
  }

 private:
  HashTable hash;
};

}  // namespace lczero