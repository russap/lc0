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

#include "abhelper\abtest.h"

#include <iostream>
#include <intrin.h>
#include "abhelper\abhashtable.h"
#include "abhelper\absearch.h"
#include "chess\board.h"
#include "chess\position.h"


namespace lczero {


  void AbTesting::run() {
    AbTesting::testHashtable1();
    AbTesting::testHashtable2();
    AbTesting::testAbSearch1();
  };

  void AbTesting::testHashtable1() {
    std::string fen = "5k2/r3nb2/1p2pN1p/pP1pPp2/P2P1P2/8/4BK2/2R5 w - - 97 1";
    ChessBoard chessBoard = ChessBoard(fen);
    Position position = Position(chessBoard, 1, 97);

    HashTable hash = HashTable();

    uint64_t key = hash.getKey(position);
    if (key == 14659219040528120199) {
       std::cout << "testHashtable1: pass" << std::endl;
    } else {
      std::cout << "testHashtable1: fail" << std::endl;
    }
  }

  void AbTesting::testHashtable2() {
    std::string fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    ChessBoard chessBoard = ChessBoard(fen);
    Position position = Position(chessBoard, 0, 1);

    HashTable hash = HashTable();

    uint64_t key = hash.getKey(position);
    if (key == 18118954766289586162) {
      std::cout << "testHashtable1: pass" << std::endl;
    } else {
      std::cout << "testHashtable1: fail" << std::endl;
    }
  }

    void AbTesting::testAbSearch1() {
    bool isPass = false;
    std::string fen =
        "5kr1/q4n2/2ppb3/4P3/1QP5/pP1BN3/P1K4R/8 b - - 2 42";
    ChessBoard chessBoard = ChessBoard(fen);
    Position position = Position(chessBoard, 0, 1);

    AlphaBetaSearch1 abSearch;

    abSearch.SearchInit(position,2);
    if (isPass) {
      std::cout << "testHashtable1: pass" << std::endl;
    } else {
      std::cout << "testHashtable1: fail" << std::endl;
    }
  }
  

}  // namespace lczero