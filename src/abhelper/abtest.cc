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

#include "abhelper\abtest.h"

#include <iostream>
#include <intrin.h>
#include "abhelper/nnue.h"
#include "abhelper\abhashtable.h"
#include "abhelper\abeval.h"
#include "abhelper\absearch.h"
#include "chess\board.h"
#include "chess\position.h"


namespace lczero {


  void AbTesting::run() {
    testKeyGeneration();
    testPositionKeyUpdatesCorrectly();
    testNnEval1();
    testAbSearch1();
  };

  void AbTesting::testKeyGeneration() {
    int testNumber = 1;
    testKeyGenerationImpl(
        "5k2/r3nb2/1p2pN1p/pP1pPp2/P2P1P2/8/4BK2/2R5 w - - 97 1",
        14659219040528120199, testNumber++);
    testKeyGenerationImpl(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        18118954766289586162, testNumber++);
  }
  
  void AbTesting::testKeyGenerationImpl(std::string fen, uint64_t result,
                                        int testNumber) {    
    ChessBoard chessBoard = ChessBoard(fen);
    Position position = Position(chessBoard, 1, 97);

    HashTable hash = HashTable();

    uint64_t key = hash.getKey(position);
    if (key == result) {
      std::cout << "testKeyGeneration(" << testNumber << "): pass"
                << std::endl;
    } else {
      std::cout << "testKeyGeneration(" << testNumber << 
          "): failed. Expected key = " << 
          result << "Actual key=" << key << std::endl;
    }
  }
  
  void AbTesting::testPositionKeyUpdatesCorrectly() {
    int testNumber = 1;
    testPositionKeyUpdatesCorrectlyImpl(/* 1 capture*/
        "5k2/r3nb2/1p2pN1p/pP1pPp2/P2P1P2/8/4BK2/2R5 w - - 97 1", testNumber++);
    testPositionKeyUpdatesCorrectlyImpl(/* 6 captures, white*/
        "r4r2/pp1q1B2/1n1N1Qpk/2p1pb2/8/3P4/PPP2PPP/R4RK1 w - - 20 1", testNumber++);
    testPositionKeyUpdatesCorrectlyImpl(/* 3 captures, black*/
        "r4r2/pp1q1B2/1n1N1Qpk/2p1pb2/8/3P4/PPP2PPP/R4RK1 b - - 20 1", testNumber++);
  }

  void AbTesting::testPositionKeyUpdatesCorrectlyImpl(std::string fen, int fenTest) {
    
    ChessBoard chessBoard = ChessBoard(fen);
    Position position = Position(chessBoard, 1, 97);

    HashTable hash = HashTable();

    uint64_t key = hash.getKey(position);

    MoveList moveList = position.GetBoard().GenerateLegalMoves();
    int failCount = 0;
    int index = 0;
    for (Move move : moveList) {
    //Move move = moveList[30];
      std::string moveInfo = move.as_string();
      Position newPosition = Position(position, move);
      std::string fenNew =GetFen(newPosition);
      uint64_t moveKey = hash.updateKey(key, position, newPosition);
      uint64_t newKey = hash.getKey(newPosition);
      if (newKey != moveKey) {
        std::cout << "testPositionKeyUpdatesCorrectly(" << fenTest <<
                   "): failed, moveIndex=" << index << 
                   " ,move=" << moveInfo << ",fenTest=" << fenTest
                  << std::endl;
        failCount++;
      }
      index++;
    }
    if (failCount == 0) {
      std::cout << "testPositionKeyUpdatesCorrectly(" << fenTest << "): pass"
                << std::endl;
    } else {
      std::cout << "testPositionKeyUpdatesCorrectly(" << fenTest << "): failed "
                <<
          failCount << "/" << moveList.size() << " times." << std::endl;      
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
  
    void AbTesting::testNnEval1() {
        nnue_init("2020-11-28-g-nn-62ef826d1a6d.nnue");
        std::string const fen = "5kr1/q4n2/2ppb3/4P3/1QP5/pP1BN3/P1K4R/8 b - - 2 42";
        std::string const fen1 =
            "4k1r1/q4n2/2p1b3/3pP3/1QP5/pP1BN3/P1K4R/8 b - - 2 42";
        std::string const fen2 =
            "4k1r1/q4n2/2p1b3/3pP3/1QP5/pP1BN3/P1K4R/8 b - d6 2 42";
        ChessBoard chessBoard = ChessBoard(fen);
        Position position = Position(chessBoard, 0, 1);

        AlphaBetaEval abEval;

        int score = nnue_evaluate_fen(fen.c_str());
        int eval = abEval.eval(position);

        bool isPass = (score == eval);
        if (isPass) {
          std::cout << "testNnEval: pass" << std::endl;
        } else {
          std::cout << "testNnEval: fail" << std::endl;
        }        
        int score1 = nnue_evaluate_fen(fen1.c_str());
        int score2 = nnue_evaluate_fen(fen2.c_str());
        bool isPass1 = (score1 == score2);
        if (isPass1) {
          std::cout << "testNnEval: pass" << std::endl;
        } else {
          std::cout << "testNnEval: fail" << std::endl;
        }        
    }
  }  // namespace lczero