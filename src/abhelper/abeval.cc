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

#include "abhelper/abeval.h"
#include "abhelper/nnue.h"

namespace lczero {
/* 
 * Piece codes are:
 * wking = 1, wqueen = 2, wrook = 3, wbishop = 4, wknight = 5, wpawn = 6, 
 * bking = 7, bqueen = 8, brook = 9, bbishop = 10, bknight = 11, bpawn = 12, 
 * Squares are *A1 = 0, B1 = 1 ... H8 = 63 
 * Input format : 
 * piece[0] is white king, square[0] is its location 
 * piece[1] is black king, square[1] is its location *..
 * piece[x], square[x] can be in any order *..
 * piece[n + 1] is set to 0 to represent end of array 
 * Returns 
 * Score relative to side to move in approximate centi-pawns 
 */
int AlphaBetaEval::eval(Position& position) {
  ChessBoard board = position.GetWhiteBoard();
  BitBoard white_pieces = board.ours();
  BitBoard black_pieces = board.theirs();

  int const piece_total = white_pieces.count_few() + black_pieces.count_few();

  int player;   /** Side to move: white=0 black=1 */
  int *pieces{new int[piece_total + 1]}; /** Array of pieces */
  int* squares{new int[piece_total + 1]}; /** Corresponding array of squares each piece stands on */ 
  int i = 0;
  
  player = position.IsBlackToMove() ? 1 : 0;

  uint64_t kings = board.kings().as_int();
  getPiecePositions(kings & white_pieces, i, pieces, squares, 1);
  getPiecePositions(kings & black_pieces, i, pieces, squares, 7);
  uint64_t queens = board.queens().as_int();
  getPiecePositions(queens & white_pieces, i, pieces, squares, 2);
  getPiecePositions(queens & black_pieces, i, pieces, squares, 8);
  uint64_t rooks = board.rooks().as_int();
  getPiecePositions(rooks & white_pieces, i, pieces, squares, 3);
  getPiecePositions(rooks & black_pieces, i, pieces, squares, 9);
  uint64_t bishops = board.bishops().as_int();
  getPiecePositions(bishops & white_pieces, i, pieces, squares, 4);
  getPiecePositions(bishops & black_pieces, i, pieces, squares, 10);
  uint64_t knights = board.knights().as_int();
  getPiecePositions(knights & white_pieces, i, pieces, squares, 5);
  getPiecePositions(knights & black_pieces, i, pieces, squares, 11);
  uint64_t pawns = board.pawns().as_int();
  getPiecePositions(pawns & white_pieces, i, pieces, squares, 6);
  getPiecePositions(pawns & black_pieces, i, pieces, squares, 12);
  pieces[i] = 0;
  int eval = nnue_evaluate(player, pieces, squares);

  delete[] pieces;
  delete[] squares;
  return eval;
};
void AlphaBetaEval::getPiecePositions(BitBoard one_color_position, int &array_index, int* pieces, int* squares, int pieceType) {

    if (!one_color_position.empty()) {
      int piece_index = 0;
      int piece_total = one_color_position.count_few();
      
      uint64_t pos = one_color_position.as_int();
      while (piece_index < piece_total) {
        unsigned long index;
        unsigned char position_index = _BitScanForward64(&index, pos);
        squares[array_index] = index; //nnue position starts at position 1.
        pieces[array_index++] = pieceType;
        pos &= pos - 1;
        piece_index++;
      };
    };
};
}  // namespace lczero