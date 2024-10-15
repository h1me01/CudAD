#pragma once

#include "bitboard.h"
#include "piece.h"
#include "position.h"
#include "square.h"

#include <cmath>
#include <algorithm>

inline int getMaterialCount(Position position) {
    int num_pawns = 0;
    int num_knights = 0;
    int num_bishops = 0;
    int num_rooks = 0;
    int num_queens = 0;

    for (int i = 0; i < 64; i++) {
        if (getBit(position.m_occupancy, i)) {
            Piece piece = position.getPiece(i);
            switch (getPieceType(piece)) {
                case PAWN: num_pawns++; break;
                case KNIGHT: num_knights++; break;
                case BISHOP: num_bishops++; break;
                case ROOK: num_rooks++; break;
                case QUEEN: num_queens++; break;
                default: break;
            }            
        }
    }

    return num_pawns + 3*num_knights + 3*num_bishops + 5*num_rooks + 9*num_queens;
}

struct WinRateParams {
    double a;
    double b;
};

inline WinRateParams win_rate_params(const Position& pos) {
    int material = getMaterialCount(pos);

    // The fitted model only uses data for material counts in [17, 78], and is anchored at count 58.
    double m = std::clamp(material, 17, 78) / 58.0;

    // Return a = p_a(material) and b = p_b(material), see github.com/official-stockfish/WDL_model
    constexpr double as[] = {-37.45051876, 121.19101539, -132.78783573, 420.70576692};
    constexpr double bs[] = {90.26261072, -137.26549898, 71.10130540, 51.35259597};

    double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
    double b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];

    return {a, b};
}

inline int win_rate_model(int16_t v, const Position& pos) {
    auto [a, b] = win_rate_params(pos);

    // Return the win rate in per mille units, rounded to the nearest integer.
    return int(0.5 + 1000 / (1 + std::exp((a - double(v)) / b)));
}

inline int getWdl(int16_t v, const Position& pos) {
    int wdl_w = win_rate_model(v, pos);
    int wdl_l = win_rate_model(-v, pos);
    int wdl_d = 1000 - wdl_w - wdl_l;

    if (wdl_d > wdl_w && wdl_d > wdl_l) {
        return 0;
    } else if (wdl_w > wdl_l) {
        return 1;
    } else {
        return -1;
    }

    return wdl_w;
}