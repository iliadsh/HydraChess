#include "serialize.hpp"

namespace hydra {
    torch::Tensor serialize(libchess::Position pos) {
        uint8_t data[12 * 64 + 4] = {0};

        //vlip so P1 = white
        if (pos.side_to_move() == libchess::constants::BLACK) pos.vflip();

        //loop through each board square
        for (int i = 0; i < 64; i++) {
            //loop through each piece
            for (libchess::Color color = libchess::constants::WHITE; color <= libchess::constants::BLACK; color++) {
                for (libchess::PieceType piece = libchess::constants::PAWN; piece <= libchess::constants::KING; piece++) {
                    auto bitboard = pos.piece_type_bb(piece, color);
                    //set blob location
                    if (bitboard & libchess::Bitboard(i)) data[(color * 6 + piece) * 64 + i] = 1;
                }
            }
        }

        //set castling rights
        if (pos.castling_rights().is_allowed(libchess::constants::WHITE_KINGSIDE)) data[12 * 64 + 0] = 1;
        if (pos.castling_rights().is_allowed(libchess::constants::WHITE_QUEENSIDE)) data[12 * 64 + 1] = 1;
        if (pos.castling_rights().is_allowed(libchess::constants::BLACK_KINGSIDE)) data[12 * 64 + 2] = 1;
        if (pos.castling_rights().is_allowed(libchess::constants::BLACK_QUEENSIDE)) data[12 * 64 + 3] = 1;

        //generate tensor
        return torch::from_blob(data, {12 * 64 + 4}, at::kByte).to(at::kFloat).clone();
    }
}