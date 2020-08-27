#include "serialize.hpp"

namespace hydra {
    torch::Tensor serialize(libchess::Position pos) {
        //6 - P1 pieces PNBRQK
        //6 - P2 pieces pnbrqk
        //4 - castling rights? KQkq
        //1 - in check?
        //1 - attacked squares
        //total: 18

        //data blob
        uint8_t data[18 * 8 * 8] = {0};

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
        if (pos.castling_rights().is_allowed(libchess::constants::WHITE_KINGSIDE)) memset(data + (12 * 8 * 8), 1, 8 * 8);
        if (pos.castling_rights().is_allowed(libchess::constants::WHITE_QUEENSIDE)) memset(data + (13 * 8 * 8), 1, 8 * 8);
        if (pos.castling_rights().is_allowed(libchess::constants::BLACK_KINGSIDE)) memset(data + (14 * 8 * 8), 1, 8 * 8);
        if (pos.castling_rights().is_allowed(libchess::constants::BLACK_QUEENSIDE)) memset(data + (15 * 8 * 8), 1, 8 * 8);

        //set in check bits
        if (pos.in_check()) memset(data + (16 * 8 * 8), 1, 8 * 8);

        //set attacked squares
        for (int i = 0; i < 64; i++) {
            auto piece = pos.piece_on(libchess::Square(i));
            if (piece.has_value()) {
                auto other_side = piece->color() == libchess::constants::WHITE ? libchess::constants::BLACK : libchess::constants::WHITE;
                if (pos.attackers_to(libchess::Square(i), other_side) != 0) data[(17 * 8 * 8) + i] = 1;
            }
        }

        //generate tensor
        return torch::from_blob(data, {18, 8, 8}, at::kByte).to(at::kFloat).clone();
    }
}