#include "search.hpp"
#include "config.hpp"
#include "neural.hpp"
#include "serialize.hpp"
#include "train.hpp"
#include <UCIService.h>

using namespace hydra;

libchess::Position global_pos{ libchess::constants::STARTPOS_FEN };
libchess::UCIService uci(config::ENGINE_NAME, config::ENGINE_AUTHOR);

bool stopped = false;

/**
 * Handle GO events by the UCI service.
 */
void handle_go(const libchess::UCIGoParameters& params) {
    stopped = false;
    int predicted_score{0};
    libchess::Move chosen_move = choose_best_move(global_pos, search_root, stopped, predicted_score);

    //send info to gui
    libchess::UCIInfoParameters info_params;
    info_params.set_score(libchess::UCIScore{ predicted_score, libchess::UCIScore::ScoreType::CENTIPAWNS });
    uci.info(info_params);
    uci.bestmove(chosen_move.to_str());
}

/**
 * Handle STOP events by the UCI service.
 */
void handle_stop() {
    stopped = true;
}

/**
 * Handles POSITION events by the UCI service. If the opponents chosen move is in the search tree (which is very likely, especially as the game goes on),
 * it will transfer down the search tree to that node. Otherwise, it will create a new search tree.
 */
void handle_position(const libchess::UCIPositionParameters& params) {
    global_pos = *libchess::Position::from_fen(params.fen());
    if (params.move_list().has_value()) {
        auto move_list = (*params.move_list()).move_list();
        for (const auto& move : move_list) {
            global_pos.make_move(*libchess::Move::from(move));
        }
        if (global_pos.previous_move().has_value()) {
            libchess::Move last_move = *global_pos.previous_move(); //get last move and possibly move tree down
            auto&& next_node = search_root->children.find(last_move.value_sans_type());
            if (next_node != search_root->children.end()) {
                search_root = std::move(next_node->second);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc > 2 && strcmp(argv[1], "-train") == 0) {
        train(value_net, argv[2]);
    } 
    else {
        //torch::load(value_net, /*config::WEIGHTS_PATH*/"D:\\testcpp\\data\\model - Copy.pt");
        value_net->to(at::kCUDA);
        //std::cout << value_net << std::endl;
        libchess::Position pos{"4r3/1P6/P7/2Nb3p/1K4k1/3R2P1/8/8 b - - 2 98"};
        auto t = serialize(pos).unsqueeze(0);
        float val = value_net->forward(t.to(at::kCUDA))[0][0].item<float>();
        std::cout << val << std::endl;
        //uci.register_position_handler(handle_position);
        //uci.register_go_handler(handle_go);
        //uci.register_stop_handler(handle_stop);
        //uci.run();
    }    
    return 0;
}