#include "search.hpp"
#include "config.hpp"
#include "neural.hpp"
#include "serialize.hpp"
#include "train.hpp"
#include <UCIService.h>

using namespace hydra;

libchess::Position global_pos{ libchess::constants::STARTPOS_FEN };
libchess::UCIService uci(config::ENGINE_NAME, config::ENGINE_AUTHOR);
MCTSearch mcts;

bool stopped = false;

/**
 * Handle GO events by the UCI service.
 */
void handle_go(const libchess::UCIGoParameters& params) {
    stopped = false;
    int predicted_score{0};
    libchess::Move chosen_move = mcts.choose_best_move(global_pos, stopped, predicted_score);

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
            mcts.shift_tree_down(last_move.value_sans_type());
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc > 2 && strcmp(argv[1], "-train") == 0) {
        Eval evaluator;
        train(evaluator, argv[2]);
    } 
    else {
        uci.register_position_handler(handle_position);
        uci.register_go_handler(handle_go);
        uci.register_stop_handler(handle_stop);
        uci.run();
    }    
    return 0;
}